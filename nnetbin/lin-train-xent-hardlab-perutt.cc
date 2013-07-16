// nnetbin/lin-train-xent-hardlab-perutt.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Train LinBL for each utterance.\n"
            "Usage: lin-train-xent-hardlab-perutt [options] <model-in> <feature-rspecifier> <alignments-rspecifier> [<model-out-wspecifier>]\n"
            "e.g.: \n"
            " lin-train-xent-hardlab-perutt nnet.init scp:train.scp ark:train.ali ark:lin.iter1\n";

    ParseOptions po(usage);
    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    std::string init_weight = "", init_bias = "";
    po.Register("init-weight", &init_weight,
                "Initial weight matrices for LIN xform");
    po.Register("init-bias", &init_bias, "Initial biases for LIN xform");

    po.Read(argc, argv);

    if (po.NumArgs() != 5 - (crossvalidate ? 2 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3);

    std::string target_weight_wspecifier, target_bias_wspecifier;
    if (!crossvalidate) {
      target_weight_wspecifier = po.GetArg(4);
      target_bias_wspecifier = po.GetArg(5);
    }

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    // construct the learn factors, only learn the first LIN layer
    std::string learn_factors = "1";
    for (int32 i = 1; i < nnet.LayerCount(); ++i) {
      if (nnet.Layer(i)->IsUpdatable())
        learn_factors += ",0";
    }

    nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);

    if (nnet.Layer(0)->GetType() != Component::kLinBL) {
      KALDI_ERR<< "The first layer is not <linbl> layer!";
    }
    LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    RandomAccessBaseFloatMatrixReader init_weight_reader(init_weight);
    RandomAccessBaseFloatVectorReader init_bias_reader(init_bias);

    BaseFloatMatrixWriter weight_writer(target_weight_wspecifier);
    BaseFloatVectorWriter bias_writer(target_bias_wspecifier);

    Xent xent;

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, glob_err;

    Matrix<BaseFloat> lin_weight;
    Vector<BaseFloat> lin_bias;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0,
        num_use_init = 0, num_use_iden = 0;
    for (; !feature_reader.Done(); /*feature_reader.Next()*/) {
      std::string key = feature_reader.Key();

      if (!alignments_reader.HasKey(key)) {
        num_no_alignment++;
      } else {

        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(key);

        // std::cout << mat;

        if ((int32) alignment.size() != mat.NumRows()) {
          KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        if (num_done % 10000 == 0)
          std::cout << num_done << ", " << std::flush;
        num_done++;

        // configure LIN
        if (!crossvalidate) {
          if (init_weight != "" && init_bias != ""
              && init_weight_reader.HasKey(key)
              && init_bias_reader.HasKey(key)) {
            const Matrix<BaseFloat> init_wt = init_weight_reader.Value(key);
            const Vector<BaseFloat> init_bs = init_bias_reader.Value(key);

            lin->SetLinearityWeight(init_wt, kNoTrans);
            lin->SetBiasWeight(init_bs);
            num_use_init++;

          } else {
            lin->SetToIdentity();
            num_use_iden++;
          }
        }

        // push features to GPU
        feats.CopyFromMat(mat);

        nnet_transf.Feedforward(feats, &feats_transf);
        nnet.Propagate(feats_transf, &nnet_out);

        xent.EvalVec(nnet_out, alignment, &glob_err);

        if (!crossvalidate) {
          nnet.Backpropagate(glob_err, NULL);

          (lin->GetLinearityWeight()).CopyToMat(&lin_weight);
          (lin->GetBiasWeight()).CopyToVec(&lin_bias);

          weight_writer.Write(key, lin_weight);
          bias_writer.Write(key, lin_bias);
        }

        tot_t += mat.NumRows();
      }

      Timer t_features;
      feature_reader.Next();
      time_next += t_features.Elapsed();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";
    KALDI_LOG<< num_use_init << " used initial LIN, " << num_use_iden << " used identity.";

    KALDI_LOG<< xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
