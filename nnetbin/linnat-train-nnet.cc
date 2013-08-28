// nnetbin/linnat-train-nnet.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Adaptive training of the nnet model in the LIN system.\n"
            "Usage:  linnat-train-nnet [options] <model-in> <lin-weight-rsepcifier> <lin-bias-rspecifier>"
            " <feature-rspecifier> <alignments-rspecifier> [<model-out>]\n"
            "e.g.: \n"
            " linnat-train-nnet nnet.init ark:weights.ark ark:bias.ark scp:train.scp ark:train.ali nnet.iter1\n";

    ParseOptions po(usage);
    bool binary = false,
        crossvalidate = false,
        randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promote sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad,
                "Whether to average the gradient in the bunch");

    std::string learn_factors = "";
    po.Register(
        "learn-factors",
        &learn_factors,
        "Learning factor for each updatable layer, separated by ',' and work together with learn rate");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    std::string utt2xform;
    po.Register("utt2xform", &utt2xform, "Utterance to LIN xform mapping");

    po.Read(argc, argv);

    if (po.NumArgs() != 6 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        weight_rspecifier = po.GetArg(2),
        bias_rspecifier = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        alignments_rspecifier = po.GetArg(5);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(6);
    }

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    if (nnet.Layer(0)->GetType() != Component::kLinBL) {
      KALDI_ERR<< "The first layer is not <linbl> layer!";
    }
    LinBL *lin = static_cast<LinBL*>(nnet.Layer(0));

    if (learn_factors == "") {
      nnet.SetLearnRate(learn_rate, NULL);
    } else {
      nnet.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet.SetMomentum(momentum);
    nnet.SetL2Penalty(l2_penalty);
    nnet.SetL1Penalty(l1_penalty);
    nnet.SetAverageGrad(average_grad);

    // ensure the input LIN layer is not updated
    lin->SetLearnRate(0.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    RandomAccessTokenReader utt2xform_reader(utt2xform);

    RandomAccessBaseFloatMatrixReader weight_reader(weight_rspecifier);
    RandomAccessBaseFloatVectorReader bias_reader(bias_rspecifier);

    Cache cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    CuMatrix<BaseFloat> feats, feats_transf, feats_lin, nnet_in, nnet_out, glob_err;
    std::vector<int32> targets;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0,
        num_cache = 0;
    std::string cur_lin="", new_lin="";

    while (1) {
      // fill the cache
      while (!cache.Full() && !feature_reader.Done()) {
        std::string key = feature_reader.Key();

        if (utt2xform == "") {
          new_lin = key;
        } else {
          if (!utt2xform_reader.HasKey(key)) {
            KALDI_ERR<< "No mapping found for utterance " << key;
          }
          new_lin=utt2xform_reader.Value(key);
        }

        if (!weight_reader.HasKey(new_lin) || !bias_reader.HasKey(new_lin)) {
          KALDI_ERR<< "No LIN weight or bias for the input feature " << new_lin;
        }

        // update the LIN xform when necessary
        if (new_lin != cur_lin) {
          const Matrix<BaseFloat> &weight = weight_reader.Value(new_lin);
          const Vector<BaseFloat> &bias = bias_reader.Value(new_lin);

          lin->SetLinearityWeight(weight, false);
          lin->SetBiasWeight(bias);

          cur_lin = new_lin;
        }

        if (!alignments_reader.HasKey(key)) {
          num_no_alignment++;
        } else {
          // get feature alignment pair
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          const std::vector<int32> &alignment = alignments_reader.Value(key);
          // chech for dimension
          if ((int32) alignment.size() != mat.NumRows()) {
            KALDI_WARN<< "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
            num_other_error++;
            continue;
          }
            // push features to GPU
          feats.CopyFromMat(mat);
          // possibly apply transform
          nnet_transf.Feedforward(feats, &feats_transf);
          lin->Propagate(feats_transf, &feats_lin);
          // add to cache
          cache.AddData(feats_lin, alignment);
          num_done++;
        }
        Timer t_features;
        feature_reader.Next();
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!crossvalidate && randomize) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
          << (cache.Randomized() ? "[RND]" : "[NO-RND]")
          << " segments: " << num_done
          << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of feature/target pairs
        cache.GetBunch(&nnet_in, &targets);
        // train 
        nnet.Propagate(nnet_in, &nnet_out);
        xent.EvalVec(nnet_out, targets, &glob_err);
        if (!crossvalidate) {
          nnet.Backpropagate(glob_err, NULL);
        }
        tot_t += nnet_in.NumRows();
      }

      // stop training when no more data
      if (feature_reader.Done())
        break;
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";

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
