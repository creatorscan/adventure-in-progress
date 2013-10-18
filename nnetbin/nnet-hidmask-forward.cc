// nnetbin/nnet-hidmask-forward.cc

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
            "Usage:  nnet-forward [options] <frontend-model-in> <backend-model-in>"
            " <mask-model-in> <feature-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " nnet-hidmask-forward front.nnet back.nnet mask.nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);
    bool binarize_mask = false;
    po.Register("binarize-mask", &binarize_mask,
                "Binarize the hidden mask or not");

    BaseFloat binarize_threshold = 0.5;
    po.Register("binarize-threshold", &binarize_threshold,
                "Threshold value to binarize mask");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts,
                "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register(
        "prior-scale",
        &prior_scale,
        "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register(
        "no-softmax",
        &no_softmax,
        "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string frontend_model_filename = po.GetArg(1),
        backend_model_filename = po.GetArg(2),
        mask_model_filename = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        feature_wspecifier = po.GetArg(5);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_front, nnet_back, nnet_mask;
    nnet_front.Read(frontend_model_filename);
    nnet_back.Read(backend_model_filename);
    nnet_mask.Read(mask_model_filename);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, hidmask, front_out, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if (class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();

      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0 / sum);
      if (apply_log || no_softmax) {
        tmp_priors.ApplyLog();
        tmp_priors.Scale(-prior_scale);
      } else {
        tmp_priors.ApplyPow(-prior_scale);
      }

      // push priors to GPU
      priors.CopyFromVec(tmp_priors);
    }

    Timer tim;
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in features of : " << feature_reader.Key();
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in features of : " << feature_reader.Key();
          }
        }
            // push it to gpu
      feats.CopyFromMat(mat);
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet_front.Feedforward(feats_transf, &front_out);

      // generate mask
      nnet_mask.Feedforward(feats_transf, &hidmask);
      if(binarize_mask){
        hidmask.Binarize(binarize_threshold);
      }

      // apply mask
      front_out.MulElements(hidmask);
      
      // forward throught backend nnet
      nnet_back.Feedforward(front_out, &nnet_out);

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }

      // divide posteriors by priors to get quasi-likelihoods
      if (class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out.MulColsVec(priors);
        }
      }

      //download from GPU 
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << feature_reader.Key();
            /*if(val == kBaseLogZero)
             nnet_out_host(r,c) = -1e10;*/
          }
        }
            // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 1000 == 0) {
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed();
    if (!silent)
      KALDI_LOG<< "Done " << num_done << " files";

#if HAVE_CUDA==1
      if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
