/*
 * codeat-forward.cc
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Nnet feed-forward with codes.
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-codeat.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
        "Perform forward pass through Neural Network with codes.\n"
            "\n"
            "Usage:  codeat-forward [options] <adapt-model-in> <back-model-in>"
            " <feature-rspecifier> <utt2set-map> <code-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " codeat-forward adapt.nnet back.nnet ark:features.ark ark:utt2set.ark ark:code.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register(
        "no-softmax",
        &no_softmax,
        "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id = 0;
    po.Register("use-gpu-id", &use_gpu_id,
                "Unused, kaldi is compiled w/o CUDA");
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string adapt_model_filename = po.GetArg(1),
        back_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        utt2set_rspecifier = po.GetArg(4),
        code_rspecifier = po.GetArg(5),
        feature_wspecifier = po.GetArg(6);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf, nnet_back;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    nnet_back.Read(back_model_filename);

    Nnet nnet;
    nnet.Read(adapt_model_filename);
    //optionally remove softmax
    if (no_softmax
        && nnet_back.GetComponent(nnet_back.NumComponents() - 1).GetType()
            == Component::kSoftmax) {
      KALDI_LOG<< "Removing softmax from the nnet " << back_model_filename;
      nnet_back.RemoveComponent(nnet_back.NumComponents()-1);
    }
      //check for some non-sense option combinations
    if (apply_log && no_softmax) {
      KALDI_ERR<< "Nonsense option combination : --apply-log=true and --no-softmax=true";
    }
    if (apply_log
        && nnet_back.GetComponent(nnet_back.NumComponents() - 1).GetType()
            != Component::kSoftmax) {
      KALDI_ERR<< "Used --apply-log=true, but nnet " << back_model_filename
      << " does not have <softmax> as last component!";
    }

    PdfPrior pdf_prior(prior_opts);
    if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
      KALDI_ERR<< "Option --class-frame-counts has to be used together with "
      << "--no-softmax or --apply-log";
    }

      /*
       * Find out all the <codeat> layers
       */
    int32 num_codeat = 0, code_dim = 0;
    std::vector<CodeAT*> layers_codeat;
    for (int32 c = 0; c < nnet.NumComponents(); ++c) {
      if (nnet.GetComponent(c).GetType() == Component::kCodeAT) {
        layers_codeat.push_back(dynamic_cast<CodeAT*>(&(nnet.GetComponent(c))));
        if (code_dim == 0) {
          code_dim = layers_codeat[num_codeat]->GetCodeDim();
        } else if (code_dim != layers_codeat[num_codeat]->GetCodeDim()) {
          KALDI_ERR<< "Inconsistent code vector dimension for different <codeat> layers";
        }
        ++num_codeat;
      }
    }
    KALDI_LOG<< "Totally " << num_codeat << " among " << nnet.NumComponents() << " layers of the nnet are <codeat> layers";

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    RandomAccessTokenReader utt2set_reader(utt2set_rspecifier);
    RandomAccessBaseFloatVectorReader code_reader(code_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out, back_out;
    Matrix<BaseFloat> back_out_host;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    std::string prev_setkey = "";
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      KALDI_VLOG(2) << "Processing utterance " << num_done + 1
          << ", " << feature_reader.Key()
          << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val) {
            KALDI_ERR<< "NaN in features of : " << feature_reader.Key();
          }
          if (val == std::numeric_limits < BaseFloat > ::infinity()) {
            KALDI_ERR<< "inf in features of : " << feature_reader.Key();
          }
        }
      }

      if (!utt2set_reader.HasKey(key)) {
        KALDI_ERR<< "Cannot find set key for " << key;
      }
      std::string setkey = utt2set_reader.Value(key);
      if (setkey != prev_setkey) {
        // update the code vector
        if (!code_reader.HasKey(setkey)) {
          KALDI_ERR<< "No code for set " << setkey;
        }
        const Vector<BaseFloat> &code = code_reader.Value(setkey);
        for(int32 c=0; c<layers_codeat.size(); ++c) {
          layers_codeat[c]->SetCode(code);
        }
        prev_setkey = setkey;
      }

          // push it to gpu
      feats = mat;
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);
      nnet_back.Feedforward(nnet_out, &back_out);

      // convert posteriors to log-posteriors
      if (apply_log) {
        back_out.ApplyLog();
      }

      // subtract log-priors from log-posteriors to get quasi-likelihoods
      if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
        pdf_prior.SubtractOnLogpost(&back_out);
      }

      //download from GPU
      back_out_host.Resize(back_out.NumRows(), back_out.NumCols());
      back_out.CopyToMat(&back_out_host);

      //check for NaN/inf
      for (int32 r = 0; r < back_out_host.NumRows(); r++) {
        for (int32 c = 0; c < back_out_host.NumCols(); c++) {
          BaseFloat val = back_out_host(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << feature_reader.Key();
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << feature_reader.Key();
          }
        }

            // write
      feature_writer.Write(feature_reader.Key(), back_out_host);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
            << time_now / 60 << " min; processed " << tot_t / time_now
            << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    KALDI_LOG<< "Done " << num_done << " files"
    << " in " << time.Elapsed()/60 << "min,"
    << " (fps " << tot_t/time.Elapsed() << ")";

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0)
      return -1;
    return 0;
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

