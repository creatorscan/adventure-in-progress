/*
 * codemat-forward.cc
 *
 *  Created on: Nov 8, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Nnet feed-forward with code matrices.
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
        "Perform forward pass through Neural Network with code matrices.\n"
            "\n"
            "Usage:  codemat-forward [options] <model-front> <model-code> <model-back>"
            " <feature-rspecifier> <code-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " codemat-forward front.nnet code.nnet back.nnet ark:features.ark ark:code.ark ark:mlpoutput.ark\n";

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

    std::string front_model_filename = po.GetArg(1),
      code_model_filename = po.GetArg(2),
      back_model_filename = po.GetArg(3),
      feature_rspecifier = po.GetArg(4),
      code_rspecifier = po.GetArg(5),
      feature_wspecifier = po.GetArg(6);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet_front, nnet_code, nnet_back;
    nnet_front.Read(front_model_filename);
    nnet_code.Read(code_model_filename);
    nnet_back.Read(back_model_filename);

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

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader code_reader(code_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, front_out, nnet_out;
    CuMatrix<BaseFloat> codes, code_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer time;
    double time_now = 0;
    int32 num_done = 0, num_mismatch = 0;
    std::string prev_setkey = "";
    // iterate over all feature files
    for (; !feature_reader.Done() && !code_reader.Done(); feature_reader.Next(), code_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      std::string code_key = code_reader.Key();

      if(key!=code_key) {
        KALDI_WARN << "Key mismatch: featrure(" << key << ") vs. code(" << code_key << ")";
        ++num_mismatch;
        continue;
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Matrix<BaseFloat> &code_mat = code_reader.Value();
      if(mat.NumRows() != code_mat.NumRows()) {
        KALDI_WARN << "Mismatched number of frames in feature(" << mat.NumRows() << ") and code(" << code_mat.NumRows() << ") for " << key;
        ++num_mismatch;
      }

      KALDI_VLOG(2) << "Processing utterance " << num_done + 1
          << ", " << key << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val) {
            KALDI_ERR<< "NaN in features of : " << key;
          }
          if (val == std::numeric_limits < BaseFloat > ::infinity()) {
            KALDI_ERR<< "inf in features of : " << key;
          }
        }
        for (int32 c = 0; c < code_mat.NumCols(); c++) {
          BaseFloat val = code_mat(r, c);
          if (val != val) {
            KALDI_ERR<< "NaN in codes of : " << code_key;
          }
          if (val == std::numeric_limits < BaseFloat > ::infinity()) {
            KALDI_ERR<< "inf in codes of : " << code_key;
          }
        }
      }

      // push it to gpu
      feats = mat;
      codes = code_mat;

      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet_front.Feedforward(feats_transf, &front_out);

      nnet_code.Feedforward(codes, &code_out);
      front_out.AddMat(1.0, code_out, 1.0);

      nnet_back.Feedforward(front_out, &nnet_out);

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }

      // subtract log-priors from log-posteriors to get quasi-likelihoods
      if (prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
        pdf_prior.SubtractOnLogpost(&nnet_out);
      }

      //download from GPU
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);

      //check for NaN/inf
      for (int32 r = 0; r < nnet_out_host.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << key;
          }
        }

      // write
      feature_writer.Write(key, nnet_out_host);

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

