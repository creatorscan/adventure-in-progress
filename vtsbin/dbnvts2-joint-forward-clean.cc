/*
 * vtsbin/dbnvts2-joint-forward-clean.cc
 *
 *  Created on: Nov 14, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Converting the discriminative logistic linear regression into Naive
 *  Bayes based generative classifier.
 *
 *  Jointly compensate both the global and component dependent parameters.
 *
 *  This takes only the clean feature, just for testing the conversion purpose.
 *
 *  The input to this tool must be the output of vts-apply-global-cmvn
 *
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm-normal.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate the new decision boundaries and perform forward pass through Neural Network.\n"
            "Usage:  dbnvts2-forward-clean [options] <pos-model-in> <neg-model-in>"
            " <back-nnet-in> <pos2neg-prior-in> <var-scale-in> <llr-scale-in> <global-cmvn-rspecifier> <feature-rspecifier>"
            " <feature-wspecifier>\n"
            "e.g.: \n"
            " dbnvts2-joint-forward-clean pos.mdl neg.mdl nnet.back pos2neg.stats var_scale.stats llr_scale.stats"
            " ark:global_cmvn.ark ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    int32 num_frames = 9;
    po.Register("num-frames", &num_frames,
                "Number of frames for the input feature");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");

    BaseFloat ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    bool use_var_scale = false;
    po.Register("use-var-scale", &use_var_scale,
                "Apply the variance scale to the new weight estimation");

    bool use_llr_scale = false;
    po.Register("use-llr-scale", &use_llr_scale,
                "Apply the LLR scale to the generative log likelihood ratio");

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

    if (po.NumArgs() != 9) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), nnet_filename = po.GetArg(3), pos2neg_prior_filename = po
        .GetArg(4), var_scale_filename = po.GetArg(5), llr_scale_filename = po
        .GetArg(6), cmvn_rspecifier = po.GetArg(7), feature_rspecifier = po
        .GetArg(8), feature_wspecifier = po.GetArg(9);

    // positive AM Gmm
    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_model_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }

    // negative AM Gmm
    AmDiagGmm neg_am_gmm;
    {
      bool binary;
      Input ki(neg_model_filename, &binary);
      neg_am_gmm.Read(ki.Stream(), binary);
    }

    KALDI_ASSERT(pos_am_gmm.NumPdfs() == neg_am_gmm.NumPdfs());
    int32 num_pdfs = pos_am_gmm.NumPdfs();

    RandomAccessDoubleMatrixReader cmvn_reader(cmvn_rspecifier);

    if (!cmvn_reader.HasKey("global")) {
      KALDI_ERR<< "No normalization statistics available for key "
      << "'global', producing no output for this utterance";
    }

    // read in the statistics
    const Matrix<double> &cmvn_stats = cmvn_reader.Value("global");
    // generate the mean and covariance from the statistics
    int32 feat_dim = cmvn_stats.NumCols() - 1;
    double counts = cmvn_stats(0, feat_dim);
    Vector<double> mean(feat_dim, kSetZero), var(feat_dim, kSetZero);
    for (int32 i = 0; i < feat_dim; ++i) {
      mean(i) = cmvn_stats(0, i) / counts;
      var(i) = (cmvn_stats(1, i) / counts) - mean(i) * mean(i);
    }
    Vector<double> expand_mean(pos_am_gmm.Dim()), expand_std(pos_am_gmm.Dim());
    for(int32 i=0; i<pos_am_gmm.Dim(); ++i){
      expand_mean(i) = mean(i%feat_dim);
      expand_std(i) = sqrt(var(i%feat_dim));
    }
    GmmToNormalizedGmm(expand_mean, expand_std, pos_am_gmm);
    GmmToNormalizedGmm(expand_mean, expand_std, neg_am_gmm);

    Nnet nnet_back;
    nnet_back.Read(nnet_filename);

    // positive to negative prior ratio
    Vector<double> pos2neg_log_prior_ratio(num_pdfs, kSetZero);
    Matrix<double> prior_stats;
    {
      bool binary;
      Input ki(pos2neg_prior_filename, &binary);
      prior_stats.Read(ki.Stream(), binary);
      KALDI_ASSERT(
          prior_stats.NumRows()==2 && prior_stats.NumCols()==num_pdfs);
    }
    for (int32 i = 0; i < num_pdfs; ++i) {
      pos2neg_log_prior_ratio(i) = log(prior_stats(0, i) / prior_stats(1, i));
    }

    // variance scale factors
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == num_pdfs);
    }

    if (use_var_scale) {
      ScaleVariance(var_scale, pos_am_gmm, neg_am_gmm);
    }

    // log likelihood ratio scale factors
    Vector<double> llr_scale;
    {
      bool binary;
      Input ki(llr_scale_filename, &binary);
      llr_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(llr_scale.Dim() == num_pdfs);
    }
    if (!use_llr_scale) {
      for (int32 i = 0; i < num_pdfs; ++i) {
        llr_scale(i) = 1.0;
      }
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feat_dev, nnet_out_dev;
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
      KALDI_LOG<< "DBNVTS FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read the features
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feat = feature_reader.Value();

      // forward through the new generative front end
      Matrix<BaseFloat> mat(feat.NumRows(), num_pdfs, kSetZero);
      ComputeGaussianLogLikelihoodRatio(feat, pos_am_gmm, neg_am_gmm,
                                        pos2neg_log_prior_ratio, llr_scale,
                                        mat);

      //check for NaN/inf
      for (int32 r = 0; r < mat.NumRows(); r++) {
        for (int32 c = 0; c < mat.NumCols(); c++) {
          BaseFloat val = mat(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in features of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in features of : " << key;
          }
        }
            // push it to gpu
      feat_dev.CopyFromMat(mat);
      // fwd-pass
      nnet_back.Feedforward(feat_dev, &nnet_out_dev);

      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out_dev.ApplyLog();
      }

      // divide posteriors by priors to get quasi-likelihoods
      if (class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out_dev.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out_dev.MulColsVec(priors);
        }
      }

      //download from GPU
      nnet_out_dev.CopyToMat(&nnet_out_host);
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
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD FINISHED " << tim.Elapsed() << "s, fps"
      << tot_t / tim.Elapsed();
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

