/*
 * vtsbin/dbnvts2-forward.cc
 *
 *  Created on: Nov 9, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Converting the discriminative logistic linear regression into Naive
 *  Bayes based generative classifier.
 *
 *  The input features are original MFCC without normalization, normalization
 *  is taken into consideration when generating the pos and neg models.
 *
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate the new decision boundaries and perform forward pass through Neural Network.\n"
            "Usage:  dbnvts2-forward [options] <pos-model-in> <neg-model-in>"
            " <back-nnet-in> <pos2neg-prior-in> <var-scale-in> <llr-scale-in> <feature-rspecifier>"
            " <noise-model-rspecifier> <feature-wspecifier>\n"
            "e.g.: \n"
            " dbnvts2-forward pos.mdl neg.mdl nnet.back pos2neg.stats var_scale.stats llr_scale.stats"
            "ark:features.ark ark:noise_params.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    bool shared_var = true;
    po.Register(
        "shared-var",
        &shared_var,
        "Whether after compensation the variances are constrained to be the same");

    BaseFloat positive_var_weight = 1.0;
    po.Register(
        "positive-var-weight",
        &positive_var_weight,
        "Only effective when shared-var=true, the weight ratio for positive variance");

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

    bool post_var_scale = true;
    po.Register(
        "post-var-scale", &post_var_scale,
        "Whether the variance scaling is applied before compensation or after");

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
        .GetArg(6), feature_rspecifier = po.GetArg(7), noise_params_rspecifier =
        po.GetArg(8), feature_wspecifier = po.GetArg(9);

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

    // noisy models
    AmDiagGmm pos_noise_am, neg_noise_am;

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

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);
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

      // read noise parameters
      if (!noiseparams_reader.HasKey(key + "_mu_h")
          || !noiseparams_reader.HasKey(key + "_mu_z")
          || !noiseparams_reader.HasKey(key + "_var_z")) {
        KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
      }
      Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
      Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
      Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));
      if (g_kaldi_verbose_level >= 1) {
        KALDI_LOG<< "Additive Noise Mean: " << mu_z;
        KALDI_LOG << "Additive Noise Covariance: " << var_z;
        KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
      }

      // compensate the postive and negative gmm models
      pos_noise_am.CopyFromAmDiagGmm(pos_am_gmm);
      neg_noise_am.CopyFromAmDiagGmm(neg_am_gmm);

      if(use_var_scale && !post_var_scale){
        ScaleVariance(var_scale, pos_noise_am, neg_noise_am);
      }

      CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                              num_fbank, dct_mat, inv_dct_mat, num_frames,
                              pos_noise_am);


      CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                              num_fbank, dct_mat, inv_dct_mat, num_frames,
                              neg_noise_am);

      if (shared_var) {
        // set the covariance to be the same for pos and neg
        InterpolateVariance(positive_var_weight, pos_noise_am, neg_noise_am);
      }

      // post compensation scaling
      if (use_var_scale && post_var_scale) {
        ScaleVariance(var_scale, pos_noise_am, neg_noise_am);
      }

      // forward through the new generative front end
      Matrix<BaseFloat> mat(feat.NumRows(), pos_noise_am.NumPdfs(), kSetZero);
      ComputeGaussianLogLikelihoodRatio(feat, pos_noise_am, neg_noise_am,
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

