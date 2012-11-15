/*
 * vtsbin/dbnvts2-compute-llr.cc
 *
 *  Created on: Nov 15, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Compute the log likelihood ratios between positive and negative models.
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
        "Compute the log likelihood ratios between positive and negative models.\n"
            "Usage:  dbnvts2-compute-llr [options] <pos-model-in> <neg-model-in> <pos2neg-in>"
            " <feature-rspecifier> <llr-wspecifier>\n"
            "e.g.: \n"
            " dbnvts2-compute-llr pos.mdl neg.mdl pos2neg.stats ark:features.ark ark:llr.ark\n";

    ParseOptions po(usage);

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

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

    std::string noise_params_rspecifier = "";
    po.Register("noise-params", &noise_params_rspecifier,
                "Noise parameters for VTS compensation if available");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), pos2neg_prior_filename = po.GetArg(3), feature_rspecifier =
        po.GetArg(4), llr_wspecifier = po.GetArg(5);

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

    // log likelihood ratio scale factors, dummy one
    Vector<double> llr_scale;
    for (int32 i = 0; i < num_pdfs; ++i) {
      llr_scale(i) = 1.0;
    }

    bool do_vts = false;
    if (noise_params_rspecifier != "") {
      do_vts = true;
    }

    // noisy models
    AmDiagGmm pos_noise_am, neg_noise_am;

    Matrix<double> dct_mat, inv_dct_mat;
    if (do_vts) {
      GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                        &inv_dct_mat);
    }

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);
    BaseFloatMatrixWriter llr_writer(llr_wspecifier);

    Timer tim;
    KALDI_LOG<< "LLR COMPUTATION STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read the features
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feat = feature_reader.Value();

      // forward through the new generative front end
      Matrix<BaseFloat> mat(feat.NumRows(), pos_noise_am.NumPdfs(), kSetZero);

      if (do_vts) {
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

        CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
            num_fbank, dct_mat, inv_dct_mat, num_frames,
            pos_noise_am);

        CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
            num_fbank, dct_mat, inv_dct_mat, num_frames,
            neg_noise_am);

        ComputeGaussianLogLikelihoodRatio_General(feat, pos_noise_am,
            neg_noise_am,
            pos2neg_log_prior_ratio,
            llr_scale, mat);
      } else {

        ComputeGaussianLogLikelihoodRatio_General(feat, pos_am_gmm,
            neg_am_gmm,
            pos2neg_log_prior_ratio,
            llr_scale, mat);
      }

          // write
      llr_writer.Write(key, mat);

      // progress log
      if (num_done % 100 == 0) {
        KALDI_LOG<< num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }

    // final message
    KALDI_LOG<< "LLR COMPUTATION FINISHED " << tim.Elapsed() << "s, fps"
    << tot_t / tim.Elapsed();
    KALDI_LOG<< "Done " << num_done << " files";

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

