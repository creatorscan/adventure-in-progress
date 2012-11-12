/*
 * dbnvts2-test-gmm.cc
 *
 *  Created on: Nov 9, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Test whether the positive and negative GMM give the same
 *  decision as the hidden unit on unseen data.
 *
 */
#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Test whether the positive and negative Gmms give the same decision as the hidden unit.\n"
            "Usage:  dbnvts2-test-gmm [options] <pos-model-in> <neg-model-in> "
            "<pos2neg-log-prior-in> <var-scale-in> <bl-acts-rspecifier> <feature-rspecifier> <stats-out>\n"
            "e.g.: \n"
            " dbnvts2-test-gmm pos.mdl neg.modl pos2neg_log.stats var_scale.stats ark:bl_acts.ark ark:feat.ark error.stats\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write stats in binary mode");

    std::string noise_params_rspecifier;
    po.Register("noise-params", &noise_params_rspecifier,
                "rspecifier for per utterance VTS noise parameters");

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

    bool use_var_scale = false;
    po.Register("use-var-scale", &use_var_scale,
                "Apply the variance scale to the new weight estimation");

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

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), pos2neg_prior_filename = po.GetArg(3), var_scale_filename =
        po.GetArg(4), blacts_rspecifier = po.GetArg(5), feature_rspecifier = po
        .GetArg(6), stats_out_filename = po.GetArg(7);

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

    KALDI_ASSERT( pos_am_gmm.NumPdfs() == neg_am_gmm.NumPdfs());

    // positive to negative prior ratio
    Vector<double> pos2neg_log_prior_ratio(pos_am_gmm.NumPdfs(), kSetZero);
    {
      bool binary;
      Input ki(pos2neg_prior_filename, &binary);
      pos2neg_log_prior_ratio.Read(ki.Stream(), binary);
      KALDI_ASSERT( pos2neg_log_prior_ratio.Dim() == pos_am_gmm.NumPdfs());
    }

    // var-scale
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == pos_am_gmm.NumPdfs());
    }
    /*
    if (!use_var_scale) {
      for (int32 i = 0; i < var_scale.Dim(); ++i) {
        var_scale(i) = 1.0;
      }
    }*/

    AmDiagGmm noise_pos_am_gmm, noise_neg_am_gmm;

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);
    RandomAccessBaseFloatMatrixReader blacts_reader(blacts_rspecifier);

    int32 num_done = 0;
    /*
     * stats(0, :): nnet_act >= 0.0 && pos_llh >= neg_llh
     * stats(1, :): nnet_act >= 0.0 && pos_llh < neg_llh
     * stats(2, :): nnet_act < 0.0 && pos_llh >= neg_llh
     * stats(3, :): nnet_act < 0.0 && pos_llh < neg_llh
     */
    Matrix<BaseFloat> ori_stats(4, pos_am_gmm.NumPdfs(), kSetZero);
    Matrix<BaseFloat> vts_stats(4, pos_am_gmm.NumPdfs(), kSetZero);
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read the features
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feat = feature_reader.Value();

      // get the DBN layer activations
      if (!blacts_reader.HasKey(key)) {
        KALDI_ERR<< "DBN acts for key: " << key << " doesn't exist!";
      }
      const Matrix<BaseFloat> &mat = blacts_reader.Value(key);

      // forward through the new generative front end
      Matrix<BaseFloat> new_mat(feat.NumRows(), pos_am_gmm.NumPdfs(), kSetZero);
      ComputeGaussianLogLikelihoodRatio(feat, pos_am_gmm, neg_am_gmm,
                                        pos2neg_log_prior_ratio, var_scale,
                                        new_mat);

      // evaluate each frame
      for (int32 i = 0; i < feat.NumRows(); ++i) {
        for (int32 pdf = 0; pdf < pos_am_gmm.NumPdfs(); ++pdf) {

          if (mat(i, pdf) >= 0.0) {
            if (new_mat(i, pdf) >= 0.0) {
              ori_stats(0, pdf) += 1;
            } else {
              ori_stats(1, pdf) += 1;
            }
          } else {
            if (new_mat(i, pdf) >= 0.0) {
              ori_stats(2, pdf) += 1;
            } else {
              ori_stats(3, pdf) += 1;
            }
          }
        }
      }

      // compute after VTS stats
      if (noise_params_rspecifier != "") {
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
          KALDI_LOG << "Additive Noise Mean: " << mu_z;
          KALDI_LOG << "Additive Noise Covariance: " << var_z;
          KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
        }

        // compensate the postive and negative gmm models
        noise_pos_am_gmm.CopyFromAmDiagGmm(pos_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, true, num_cepstral, num_fbank,
            dct_mat, inv_dct_mat, num_frames,
            noise_pos_am_gmm);

        noise_neg_am_gmm.CopyFromAmDiagGmm(neg_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, true, num_cepstral, num_fbank,
            dct_mat, inv_dct_mat, num_frames,
            noise_neg_am_gmm);

        if (shared_var) {
          // set the covariance to be the same for pos and neg
          InterpolateVariance(positive_var_weight, noise_pos_am_gmm, noise_neg_am_gmm);
        }

        if(use_var_scale) {
         ScaleVariance(var_scale, noise_pos_am_gmm, noise_neg_am_gmm);
         }

        new_mat.SetZero();
        ComputeGaussianLogLikelihoodRatio(feat, noise_pos_am_gmm, noise_neg_am_gmm,
            pos2neg_log_prior_ratio, var_scale, new_mat);

        // evaluate each frame
        for (int32 i = 0; i < feat.NumRows(); ++i) {
          for (int32 pdf = 0; pdf < pos_am_gmm.NumPdfs(); ++pdf) {

            if (mat(i, pdf) >= 0.0) {
              if (new_mat(i, pdf) >= 0.0) {
                vts_stats(0, pdf) += 1;
              } else {
                vts_stats(1, pdf) += 1;
              }
            } else {
              if (new_mat(i,pdf) >= 0.0) {
                vts_stats(2, pdf) += 1;
              } else {
                vts_stats(3, pdf) += 1;
              }
            }
          }
        }

      }

          // progress log
      if (num_done % 100 == 0) {
        KALDI_LOG<< num_done << ", " << std::flush;
        {
          Output ko(stats_out_filename, binary);
          ori_stats.Write(ko.Stream(), binary);
          vts_stats.Write(ko.Stream(), binary);
        }
        KALDI_LOG<< "Write current stats out.";
      }
      num_done++;
    }

    {
      Output ko(stats_out_filename, binary);
      ori_stats.Write(ko.Stream(), binary);
      vts_stats.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Write stats out.";

    return 1;
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

