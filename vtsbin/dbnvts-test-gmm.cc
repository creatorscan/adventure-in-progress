/*
 * dbnvts-test-gmm.cc
 *
 *  Created on: Oct 29, 2012
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
#include "vts/dbnvts-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Test whether the positive and negative Gmms give the same decision as the hidden unit.\n"
            "Usage:  dbnvts-test-gmm [options] <biaslinearity-layer-in> <pos-model-in> <neg-model-in> "
            "<pos2neg-prior-in> <var-scale-in> <feature-rspecifier> <stats-out>\n"
            "e.g.: \n"
            " dbnvts-test-gmm bl_layer pos.mdl neg.modl ark:feat.ark stats\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write stats in binary mode");

    std::string cmvn_stats_rspecifier;
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

    std::string noise_params_rspecifier;
    po.Register("noise-params", &noise_params_rspecifier,
                "rspecifier for per utterance VTS noise parameters");

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

    std::string bl_layer_filename = po.GetArg(1), pos_model_filename =
        po.GetArg(2), neg_model_filename = po.GetArg(3),
        pos2neg_prior_filename = po.GetArg(4), var_scale_filename = po.GetArg(
            5), feature_rspecifier = po.GetArg(6), stats_out_filename = po
            .GetArg(7);

    // biased linearity layer weight and bias
    Matrix<BaseFloat> linearity;
    Vector<BaseFloat> bias;
    {
      bool binary;
      Input ki(bl_layer_filename, &binary);
      if (!ReadBiasedLinearityLayer(ki.Stream(), binary, linearity, bias)) {
        KALDI_ERR << "Load biased linearity layer from " << bl_layer_filename
            << " failed!";
      }
      KALDI_ASSERT(linearity.NumRows() == bias.Dim());
    }

    // positive to negative prior ratio
    Vector<BaseFloat> pos2neg_log_prior_ratio(linearity.NumRows(), kSetZero);
    Matrix<double> prior_stats;
    {
      bool binary;
      Input ki(pos2neg_prior_filename, &binary);
      prior_stats.Read(ki.Stream(), binary);
      KALDI_ASSERT(
          prior_stats.NumRows()==2 && prior_stats.NumCols()==linearity.NumRows());
    }
    for (int32 i = 0; i < linearity.NumRows(); ++i) {
      pos2neg_log_prior_ratio(i) = log(prior_stats(0, i) / prior_stats(1, i));
    }

    // var-scale
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == linearity.NumRows());
    }

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

    KALDI_ASSERT(
        pos_am_gmm.NumPdfs() == linearity.NumRows() && neg_am_gmm.NumPdfs() == linearity.NumRows());

    AmDiagGmm ori_pos_am_gmm, ori_neg_am_gmm, noise_pos_am_gmm,
        noise_neg_am_gmm;
    ori_pos_am_gmm.CopyFromAmDiagGmm(pos_am_gmm);
    ori_neg_am_gmm.CopyFromAmDiagGmm(neg_am_gmm);

    // converting the normalized gmms to original feature space
    // if necessary
    Vector<double> global_mean(linearity.NumCols());
    Vector<double> global_std(linearity.NumCols());
    if (cmvn_stats_rspecifier != "") {
      // convert the models back to the original feature space
      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_stats_rspecifier);
      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR << "No normalization statistics available for key global";
      }
      const Matrix<double> &stats = cmvn_reader.Value("global");
      // convert stats to mean and std
      int32 dim = stats.NumCols() - 1;
      double count = stats(0, dim);
      if (count < 1.0)
        KALDI_ERR
            << "Insufficient stats for cepstral mean and variance normalization: "
            << "count = " << count;

      KALDI_ASSERT(global_mean.Dim() % dim == 0);

      Vector<double> mean(dim), std(dim);
      for (int32 i = 0; i < dim; ++i) {
        mean(i) = stats(0, i) / count;
        std(i) = sqrt((stats(1, i) / count) - mean(i) * mean(i));
      }
      for (int32 i = 0; i < linearity.NumCols(); ++i) {
        global_mean(i) = mean(i % dim);
        global_std(i) = std(i % dim);
      }

      NormalizedGmmToGmm(global_mean, global_std, ori_pos_am_gmm);
      NormalizedGmmToGmm(global_mean, global_std, ori_neg_am_gmm);
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);

    int32 num_done = 0;
    /*
     * stats(0, :): nnet_act >= 0.0 && pos_llh >= neg_llh
     * stats(1, :): nnet_act >= 0.0 && pos_llh < neg_llh
     * stats(2, :): nnet_act < 0.0 && pos_llh >= neg_llh
     * stats(3, :): nnet_act < 0.0 && pos_llh < neg_llh
     */
    Matrix<BaseFloat> ori_stats(4, linearity.NumRows(), kSetZero);
    Matrix<BaseFloat> vts_stats(4, linearity.NumRows(), kSetZero);
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read the features
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feat = feature_reader.Value();

      // compute the hidden activations
      Matrix<BaseFloat> mat(feat.NumRows(), linearity.NumRows());
      mat.AddMatMat(1.0, feat, kNoTrans, linearity, kTrans, 0.0);  // w_T * feat
      mat.AddVecToRows(1.0, bias);

      // estimate the new decision boundary
      Matrix<BaseFloat> new_linearity(linearity);
      Vector<BaseFloat> new_bias(bias);
      ComputeGaussianBoundary(pos_am_gmm, neg_am_gmm,
                              pos2neg_log_prior_ratio, var_scale, new_linearity,
                              new_bias);

      Matrix<BaseFloat> new_mat(feat.NumRows(), linearity.NumRows());
      new_mat.AddMatMat(1.0, feat, kNoTrans, new_linearity, kTrans, 0.0);  // w_T * feat
      new_mat.AddVecToRows(1.0, new_bias);

      // evaluate each frame
      for (int32 i = 0; i < feat.NumRows(); ++i) {
        for (int32 pdf = 0; pdf < linearity.NumRows(); ++pdf) {

          if (mat(i, pdf) >= 0.0) {
            if (new_mat(i,pdf) >= 0.0) {
              ori_stats(0, pdf) += 1;
            } else {
              ori_stats(1, pdf) += 1;
            }
          } else {
            if (new_mat(i,pdf) >= 0.0) {
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
          KALDI_ERR
              << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
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
        noise_pos_am_gmm.CopyFromAmDiagGmm(ori_pos_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, true, num_cepstral, num_fbank,
                                dct_mat, inv_dct_mat, num_frames,
                                noise_pos_am_gmm);

        noise_neg_am_gmm.CopyFromAmDiagGmm(ori_neg_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, true, num_cepstral, num_fbank,
                                dct_mat, inv_dct_mat, num_frames,
                                noise_neg_am_gmm);

        // convert back to normalized feature space if necessary
        if (cmvn_stats_rspecifier != "") {
          GmmToNormalizedGmm(global_mean, global_std, noise_pos_am_gmm);
          GmmToNormalizedGmm(global_mean, global_std, noise_neg_am_gmm);
        }

        // estimate the new decision boundary
        Matrix<BaseFloat> vts_linearity(linearity);
        Vector<BaseFloat> vts_bias(bias);
        ComputeGaussianBoundary(noise_pos_am_gmm, noise_neg_am_gmm,
                                pos2neg_log_prior_ratio, var_scale,
                                vts_linearity, vts_bias);

        new_mat.AddMatMat(1.0, feat, kNoTrans, vts_linearity, kTrans, 0.0);  // w_T * feat
        new_mat.AddVecToRows(1.0, vts_bias);

        // evaluate each frame
        for (int32 i = 0; i < feat.NumRows(); ++i) {
          for (int32 pdf = 0; pdf < linearity.NumRows(); ++pdf) {

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
        KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
    }

    {
      Output ko(stats_out_filename, binary);
      ori_stats.Write(ko.Stream(), binary);
      vts_stats.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Write stats out.";

    return 1;
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

