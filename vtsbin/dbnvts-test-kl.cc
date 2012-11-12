/*
 * dbnvts-test-kl.cc
 *
 *  Created on: Nov 2, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *

 *
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-activation.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/mle-am-diag-gmm.h"
#include "gmm/am-diag-gmm.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute the KL Divengence between the VST compensated Gaussians \n"
            "Usage:  dbnvts-test-kl [options] <feature-rspecifier> <noise-rspecifier>\n"
            "<clean-bl-layer-act-rspecifier> <pos-model-in> <neg-model-in> <var-scale-in> <kl-stats-wspecifier>"
            "e.g.: \n"
            " dbnvts-test-kl ark:features.ark ark:noise.ark ark:bl_acts.ark pos.mdl neg.mdl var_scale.stats ark:kl_stats.ark\n";

    ParseOptions po(usage);

    bool clean_data = false;
    po.Register("clean-data", &clean_data, "Whether the features are clean data or noise data");

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    bool clean_cmvn = true;
    po.Register(
        "clean-cmvn",
        &clean_cmvn,
        "Inputs are clean CMVN normalized features or noise compensated CMVN normalized features");

    std::string clean_type = "clean1";
    po.Register("clean-type", &clean_type,
                "Filename prefix for the clean act data");

    std::string noise_type = "n1_snr20";
    po.Register("noise-type", &noise_type,
                "Filename prefix for the noisy data");

    bool hard_decision = true;
    po.Register(
     "hard-decision", &hard_decision,
     "Using hard decision or probabilistic decision for stats accumulation");

    std::string cmvn_stats_rspecifier;
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

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

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1), noise_rspecifier = po.GetArg(
        2), blacts_rspecifier = po.GetArg(3), pos_gmm_filename = po.GetArg(4),
        neg_gmm_filename = po.GetArg(5), var_scale_filename = po.GetArg(6),
        klstats_wspecifier = po.GetArg(7);  // global normalization parameters

    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_gmm_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }
    AmDiagGmm neg_am_gmm;
    {
      bool binary;
      Input ki(neg_gmm_filename, &binary);
      neg_am_gmm.Read(ki.Stream(), binary);
    }

    int32 frame_dim = pos_am_gmm.Dim() / num_frames;
    KALDI_ASSERT(frame_dim * num_frames == pos_am_gmm.Dim());
    // converting the normalized gmms to original feature space
    // if necessary
    Vector<double> global_mean(pos_am_gmm.Dim()), frame_mean(frame_dim);
    Vector<double> global_std(pos_am_gmm.Dim()), frame_var(frame_dim),
        frame_std(frame_dim);
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

      for (int32 i = 0; i < dim; ++i) {
        frame_mean(i) = stats(0, i) / count;
        frame_var(i) = stats(1, i) / count - frame_mean(i) * frame_mean(i);
        frame_std(i) = sqrt(frame_var(i));
      }
      for (int32 i = 0; i < pos_am_gmm.Dim(); ++i) {
        global_mean(i) = frame_mean(i % dim);
        global_std(i) = frame_std(i % dim);
      }

      // for clean data, no compensation, thus no need to convert back to un-normalized space
      if (!clean_data){
        NormalizedGmmToGmm(global_mean, global_std, pos_am_gmm);
        NormalizedGmmToGmm(global_mean, global_std, neg_am_gmm);
      }
    }

    // variance scale factors
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == pos_am_gmm.NumPdfs());
    }
    if (!use_var_scale) {  // if no scaling is applied, set them to 1
      for (int32 i = 0; i < pos_am_gmm.NumPdfs(); ++i) {
        var_scale(i) = 1.0;
      }
    }

    // noisy models
    AmDiagGmm pos_noise_am, neg_noise_am;
    pos_noise_am.CopyFromAmDiagGmm(pos_am_gmm);
    neg_noise_am.CopyFromAmDiagGmm(neg_am_gmm);


    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    RandomAccessBaseFloatMatrixReader blacts_reader(blacts_rspecifier);
    BaseFloatVectorWriter klstats_writer(klstats_wspecifier);

    Timer tim;
    if (!silent)
      KALDI_LOG << "ACCUMULATION STARTED";

    int32 num_done = 0;
    bool first = true;
    Matrix<double> pos_mean(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
    Matrix<double> pos_var(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
    Vector<double> pos_count(pos_am_gmm.NumPdfs(), kSetZero);
    Matrix<double> neg_mean(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
    Matrix<double> neg_var(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
    Vector<double> neg_count(pos_am_gmm.NumPdfs(), kSetZero);

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string noise_key = feature_reader.Key();
      std::string clean_key = noise_key;
      clean_key.replace(0, noise_type.length(), clean_type);

      if (first && !clean_data) {
        // we use a global noise model for all the features
        // thus only needs to compensate once
        first=false;

        /*
         * Compute VTS compensated postive and negative Gaussians
         */
        // read noise parameters
        if (!noiseparams_reader.HasKey(noise_key + "_mu_h")
            || !noiseparams_reader.HasKey(noise_key + "_mu_z")
            || !noiseparams_reader.HasKey(noise_key + "_var_z")) {
          KALDI_ERR
              << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }
        Vector<double> mu_h(noiseparams_reader.Value(noise_key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(noise_key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(noise_key + "_var_z"));
        if (g_kaldi_verbose_level >= 1) {
          KALDI_LOG << "Additive Noise Mean: " << mu_z;
          KALDI_LOG << "Additive Noise Covariance: " << var_z;
          KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
        }

        // compensate the postive and negative gmm models
        pos_noise_am.CopyFromAmDiagGmm(pos_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                                num_fbank, dct_mat, inv_dct_mat, num_frames,
                                pos_noise_am);

        neg_noise_am.CopyFromAmDiagGmm(neg_am_gmm);
        CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                                num_fbank, dct_mat, inv_dct_mat, num_frames,
                                neg_noise_am);

        // convert back to normalized feature space if necessary
        if (cmvn_stats_rspecifier != "") {

          if (clean_cmvn) {
            GmmToNormalizedGmm(global_mean, global_std, pos_noise_am);
            GmmToNormalizedGmm(global_mean, global_std, neg_noise_am);
          } else {
            Vector<double> noise_frame_mean(frame_mean);
            Vector<double> noise_frame_var(frame_var);
            Matrix<double> Jx, Jz;

            CompensateDiagGaussian(mu_h, mu_z, var_z, num_cepstral, num_fbank,
                                   dct_mat, inv_dct_mat, noise_frame_mean,
                                   noise_frame_var, Jx, Jz);

            Vector<double> noise_mean(global_mean.Dim());
            Vector<double> noise_std(global_mean.Dim());
            for (int32 i = 0; i < noise_mean.Dim(); ++i) {
              noise_mean(i) = noise_frame_mean(i % frame_dim);
              noise_std(i) = sqrt(noise_frame_var(i % frame_dim));  // convert covariance to standard deviation
            }

            GmmToNormalizedGmm(noise_mean, noise_std, pos_noise_am);
            GmmToNormalizedGmm(noise_mean, noise_std, neg_noise_am);
          }

        }
      }

      /*
       * Compute true postive and negative Gaussians using noisy data and
       * clean acts.
       */
      const Matrix<BaseFloat> &feats = feature_reader.Value();

      if (!blacts_reader.HasKey(clean_key)) {
        KALDI_ERR << "Clean acts for key: " << noise_key << "(" << clean_key
            << ") doesn't exist!";
      }
      const Matrix<BaseFloat> &acts = blacts_reader.Value(clean_key);
      KALDI_ASSERT(acts.NumRows() == feats.NumRows());

      for (int32 r = 0; r < acts.NumRows(); r++) {  // iterate all the frames
        for (int32 c = 0; c < acts.NumCols(); c++) {
          BaseFloat val = acts(r, c);

          if (hard_decision) {  // hard decision
            if (val >= 0.0) {  // val >= 0.0, i.e. sigmoid(val) > 0.5
              pos_count(c) += 1.0;
              (pos_mean.Row(c)).AddVec(1.0, feats.Row(r));
              (pos_var.Row(c)).AddVec2(1.0, feats.Row(r));
            } else {
              neg_count(c) += 1.0;
              (neg_mean.Row(c)).AddVec(1.0, feats.Row(r));
              (neg_var.Row(c)).AddVec2(1.0, feats.Row(r));
            }
          } else {  // soft decision
            val = 1.0 / (1.0 + exp(-val));
            pos_count(c) += val;
            (pos_mean.Row(c)).AddVec(val, feats.Row(r));
            (pos_var.Row(c)).AddVec2(val, feats.Row(r));
            neg_count(c) += (1.0 - val);
            (neg_mean.Row(c)).AddVec(1.0 - val, feats.Row(r));
            (neg_var.Row(c)).AddVec2(1.0 - val, feats.Row(r));
          }
        }
      }

      // progress log
      if (num_done % 1000 == 0) {
        if (!silent)
          KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += feats.NumRows();
    }

    // compute the mean and variance for each hidden unit
    // and compute the KL divergence
    Matrix<double> klstats(2, pos_am_gmm.NumPdfs(), kSetZero);
    for (int32 c = 0; c < pos_am_gmm.NumPdfs(); ++c) {
      (pos_mean.Row(c)).Scale(1.0 / pos_count(c));
      (pos_var.Row(c)).Scale(1.0 / pos_count(c));
      (pos_var.Row(c)).AddVec2(-1.0, pos_mean.Row(c));

      DiagGmm *pos_gmm = &(pos_noise_am.GetPdf(c));
      DiagGmmNormal pos_ngmm(*pos_gmm);

      klstats(0, c) += KLDivergenceDiagGaussian(
          Vector<double>(pos_mean.Row(c)), Vector<double>(pos_var.Row(c)),
          Vector<double>(pos_ngmm.means_.Row(0)),
          Vector<double>(pos_ngmm.vars_.Row(0)));

      (neg_mean.Row(c)).Scale(1.0 / neg_count(c));
      (neg_var.Row(c)).Scale(1.0 / neg_count(c));
      (neg_var.Row(c)).AddVec2(-1.0, neg_mean.Row(c));

      DiagGmm *neg_gmm = &(neg_noise_am.GetPdf(c));
      DiagGmmNormal neg_ngmm(*neg_gmm);

      klstats(1, c) += KLDivergenceDiagGaussian(
          Vector<double>(neg_mean.Row(c)), Vector<double>(neg_var.Row(c)),
          Vector<double>(neg_ngmm.means_.Row(0)),
          Vector<double>(neg_ngmm.vars_.Row(0)));

    }

    // output the average KL Divergence
    klstats_writer.Write("pos", Vector<BaseFloat>(klstats.Row(0)));
    klstats_writer.Write("neg", Vector<BaseFloat>(klstats.Row(1)));

    KALDI_LOG << "Written KL stats.";
    KALDI_LOG << "Average KL Divergence for positive class is: "
        << (klstats.Row(0)).Sum() / klstats.NumCols();
    KALDI_LOG << "Average KL Divergence for negative class is: "
        << (klstats.Row(1)).Sum() / klstats.NumCols();

    // final message
    if (!silent)
      KALDI_LOG << "ACCUMULATED FINISHED " << tim.Elapsed() << "s, fps"
          << tot_t / tim.Elapsed();
    if (!silent)
      KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
