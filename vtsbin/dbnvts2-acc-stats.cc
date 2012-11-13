/*
 * dbnvts2-acc-stats.cc
 *
 *  Created on: Nov 8, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Accumulate both poistive and negative statistics from the original feature space,
 * instead of the others which all collecting fromt he normalized space.
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
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate a positive and negative Gaussian for each sigmoid hidden unit "
            "of the first NN layer in the original feature space.\n"
            "Usage:  dbnvts2-acc-stats [options] <biasedlinearity-layer-in> <gmm-model-in> "
            "<ori-feature-rspecifier> <pos-stats-out> "
            "<neg-stats-out> <pos2neg-prior-stats-out>\n"
            "e.g.: \n"
            " dbnvts2-acc-stats bl_layer gmm.mdl ark:ori_features.ark ark:noise.ark "
            "pos.acc neg.acc pos2neg\n";

    ParseOptions po(usage);

    bool hard_decision = true;
    po.Register(
        "hard-decision", &hard_decision,
        "Using hard decision or probabilistic decision for stats accumulation");

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string update_flags_str = "mvw";  // note: t is ignored
    po.Register("update-flags", &update_flags_str,
                "Which GMM parameters will be "
                "updated: subset of mvw.");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    int32 num_frames = 9;
    po.Register("num-frames", &num_frames, "Input feature window length");

    std::string cmvn_stats_rspecifier = "";
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string bl_layer_filename = po.GetArg(1), gmm_model_filename =
        po.GetArg(2), feature_rspecifier = po.GetArg(3), pos_accs_wxfilename =
        po.GetArg(4), neg_accs_wxfilename = po.GetArg(5),
        pos2neg_stats_wxfilename = po.GetArg(6);  // global prior ratio parameters

    // biased linearity layer weight and bias
    Matrix<BaseFloat> linearity;
    Vector<BaseFloat> bias;
    {
      bool binary;
      Input ki(bl_layer_filename, &binary);
      if (!ReadBiasedLinearityLayer(ki.Stream(), binary, linearity, bias)) {
        KALDI_ERR<< "Load biased linearity layer from " << bl_layer_filename
        << " failed!";
      }
    }

    // converting the normalized weights to original feature space
    // if necessary
    Vector<double> norm_mean, norm_std;
    if (cmvn_stats_rspecifier != "") {
      // convert the models back to the original feature space
      RandomAccessDoubleMatrixReader cmvn_reader(cmvn_stats_rspecifier);
      if (!cmvn_reader.HasKey("global")) {
        KALDI_ERR<< "No normalization statistics available for key global";
      }
      const Matrix<double> &stats = cmvn_reader.Value("global");
      // convert stats to mean and std
      int32 dim = stats.NumCols() - 1;
      norm_mean.Resize(dim, kSetZero);
      norm_std.Resize(dim, kSetZero);
      double count = stats(0, dim);
      if (count < 1.0)
        KALDI_ERR<< "Insufficient stats for cepstral mean and variance normalization: "
        << "count = " << count;

      for (int32 i = 0; i < dim; ++i) {
        norm_mean(i) = stats(0, i) / count;
        norm_std(i) = sqrt((stats(1, i) / count) - norm_mean(i) * norm_mean(i));
      }

      // converting weights
      ConvertWeightToOriginalSpace(num_frames, norm_mean, norm_std, linearity,
                                   bias);
    }

    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(gmm_model_filename, &binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    AccumAmDiagGmm pos_gmm_accs, neg_gmm_accs;
    pos_gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));
    neg_gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Matrix<BaseFloat> acts;

    Timer tim;
    if (!silent)
      KALDI_LOG<< "ACCUMULATION STARTED";

    int32 num_done = 0;
    Matrix<double> prior_counts(2, am_gmm.NumPdfs(), kSetZero);

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &feats = feature_reader.Value();

      // fwd-pass through the biased linearity layers only
      acts.Resize(feats.NumRows(), linearity.NumRows(), kSetZero);
      acts.AddVecToRows(1.0, bias);  // b
      acts.AddMatMat(1.0, feats, kNoTrans, linearity, kTrans, 1.0);  // w_T * x + b

      //accumulate statistics
      for (int32 r = 0; r < acts.NumRows(); r++) {
        for (int32 c = 0; c < acts.NumCols(); c++) {
          BaseFloat val = acts(r, c);

          if (hard_decision) {  // hard decision
            if (val >= 0.0) {  // val >= 0.0, i.e. sigmoid(val) > 0.5
              pos_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1.0);
              prior_counts(0, c) += 1;
            } else {
              neg_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1.0);
              prior_counts(1, c) += 1;
            }
          } else {  // soft decision
            val = 1.0 / (1.0 + exp(-val));
            pos_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, val);
            neg_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1 - val);
            prior_counts(0, c) += val;
            prior_counts(1, c) += (1 - val);
          }
        }
      }

      // progress log
      if (num_done % 100 == 0) {
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += feats.NumRows();
    }

    {
      Output ko(pos_accs_wxfilename, binary);
      pos_gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Written positive accs.";

    {
      Output ko(neg_accs_wxfilename, binary);
      neg_gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Written negative accs.";

    {
      Output ko(pos2neg_stats_wxfilename, binary);
      prior_counts.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Written prior counts.";

    // final message
    if (!silent)
      KALDI_LOG<< "ACCUMULATED FINISHED " << tim.Elapsed() << "s, fps"
      << tot_t / tim.Elapsed();
    if (!silent)
      KALDI_LOG<< "Done " << num_done << " files";

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
