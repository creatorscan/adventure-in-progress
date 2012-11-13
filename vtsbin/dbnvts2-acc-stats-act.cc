/*
 * vtsbin/dbnvts2-acc-stats-act.cc
 *
 *  Created on: Nov 13, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Accumulate both poistive and negative statistics from the original feature space,
 * instead of the others which all collecting fromt he normalized space.
 *
 * Using the given reference activations.
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
            "Usage:  dbnvts2-acc-stats-act [options] <gmm-model-in> <ori-feature-rspecifier>"
            "<act-rspecifier> <pos-stats-out> <neg-stats-out>\n"
            "e.g.: \n"
            " dbnvts2-acc-stats-act gmm.mdl ark:ori_features.ark ark:act.ark pos.acc neg.acc\n";

    ParseOptions po(usage);

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

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string update_flags_str = "mvw";  // note: t is ignored
    po.Register("update-flags", &update_flags_str,
                "Which GMM parameters will be "
                "updated: subset of mvw.");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string gmm_model_filename = po.GetArg(1), feature_rspecifier = po
        .GetArg(2), act_rspecifier = po.GetArg(3), pos_accs_wxfilename = po
        .GetArg(4), neg_accs_wxfilename = po.GetArg(5);

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
    RandomAccessBaseFloatMatrixReader act_reader(act_rspecifier);

    Timer tim;
    if (!silent)
      KALDI_LOG<< "ACCUMULATION STARTED";

    int32 num_done = 0;

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string noise_key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();

      std::string clean_key = noise_key;
      clean_key.replace(0, noise_type.length(), clean_type);

      if (!act_reader.HasKey(clean_key)) {
        KALDI_WARN<< "No activations found for the feature: " << noise_key << ", skipped!";
        continue;
      }
      const Matrix<BaseFloat> &acts = act_reader.Value(clean_key);

      //accumulate statistics
      for (int32 r = 0; r < acts.NumRows(); r++) {
        for (int32 c = 0; c < acts.NumCols(); c++) {
          BaseFloat val = acts(r, c);

          if (hard_decision) {  // hard decision
            if (val >= 0.0) {  // val >= 0.0, i.e. sigmoid(val) > 0.5
              pos_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1.0);
            } else {
              neg_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1.0);
            }
          } else {  // soft decision
            val = 1.0 / (1.0 + exp(-val));
            pos_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, val);
            neg_gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1 - val);
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
