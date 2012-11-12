/*
 * dbnvts-est-gauss.cc
 *
 *  Created on: Oct 27, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * TODO:: If hard-decision and 0.5 threshold is the best choice, then
 * no need to do the sigmoid computation. As long as the before sigmoid
 * activation is positive, the sigmoid result will be larger than 0.5.
 * We could thus save the cost of sigmoid computation.
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
        "Estimate a single Gaussian for each sigmoid hidden unit of the first NN layer.\n"
            "Usage:  dbnvts-acc-stats [options] <biasedlinearity-layer-in> <gmm-model-in> "
            "<feature-rspecifier> <stats-out> <pos_neg-prior-counts>\n"
            "e.g.: \n"
            " dbnvts-acc-stats bl_layer gmm.mdl ark:features.ark 1.acc pos2neg.stats\n";

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

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string bl_layer_filename = po.GetArg(1), gmm_model_filename =
        po.GetArg(2), feature_rspecifier = po.GetArg(3), accs_wxfilename = po
        .GetArg(4), prior_wxfilename = po.GetArg(5);  // global normalization parameters

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
    }

    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(gmm_model_filename, &binary);
      am_gmm.Read(ki.Stream(), binary);
    }
    AccumAmDiagGmm gmm_accs;
    gmm_accs.Init(am_gmm, StringToGmmFlags(update_flags_str));

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    Matrix<BaseFloat> acts;

    Timer tim;
    if (!silent)
      KALDI_LOG << "ACCUMULATION STARTED";

    int32 num_done = 0;
    Matrix<double> prior_counts(2, am_gmm.NumPdfs(), kSetZero);

    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      const Matrix<BaseFloat> &feats = feature_reader.Value();

      // fwd-pass through the biased linearity layers only
      acts.Resize(feats.NumRows(), linearity.NumRows(), kSetZero);
      acts.AddVecToRows(1.0, bias); // b
      acts.AddMatMat(1.0, feats, kNoTrans, linearity, kTrans, 1.0); // w_T * x + b

      //check for NaN/inf and accumulate statistics
      for (int32 r = 0; r < acts.NumRows(); r++) {
        for (int32 c = 0; c < acts.NumCols(); c++) {
          BaseFloat val = acts(r, c);

          if (hard_decision) {  // hard decision
            if (val >= 0.0) { // val >= 0.0, i.e. sigmoid(val) > 0.5
              gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, 1.0);
              prior_counts(0, c) += 1;
            } else {
              prior_counts(1, c) += 1;
            }
          } else {  // soft decision
            val = 1.0 / (1.0 + exp(-val));
            gmm_accs.AccumulateForGmm(am_gmm, feats.Row(r), c, val);
            prior_counts(0, c) += val;
            prior_counts(1, c) += (1 - val);
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

    {
      Output ko(accs_wxfilename, binary);
      gmm_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";

    {
      Output ko(prior_wxfilename, binary);
      prior_counts.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written prior counts.";

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
