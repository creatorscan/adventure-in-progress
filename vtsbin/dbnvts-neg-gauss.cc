/*
 * dbnvts-neg-gauss.cc
 *
 *  Created on: Oct 29, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Generate negative Gaussians based on the positive Gaussians
 *  and the decision boundary.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/am-diag-gmm.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Generate negative Gaussians based on positive Gaussians and decision boundary.\n"
            "Usage:  dbnvts-neg-gauss [options] <pos-am-in> <biaslinearity-layer-in> "
            "<pos_neg-prior-counts-in> <neg-am-out> <var-scale-out>\n"
            "e.g.: \n"
            " dbnvts-neg-gauss pos.mdl bl_layer pos2neg.stats neg.mdl shared_var_scale.stats\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary,
                "Write out the model in binary or text mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_am_filename = po.GetArg(1), bl_layer_filename = po.GetArg(
        2), pos2neg_prior_filename = po.GetArg(3), neg_am_out_filename = po
        .GetArg(4), var_scale_wxfilename = po.GetArg(5);

    // positive AM Gmm
    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_am_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }

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

    AmDiagGmm neg_am_gmm;
    Vector<double> var_scale(linearity.NumRows(), kSetZero);
    neg_am_gmm.CopyFromAmDiagGmm(pos_am_gmm);
    ComputeNegativeGmm(linearity, bias, pos2neg_log_prior_ratio, neg_am_gmm, var_scale);

    {
      Output ko(neg_am_out_filename, binary);
      neg_am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Write negative Gmm model.";

    {
      Output ko(var_scale_wxfilename, binary);
      var_scale.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Write covariance scales.";

    return 1;
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

