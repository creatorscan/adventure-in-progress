/*
 * dbnvts2-neg-gauss.cc
 *
 *  Created on: Nov 9, 2012
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
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Generate negative Gaussians based on positive Gaussians and decision boundary.\n"
            "Usage:  dbnvts-neg-gauss [options] <pos-am-in> <biaslinearity-layer-in> "
            "<pos_neg-prior-counts-in> <neg-am-out>  <var-scale-out> \n"
            "e.g.: \n"
            " dbnvts2-neg-gauss pos.mdl bl_layer neg.mdl pos2neg.stats var_scale.stats\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary,
                "Write out the model in binary or text mode");

    int32 num_frames = 9;
    po.Register("num-frames", &num_frames, "Input feature window length");

    std::string cmvn_stats_rspecifier;
    po.Register("cmvn-stats", &cmvn_stats_rspecifier,
                "rspecifier for global CMVN feature normalization");

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
        KALDI_ERR<< "Load biased linearity layer from " << bl_layer_filename
        << " failed!";
      }
    }

    // positive to negative prior ratio
    Vector<double> pos2neg_log_prior_ratio(linearity.NumRows(), kSetZero);
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

    // compute the negative Gaussians
    AmDiagGmm neg_am_gmm;
    neg_am_gmm.CopyFromAmDiagGmm(pos_am_gmm);
    Vector<double> var_scale(linearity.NumRows(), kSetZero);
    ComputeNegativeReflectionGmm(linearity, bias, pos2neg_log_prior_ratio,
                                 neg_am_gmm, var_scale);

    {
      Output ko(neg_am_out_filename, binary);
      neg_am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Write negative Gmm model.";

    {
      Output ko(var_scale_wxfilename, binary);
      var_scale.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Write var scale stats.";

    return 1;
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

