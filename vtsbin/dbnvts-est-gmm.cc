/*
 * dbnvts-est-gmm.cc
 *
 *  Created on: Oct 29, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Estimate the positive GMM using the accumulated statistics.
 *
 *  No transition model is involved.
 *
 */


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/mle-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    MleDiagGmmOptions gmm_opts;

    const char *usage =
        "Do Maximum Likelihood re-estimation of GMM-based acoustic model\n"
        "Usage:  dbnvts-est-gmm [options] <model-in> <stats-in> <model-out>\n"
        "e.g.: dbnvts-est-gmm 1.mdl 1.acc 2.mdl\n";

    bool binary_write = false;
    int32 mixup = 0;
    int32 mixdown = 0;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat min_count = 20.0;
    std::string update_flags_str = "mvw";
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("mix-up", &mixup, "Increase number of mixture components to "
                "this overall target.");
    po.Register("min-count", &min_count,
                "Minimum per-Gaussian count enforced while mixing up and down.");
    po.Register("mix-down", &mixdown, "If nonzero, merge mixture components to this "
                "target.");
    po.Register("power", &power, "If mixing up, power to allocate Gaussians to"
                " states.");
    po.Register("update-flags", &update_flags_str, "Which GMM parameters to "
                "update: subset of mvw.");
    po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
                "means by standard deviation times this factor.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    gmm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::GmmFlagsType update_flags =
        StringToGmmFlags(update_flags_str);

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    AmDiagGmm am_gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    AccumAmDiagGmm gmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      gmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    {  // Update GMMs.
      BaseFloat objf_impr, count;
      BaseFloat tot_like = gmm_accs.TotLogLike(),
          tot_t = gmm_accs.TotCount();
      MleAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
                         &objf_impr, &count);
      KALDI_LOG << "GMM update: average " << (objf_impr/count)
                << " objective function improvement per frame over "
                <<  count <<  " frames";
      KALDI_LOG << "GMM update: Overall avg like per frame = "
                << (tot_like/tot_t) << " over " << tot_t << " frames.";
    }

    if (mixup != 0 || mixdown != 0 || !occs_out_filename.empty()) {  // get state occs
      Vector<BaseFloat> state_occs;
      state_occs.Resize(gmm_accs.NumAccs());
      for (int i = 0; i < gmm_accs.NumAccs(); i++)
        state_occs(i) = gmm_accs.GetAcc(i).occupancy().Sum();

      if (mixdown != 0)
        am_gmm.MergeByCount(state_occs, mixdown, power, min_count);

      if (mixup != 0)
        am_gmm.SplitByCount(state_occs, mixup, perturb_factor,
                            power, min_count);

      if (!occs_out_filename.empty()) {
        bool binary = true;  // write this in text mode-- useful to look at.
        kaldi::Output ko(occs_out_filename, binary);
        state_occs.Write(ko.Stream(), binary);
      }
    }

    {
      Output ko(model_out_filename, binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


