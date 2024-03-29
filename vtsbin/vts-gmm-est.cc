/*
 * vtsbin/vts-gmm-est.cc
 *
 *  Created on: Nov 25, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "vts/vts-accum-am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    VtsDiagGmmOptions gmm_opts;

    const char *usage =
        "Do Maximum Likelihood re-estimation of GMM-based acoustic model in VAT\n"
        "Usage:  vts-gmm-est [options] <model-in> <stats-in> <model-out>\n"
        "Warning: The objective value changes computed in this tool is not implemented yet.\n"
        "e.g.: vts-gmm-est 1.mdl 1.acc 2.mdl\n";

    bool binary_write = true;
    TransitionUpdateConfig tcfg;
    int32 mixup = 0;
    int32 mixdown = 0;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat min_count = 20.0;
    std::string update_flags_str = "mvwt";
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
                "update: subset of mvwt.");
    po.Register("perturb-factor", &perturb_factor, "While mixing up, perturb "
                "means by standard deviation times this factor.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    tcfg.Register(&po);
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
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    Vector<double> transition_accs;
    VtsAccumAmDiagGmm gmm_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      gmm_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    if (update_flags & kGmmTransitions) {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.Update(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    {  // Update GMMs.
      BaseFloat objf_impr, count;
      BaseFloat tot_like = gmm_accs.TotLogLike(),
          tot_t = gmm_accs.TotCount();
      VtsAmDiagGmmUpdate(gmm_opts, gmm_accs, update_flags, &am_gmm,
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
      trans_model.Write(ko.Stream(), binary_write);
      am_gmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}





