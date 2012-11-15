/*
 * vtsbin/remove-trans-model.cc
 *
 *  Created on: Nov 15, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  The difference between the original HMM and the one used for dbnvts
 *  is the transition model is not available in dbnvts HMMs.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage = "Remove the transition model of the input HMM model.\n"
        "Usage:  remove-trans-model [options] <model-in> <model-out>"
        "e.g.: \n"
        " remove-trans-model 1.mdl 2.mdl\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write out the model in binary");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1), model_out_filename =
        po.GetArg(2);

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    {
      Output ko(model_out_filename, binary);
      am_gmm.Write(ko.Stream(), binary);
    }

    return 0;
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

