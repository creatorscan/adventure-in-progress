// vtsbin/dbnvts2-sum-priors.cc

// Copyright 2009-2011  Saarland University;  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {

    const char *usage =
        "Sum multiple accumulated positive and negative prior count stats files.\n"
            "Usage: dbnvts2-sum-priors [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = false;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    Matrix<double> prior_stats;

    int num_accs = po.NumArgs() - 1;
    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      prior_stats.Read(ki.Stream(), binary_read, true /*add read values*/);
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      prior_stats.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Summed " << num_accs << " prior stats.";
    KALDI_LOG<< "Written stats to " << stats_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

