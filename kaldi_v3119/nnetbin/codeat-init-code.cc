/*
 * codeat-init-code.cc
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Generate initial code vectors for different set.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialise code vectors for each set\n"
        "Usage: codeat-init-code [options] <set2utt-map> <code-wspecifier> "
        "e.g.: \n"
        " codeat-init-code --code-dim=10 ark:set2utt.map ark,t:code.ark \n";

    ParseOptions po(usage);
    int32 code_dim = 0;
    bool add_gauss_noise = false;
    po.Register("code-dim", &code_dim, "Code dimension");
    po.Register("add-gauss-noise", &add_gauss_noise, "Add Gaussian noise or not");

    po.Read(argc, argv);

    if(po.NumArgs() !=2 ){
      po.PrintUsage();
      exit(1);
    }

    std::string set2utt_rspecifier = po.GetArg(1);
    std::string code_wspecifier = po.GetArg(2);

    SequentialTokenVectorReader set2utt_reader(set2utt_rspecifier);
    BaseFloatVectorWriter code_writer(code_wspecifier);

    Vector<BaseFloat> code(code_dim, kSetZero);

    int32 num_set = 0;
    for(; !set2utt_reader.Done(); set2utt_reader.Next()) {
      std::string key = set2utt_reader.Key();

      if(add_gauss_noise){
        code.SetRandn();
      }

      code_writer.Write(key, code);
      ++num_set;
    }

    KALDI_LOG << "Created " << num_set << (add_gauss_noise? " random ":"") << " codes.";

    return 0;

  } catch (const std::exception &e){
    std::cerr << e.what();
    return -1;
  }
}



