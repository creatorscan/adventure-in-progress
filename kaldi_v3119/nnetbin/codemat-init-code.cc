/*
 * codemat-init-code.cc
 *
 *  Created on: Nov 8, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Generate initial code vectors for each utterance.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using kaldi::int32;

    const char *usage =
        "Initialise code vectors for each utterance\n"
        "Usage: codemat-init-code [options] <feat-rspecifier> <code-wspecifier> "
        "e.g.: \n"
        " codemat-init-code --code-dim=10 ark:feat.scp ark,t:code.ark \n";

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

    std::string feat_rspecifier = po.GetArg(1);
    std::string code_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter code_writer(code_wspecifier);

    Matrix<BaseFloat> code;

    int32 num_utt = 0;
    for(; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &mat = feat_reader.Value();

      code.Resize(mat.NumRows(), code_dim, kSetZero);

      if(add_gauss_noise){
        code.SetRandn();
      }

      code_writer.Write(key, code);
      ++num_utt;
    }

    KALDI_LOG << "Created " << num_utt << (add_gauss_noise? " random ":"") << " codes.";

    return 0;

  } catch (const std::exception &e){
    std::cerr << e.what();
    return -1;
  }
}



