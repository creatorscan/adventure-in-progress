/*
 * codebl-create.cc
 *
 *  Created on: Oct 5, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Create <codebl> layer from <biasedlinearity>. The code matrix is set to 0.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-codebl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Creat <codebl> from a <biasedlinearity> layer\n"
            "Usage: codebl-create [options] <bl-in> <codebl-out>"
            "e.g.: \n"
            " codebl-create --binary=false bl.nnet codebl.nnet \n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    int32 code_dim = 0;
    po.Register("code-dim", &code_dim, "Dimension of the code vector");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string bl_filename = po.GetArg(1),
        codebl_filename = po.GetArg(2);

    Nnet nnet;
    {
      bool binary_read;
      Input ki(bl_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);

      KALDI_LOG<< "Load model from " << bl_filename;
    }

    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT((nnet.Layer(0))->GetType() == Component::kBiasedLinearity);

    BiasedLinearity *bl = static_cast<BiasedLinearity*>(nnet.Layer(0));
    int32 bl_in_dim = bl->InputDim();
    int32 bl_out_dim = bl->OutputDim();

    Matrix<BaseFloat> linearity(bl_out_dim, bl_in_dim + code_dim, kSetZero),
        bl_linearity;
    Vector<BaseFloat> bias(bl_out_dim);

    (bl->GetLinearityWeight()).CopyToMat(&bl_linearity);
    (bl->GetBiasWeight()).CopyToVec(&bias);

    (SubMatrix<BaseFloat>(linearity, 0, bl_out_dim, code_dim, bl_in_dim))
        .CopyFromMat(bl_linearity);

    // write the output
    {
      Output ko(codebl_filename, binary);

      WriteToken(ko.Stream(), binary,
                 Component::TypeToMarker(Component::kCodeBL));
      WriteBasicType(ko.Stream(), binary, bl_out_dim);
      WriteBasicType(ko.Stream(), binary, bl_in_dim);
      if (!binary)
        ko.Stream() << "\n";
      WriteBasicType(ko.Stream(), binary, code_dim);
      if (!binary)
        ko.Stream() << "\n";
      linearity.Write(ko.Stream(), binary);
      bias.Write(ko.Stream(), binary);

      KALDI_LOG<< "Write model to " << codebl_filename;
    }

  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

