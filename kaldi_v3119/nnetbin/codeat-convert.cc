/*
 * codeat-convert.cc
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Convert all the <affinetransform> of the input nnet to <codeat> layers.
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-codeat.h"
#include "nnet/nnet-affine-transform.h"

int main(int argc, char *argv[]) {
  try {

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert all <affinetransform> layers to <codeat> \n"
            "Usage: codeat-convert [options] <nnet-in> <nnet-out>"
            "e.g.: \n"
            " codeat-convert --binary=false at.nnet codeat.nnet \n";

    ParseOptions po(usage);

    bool binary = false;
    bool gauss_random = true;
    int32 code_dim = 0;
    BaseFloat random_scale = 0.1;

    po.Register("binary", &binary, "Write output in binary format");
    po.Register("gauss-random", &gauss_random,
                "Add Gaussian noise to the code weights");
    po.Register("code-dim", &code_dim, "Code dimension");
    po.Register("random-scale", &random_scale,
                "Scaling factor to the random values");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string in_filename = po.GetArg(1),
        out_filename = po.GetArg(2);

    Nnet nnet;
    {
      bool binary_read;
      Input ki(in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
      KALDI_LOG<< "Load model from " << in_filename;
    }


    // write the output
    {
      Output ko(out_filename, binary);

      for (int32 i = 0; i < nnet.NumComponents(); ++i) {
        if (nnet.GetComponent(i).GetType() == Component::kAffineTransform) {
          AffineTransform& at =
              dynamic_cast<AffineTransform&>(nnet.GetComponent(i));

          int32 at_in_dim = at.InputDim();
          int32 at_out_dim = at.OutputDim();

          Matrix<BaseFloat> linearity(at_out_dim, at_in_dim + code_dim,
                                      kSetZero);
          Matrix<BaseFloat> at_linearity(at_out_dim, at_in_dim, kSetZero);
          Vector<BaseFloat> bias(at_out_dim);

          if (gauss_random) {
            linearity.SetRandn();
            linearity.Scale(random_scale);
          }

          (at.GetLinearity()).CopyToMat(&at_linearity);
          (at.GetBias()).CopyToVec(&bias);

          (SubMatrix<BaseFloat>(linearity, 0, at_out_dim, 0, at_in_dim))
              .CopyFromMat(at_linearity, kNoTrans);

          // Write out the component
          WriteToken(ko.Stream(), binary,
                     Component::TypeToMarker(Component::kCodeAT));
          WriteBasicType(ko.Stream(), binary, at_out_dim);
          WriteBasicType(ko.Stream(), binary, at_in_dim);
          if (!binary)
            ko.Stream() << "\n";
          WriteBasicType(ko.Stream(), binary, code_dim);
          if (!binary)
            ko.Stream() << "\n";
          linearity.Write(ko.Stream(), binary);
          bias.Write(ko.Stream(), binary);

        } else {
          nnet.GetComponent(i).Write(ko.Stream(), binary);
        }
      }

      KALDI_LOG << "Write model to " << out_filename;
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

