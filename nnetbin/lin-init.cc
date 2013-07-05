// nnetbin/lin-init.cc

/*
 * Initialize a LIN NN with SI NN.
 *
 */
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-linbl.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Append a LIN to the existing nnet.\n"
            "Usage:  lin-init [options] <nnet-in> <nnet-out>\n"
            "e.g.:\n"
            " append-lin --binary=false nnet.mdl nnet_lin.mdl\n";

    ParseOptions po(usage);

    bool binary_write = false;
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    int32 in_dim = nnet.InputDim();
    {
      Output ko(model_out_filename, binary_write);

      LinBL lin(in_dim, in_dim, NULL);
      lin.SetToIdentity();
      lin.Write(ko.Stream(), binary_write);

      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG<< "Written model to " << model_out_filename;
  }
  catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

