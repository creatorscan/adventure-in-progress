/*
 * dbnvts-init-model.cc
 *
 *  Created on: Oct 27, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Initialise the GMM model based on the dimension of the first hidden layer
 *  of the given NNet.
 *
 */

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "gmm/am-diag-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/diag-gmm-normal.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Initialise the GMM with a single Gaussian per hidden unit of the first NN layer and split the nnet.\n"
            "Usage:  dbnvts-init-model [options] <nnet-in> <gmm-out> <biaslinearity-layer-out> <back-nnet-out>\n"
            "e.g.: \n"
            " dbnvts-init-model nnet gmm.mdl bl_layer nnet.back\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary,
                "Write out the model in binary or text mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_filename = po.GetArg(1), model_out_filename = po.GetArg(2),
        bl_layer_out_filename=po.GetArg(3), back_out_filename=po.GetArg(4);

    Nnet nnet;
    nnet.Read(nnet_filename);

    // use the first layer of nnet to create gmm
    Component *layer = nnet.Layer(0);

    KALDI_LOG << "The number of layers in nnet: " << nnet.LayerCount();
    KALDI_LOG << "Layer info for 0: [Type: "
        << layer->TypeToMarker(layer->GetType()) << ", InputDim: "
        << layer->InputDim() << ", OutputDim: " << layer->OutputDim();

    int32 feat_dim = layer->InputDim();
    int32 num_pdfs = layer->OutputDim();

    AmDiagGmm am_gmm;

    // zero mean and unit diagonal variance Gaussian
    Vector<double> mean(feat_dim, kSetZero);
    Vector<double> var(feat_dim, kSetZero);
    var.Set(1.0);

    DiagGmmNormal ngmm;
    ngmm.Resize(1, feat_dim);
    ngmm.weights_(0) = 1.0;
    ngmm.means_.CopyRowFromVec(mean, 0);
    ngmm.vars_.CopyRowFromVec(var, 0);

    DiagGmm gmm(1, feat_dim);
    ngmm.CopyToDiagGmm(&gmm);

    // create gmm
    am_gmm.Init(gmm, num_pdfs);
    am_gmm.ComputeGconsts();

    // write out the gmm model
    {
      Output ko(model_out_filename, binary);
      am_gmm.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Wrote gmm model.";

    // write out the biasedlineary layer
    {
      Output ko(bl_layer_out_filename, binary);
      (nnet.Layer(0))->Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Write front nnet model.";

    // write out the other layers, starts with the sigmoid layer
    {
      Output ko(back_out_filename, binary);
      for(int32 i=1; i<nnet.LayerCount(); ++i){
        (nnet.Layer(i))->Write(ko.Stream(), binary);
      }
    }
    KALDI_LOG << "Write back nnet model.";

    return 1;
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
