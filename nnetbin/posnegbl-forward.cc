// nnetbin/posnegbl-forward.cc

// forward through the PosNegBL layer

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-posnegbl.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
            "Usage: posnegbl-forward [options] <model-in> <feature-rspecifier> "
            "<feature-wspecifier> [<noise-rspecifier>]\n"
            "e.g.: \n"
            " posnegbl-forward nnet ark:features.ark ark:out.ark ark:noise.ark\n";

    ParseOptions po(usage);

    bool compensate_var = true;
    po.Register(
        "compensate-var",
        &compensate_var,
        "Whether apply VTS compensation to covariance. Mean will always be compensated");

    bool shared_var = true;
    po.Register(
        "shared-var",
        &shared_var,
        "Whether after compensation the variances are constrained to be the same");

    BaseFloat positive_var_weight = 1.0;
    po.Register(
        "positive-var-weight",
        &positive_var_weight,
        "Only effective when shared-var=true, the weight ratio for positive variance");

    bool silent = false;
    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    bool have_noise = false;
    if (po.NumArgs() == 4) {
      have_noise = true;
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3),
        noise_rspecifier = po.GetOptArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet;
    nnet.Read(model_filename);

    KALDI_ASSERT(nnet.LayerCount() == 1);
    Component *comp = nnet.Layer(0);
    KALDI_ASSERT(comp->GetType() == Component::kPosNegBL);

    PosNegBL *layer = (PosNegBL*) comp;

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    Matrix<BaseFloat> nnet_out;

    Timer tim;
    if (!silent)
      KALDI_LOG<< "POSNEGBL FEEDFORWARD STARTED";

    int32 num_done = 0;
    int32 outdim = layer->OutputDim();
    // iterate over all the feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      nnet_out.Resize(feats.NumRows(), outdim, kSetZero);

      if (have_noise) {
        // read noise parameters
        if (!noiseparams_reader.HasKey(key + "_mu_h")
            || !noiseparams_reader.HasKey(key + "_mu_z")
            || !noiseparams_reader.HasKey(key + "_var_z")) {
          KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }
        Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));
        if (g_kaldi_verbose_level >= 1) {
          KALDI_LOG<< "Additive Noise Mean: " << mu_z;
          KALDI_LOG << "Additive Noise Covariance: " << var_z;
          KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
        }

        layer->SetNoise(compensate_var, mu_h, mu_z, var_z, positive_var_weight);
      }

      layer->Forward(feats, &nnet_out);

      //check for NaN/inf
      for (int32 r = 0; r < nnet_out.NumRows(); r++) {
        for (int32 c = 0; c < nnet_out.NumCols(); c++) {
          BaseFloat val = nnet_out(r, c);
          if (val != val)
            KALDI_ERR<< "NaN in NNet output of : " << key;
          if (val == std::numeric_limits < BaseFloat > ::infinity())
            KALDI_ERR<< "inf in NNet coutput of : " << key;
          }
        }

      // write
      feature_writer.Write(feature_reader.Key(), nnet_out);

      // progress log
      if (num_done % 100 == 0) {
        if (!silent)
          KALDI_LOG<< num_done << ", " << std::flush;
        }
      num_done++;
      tot_t += feats.NumRows();
    }

    // final message
    if (!silent)
      KALDI_LOG<< "MLP FEEDFORWARD FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed();
    if (!silent)
      KALDI_LOG<< "Done " << num_done << " files";

    return ((num_done > 0) ? 0 : 1);
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}
