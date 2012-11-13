/*
 * vtsbin/dbnvts2-compensate-model.cc
 *
 *  Created on: Nov 13, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * This tool is mainly for testing the dbnvts2-forward's noise compensation
 * functionalities.
 *
 * The compensated pos and neg models will be written out for analysis.
 *
 * The noise parameters must be global.
 *
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

#include "vts/vts-first-order.h"
#include "vts/dbnvts2-first-order.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Genearated the noise compensated positive and negative models.\n"
            "Usage:  dbnvts2-compensate-model [options] <pos-model-in> <neg-model-in>"
            " <var-scale-in> <noise-model-rspecifier> <pos-model-out> <neg-model-out>\n"
            "e.g.: \n"
            " dbnvts2-compensate-model pos.mdl neg.mdl var_scale.stats ark:noise_params.ark"
            " noise_pos.mdl noise_neg.mdl\n";

    ParseOptions po(usage);

    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    std::string noise_key = "";
    po.Register("noise-key", &noise_key,
                "The key in the noise archieve to be used for compensation");

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

    int32 num_frames = 9;
    po.Register("num-frames", &num_frames,
                "Number of frames for the input feature");

    int32 num_cepstral = 13;
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");

    int32 num_fbank = 26;
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");

    BaseFloat ceplifter = 22;
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    bool use_var_scale = false;
    po.Register("use-var-scale", &use_var_scale,
                "Apply the variance scale to the new weight estimation");

    bool post_var_scale = true;
    po.Register(
        "post-var-scale", &post_var_scale,
        "Whether the variance scaling is applied before compensation or after");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string pos_model_filename = po.GetArg(1), neg_model_filename = po
        .GetArg(2), var_scale_filename = po.GetArg(3), noise_params_rspecifier =
        po.GetArg(4), noise_pos_model_wxfilename = po.GetArg(5),
        noise_neg_model_wxfilename = po.GetArg(6);

    // positive AM Gmm
    AmDiagGmm pos_am_gmm;
    {
      bool binary;
      Input ki(pos_model_filename, &binary);
      pos_am_gmm.Read(ki.Stream(), binary);
    }

    // negative AM Gmm
    AmDiagGmm neg_am_gmm;
    {
      bool binary;
      Input ki(neg_model_filename, &binary);
      neg_am_gmm.Read(ki.Stream(), binary);
    }

    KALDI_ASSERT(pos_am_gmm.NumPdfs() == neg_am_gmm.NumPdfs());
    int32 num_pdfs = pos_am_gmm.NumPdfs();

    // noisy models
    AmDiagGmm pos_noise_am, neg_noise_am;

    // variance scale factors
    Vector<double> var_scale;
    {
      bool binary;
      Input ki(var_scale_filename, &binary);
      var_scale.Read(ki.Stream(), binary);
      KALDI_ASSERT(var_scale.Dim() == num_pdfs);
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    RandomAccessDoubleVectorReader noiseparams_reader(noise_params_rspecifier);

    // read noise parameters
    if (!noiseparams_reader.HasKey(noise_key + "_mu_h")
        || !noiseparams_reader.HasKey(noise_key + "_mu_z")
        || !noiseparams_reader.HasKey(noise_key + "_var_z")) {
      KALDI_ERR<< "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
    }
    Vector<double> mu_h(noiseparams_reader.Value(noise_key + "_mu_h"));
    Vector<double> mu_z(noiseparams_reader.Value(noise_key + "_mu_z"));
    Vector<double> var_z(noiseparams_reader.Value(noise_key + "_var_z"));
    if (g_kaldi_verbose_level >= 1) {
      KALDI_LOG<< "Additive Noise Mean: " << mu_z;
      KALDI_LOG << "Additive Noise Covariance: " << var_z;
      KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
    }

      // compensate the postive and negative gmm models
    pos_noise_am.CopyFromAmDiagGmm(pos_am_gmm);
    neg_noise_am.CopyFromAmDiagGmm(neg_am_gmm);

    if (use_var_scale && !post_var_scale) {
      ScaleVariance(var_scale, pos_noise_am, neg_noise_am);
    }

    CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                            num_fbank, dct_mat, inv_dct_mat, num_frames,
                            pos_noise_am);

    CompensateMultiFrameGmm(mu_h, mu_z, var_z, compensate_var, num_cepstral,
                            num_fbank, dct_mat, inv_dct_mat, num_frames,
                            neg_noise_am);

    if (shared_var) {
      // set the covariance to be the same for pos and neg
      InterpolateVariance(positive_var_weight, pos_noise_am, neg_noise_am);
    }

    // post compensation scaling
    if (use_var_scale && post_var_scale) {
      ScaleVariance(var_scale, pos_noise_am, neg_noise_am);
    }

    {
      Output ko(noise_pos_model_wxfilename, binary);
      pos_noise_am.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Write positive noisy Gmm model.";

    {
      Output ko(noise_neg_model_wxfilename, binary);
      neg_noise_am.Write(ko.Stream(), binary);
    }
    KALDI_LOG<< "Write negative noisy Gmm model.";

    KALDI_LOG<< "Noise Compensation Done!";

    return 0;
  } catch (const std::exception &e) {
    KALDI_ERR<< e.what();
    return -1;
  }
}

