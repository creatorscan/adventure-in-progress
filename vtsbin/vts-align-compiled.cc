/*
 * vts-align-compiled.cc
 *
 *  Created on: Oct 30, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Alignment using VTS compensated model.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc
#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Align features given VTS compensated GMM-based models.\n"
            "Usage:   vts-align-compiled [options] model-in graphs-rspecifier feature-rspecifier "
            "noise-rspecifier alignments-wspecifier [<score-wspecifier>]\n"
            "e.g.: \n"
            " vts-align-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:noise.ark ark:1.ali\n"
            "or:\n"
            " compile-train-graphs tree 1.mdl lex.fst ark:train.tra b, ark:- | \\\n"
            "   vts-align-compiled 1.mdl ark:- scp:train.scp ark:noise.ark t, ark:1.ali\n";

    ParseOptions po(usage);
    bool binary = true;
    BaseFloat beam = 200.0;
    BaseFloat retry_beam = 0.0;
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("beam", &beam, "Decoding beam");
    po.Register("retry-beam", &retry_beam,
                "Decoding beam for second try at alignment");
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register(
        "self-loop-scale",
        &self_loop_scale,
        "Scale of self-loop versus non-self-loop log probs [relative to acoustics]");
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");
    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 6) {
      po.PrintUsage();
      exit(1);
    }
    if (retry_beam != 0 && retry_beam <= beam)
      KALDI_WARN << "Beams do not make sense: beam " << beam << ", retry-beam "
          << retry_beam;

    FasterDecoderOptions decode_opts;
    decode_opts.beam = beam;  // Don't set the other options.

    std::string model_in_filename = po.GetArg(1);
    std::string fst_rspecifier = po.GetArg(2);
    std::string feature_rspecifier = po.GetArg(3);
    std::string noise_rspecifier = po.GetArg(4);
    std::string alignment_wspecifier = po.GetArg(5);
    std::string scores_wspecifier = po.GetOptArg(6);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessDoubleVectorReader noiseparams_reader(noise_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier);
    BaseFloatWriter scores_writer(scores_wspecifier);

    int num_success = 0, num_no_feat = 0, num_other_error = 0;
    BaseFloat tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      if (!feature_reader.HasKey(key)) {
        num_no_feat++;
        KALDI_WARN << "No features for utterance " << key;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(key);

        if (!noiseparams_reader.HasKey(key + "_mu_h")
            || !noiseparams_reader.HasKey(key + "_mu_z")
            || !noiseparams_reader.HasKey(key + "_var_z")) {
          KALDI_ERR
              << "Not all the noise parameters (mu_h, mu_z, var_z) are available!";
        }

        int feat_dim = features.NumCols();
        if (feat_dim != 39) {
          KALDI_ERR
              << "Could not decode the features, only 39D MFCC_0_D_A is supported!";
        }

        /************************************************
         Extract the noise parameters
         *************************************************/

        Vector<double> mu_h(noiseparams_reader.Value(key + "_mu_h"));
        Vector<double> mu_z(noiseparams_reader.Value(key + "_mu_z"));
        Vector<double> var_z(noiseparams_reader.Value(key + "_var_z"));

        if (g_kaldi_verbose_level >= 1) {
          KALDI_LOG << "Additive Noise Mean: " << mu_z;
          KALDI_LOG << "Additive Noise Covariance: " << var_z;
          KALDI_LOG << "Convoluational Noise Mean: " << mu_h;
        }
        /************************************************
         Compensate the model
         *************************************************/

        AmDiagGmm noise_am_gmm;
        // Initialize with the clean speech model
        noise_am_gmm.CopyFromAmDiagGmm(am_gmm);

        std::vector<Matrix<double> > Jx(am_gmm.NumGauss()), Jz(
            am_gmm.NumGauss());  // not necessary for compensation only
        CompensateModel(mu_h, mu_z, var_z, num_cepstral, num_fbank, dct_mat,
                        inv_dct_mat, noise_am_gmm, Jx, Jz);


        VectorFst < StdArc > decode_fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << key;
          num_other_error++;
          continue;
        }
        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << key;
          num_other_error++;
          continue;
        }

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms, transition_scale,
                             self_loop_scale, &decode_fst);
        }

        // SimpleDecoder decoder(decode_fst, beam);
        FasterDecoder decoder(decode_fst, decode_opts);
        // makes it a bit faster: 37 sec -> 26 sec on 1000 RM utterances @ beam 200.

        DecodableAmDiagGmmScaled gmm_decodable(noise_am_gmm, trans_model, features,
                                               acoustic_scale);
        decoder.Decode(&gmm_decodable);

        VectorFst<LatticeArc> decoded;  // linear FST.
        bool ans = decoder.ReachedFinal()  // consider only final states.
        && decoder.GetBestPath(&decoded);
        if (!ans && retry_beam != 0.0) {
          KALDI_WARN << "Retrying utterance " << key << " with beam "
              << retry_beam;
          decode_opts.beam = retry_beam;
          decoder.SetOptions(decode_opts);
          decoder.Decode(&gmm_decodable);
          ans = decoder.ReachedFinal()  // consider only final states.
          && decoder.GetBestPath(&decoded);
          decode_opts.beam = beam;
          decoder.SetOptions(decode_opts);
        }
        if (ans) {
          std::vector<int32> alignment;
          std::vector<int32> words;
          LatticeWeight weight;
          frame_count += features.NumRows();

          GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
          BaseFloat like = -(weight.Value1() + weight.Value2())
              / acoustic_scale;
          tot_like += like;
          if (scores_writer.IsOpen())
            scores_writer.Write(key, -(weight.Value1() + weight.Value2()));
          alignment_writer.Write(key, alignment);
          num_success++;
          if (num_success % 50 == 0) {
            KALDI_LOG << "Processed " << num_success << " utterances, "
                << "log-like per frame for " << key << " is "
                << (like / features.NumRows()) << " over " << features.NumRows()
                << " frames.";
          }
        } else {
          KALDI_WARN << "Did not successfully decode file " << key << ", len = "
              << (features.NumRows());
          num_other_error++;
        }
      }
    }
    KALDI_LOG << "Overall log-likelihood per frame is "
        << (tot_like / frame_count) << " over " << frame_count << " frames.";
    KALDI_LOG << "Done " << num_success << ", could not find features for "
        << num_no_feat << ", other errors on " << num_other_error;
    if (num_success != 0)
      return 0;
    else
      return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

