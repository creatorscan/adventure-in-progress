/*
 * nnet/nnet-hidmask-train-frmshuff.cc
 *
 *  Created on: Oct 1, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-xent-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

#define MAX_HIDMASK_NNETS 10

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform iteration of Neural Network training by stochastic gradient descent.\n"
            "Usage: nnet-hidmask-train-frmshuff [options] <model-in-l1> <model-in-backend> <noisy-feature-rspecifier> "
            "<clean-feature-rspecifier> <alignments-rspecifier> [<model-out-l1> <model-out-backend>]\n"
            "e.g.: \n "
            " nnet-hidmask-train-frmshuff l1.nnet backend.nnet scp:train_noisy.scp scp:train_clean.scp "
            "ark:train.ali l1.iter1 backend.iter1\n";

    ParseOptions po(usage);
    bool binary = false,
        cross_validate = false,
        randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &cross_validate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");

    BaseFloat learn_rate = 0.008,
        momentum = 0.0,
        l2_penalty = 0.0,
        l1_penalty = 0.0;
    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("momentum", &momentum, "Momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("l1-penalty", &l1_penalty, "L1 penalty (promte sparsity)");

    bool average_grad = false;
    po.Register("average-grad", &average_grad,
                "Whether to average the gradient in the bunch");

    std::string learn_factors = "";
    po.Register("learn-factors", &learn_factors,
                "Learning factors for backend Nnet");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform Neural Network");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    po.Read(argc, argv);

    if (po.NumArgs() != 7 - (cross_validate ? 2 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_hidmask_l1_filename = po.GetArg(1),
        model_backend_filename = po.GetArg(2),
        noisyfeats_rspecifier = po.GetArg(3),
        cleanfeats_rspecifier = po.GetArg(4),
        alignments_rspecifier = po.GetArg(5);

    std::string target_hidmask_l1_filename, target_backend_filename;
    if (!cross_validate) {
      target_hidmask_l1_filename = po.GetArg(6);
      target_backend_filename = po.GetArg(7);
    }

    /*
     * Load Nnets
     */
    // feature transformation nnet
    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    // each hidmask nnet contains only one <biasedlinearity> layer and other activation layers.
    Nnet nnet_hidmask[MAX_HIDMASK_NNETS];
    nnet_hidmask[0].Read(model_hidmask_l1_filename);
    nnet_hidmask[0].SetLearnRate(learn_rate, NULL);
    nnet_hidmask[0].SetMomentum(momentum);
    nnet_hidmask[0].SetL2Penalty(l2_penalty);
    nnet_hidmask[0].SetL1Penalty(l1_penalty);
    nnet_hidmask[0].SetAverageGrad(average_grad);

    Nnet nnet_backend;
    nnet_backend.Read(model_backend_filename);
    if (learn_factors == "") {
      nnet_backend.SetLearnRate(learn_rate, NULL);
    } else {
      nnet_backend.SetLearnRate(learn_rate, learn_factors.c_str());
    }
    nnet_backend.SetMomentum(momentum);
    nnet_backend.SetL2Penalty(l2_penalty);
    nnet_backend.SetL1Penalty(l1_penalty);
    nnet_backend.SetAverageGrad(average_grad);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader noisyfeats_reader(noisyfeats_rspecifier);
    SequentialBaseFloatMatrixReader cleanfeats_reader(cleanfeats_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    CacheXentTgtMat cache;  // using the tgtMat to save the clean feats
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    CuMatrix<BaseFloat> noisy_feats, clean_feats, noisy_feats_transf,
        clean_feats_transf,
        backend_in, backend_out, glob_err;
    CuMatrix<BaseFloat> nnet_noisy_in[MAX_HIDMASK_NNETS],
        nnet_clean_in[MAX_HIDMASK_NNETS], hid_masks[MAX_HIDMASK_NNETS],
        layer_err[MAX_HIDMASK_NNETS];
    std::vector<int32> nnet_labs;

    Timer tim;
    double time_next = 0;
    KALDI_LOG<< (cross_validate? "CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_align = 0, num_other_error = 0, num_cache = 0;
    while (1) {
      // fill the cache
      while (!cache.Full() && !noisyfeats_reader.Done()
          && !cleanfeats_reader.Done()) {
        std::string noisy_key = noisyfeats_reader.Key();
        std::string clean_key = cleanfeats_reader.Key();
        // stopping if the keys do not match
        if (noisy_key != clean_key) {
          KALDI_ERR<< "Key mismatches for parallel data: " << noisy_key << " vs. " << clean_key;
        }
          // now we should have a pair
        if (noisy_key == clean_key) {
          // get feature pari
          const Matrix<BaseFloat> &noisy_mats = noisyfeats_reader.Value();
          const Matrix<BaseFloat> &clean_mats = cleanfeats_reader.Value();
          if (noisy_mats.NumRows() != clean_mats.NumRows()
              || noisy_mats.NumCols() != clean_mats.NumCols()) {
            KALDI_WARN<< "Feature mismatch: " << noisy_key;
            num_other_error++;
            continue;
          }

          if (!alignments_reader.HasKey(noisy_key)) {
            ++num_no_align;
            KALDI_WARN<< "No alignments for: " << noisy_key;
            continue;
          }

          const std::vector<int32> &labs = alignments_reader.Value(noisy_key);
          if (labs.size() != noisy_mats.NumRows()) {
            ++num_other_error;
            KALDI_WARN<< "Alignment has wrong size " << (labs.size()) << " vs. " << (noisy_mats.NumRows());
            continue;
          }

          // push features to GPU
          noisy_feats.CopyFromMat(noisy_mats);
          clean_feats.CopyFromMat(clean_mats);
          // possibly apply feature transform
          nnet_transf.Feedforward(noisy_feats, &noisy_feats_transf);
          nnet_transf.Feedforward(clean_feats, &clean_feats_transf);
          // add to cache
          cache.AddData(noisy_feats_transf, labs, clean_feats_transf);
          num_done++;
        }
        Timer t_features;
        noisyfeats_reader.Next();
        cleanfeats_reader.Next();
        time_next += t_features.Elapsed();
      }
      // randomize
      if (!cross_validate) {
        cache.Randomize();
      }
      // report
      std::cerr << "Cache #" << ++num_cache << " "
          << (cache.Randomized() ? "[RND]" : "[NO-RND]")
          << " segments: " << num_done
          << " frames: " << tot_t << "\n";
      // train with the cache
      while (!cache.Empty()) {
        // get block of features pairs
        cache.GetBunch(&(nnet_noisy_in[0]), &nnet_labs, &(nnet_clean_in[0]));

        // forward through nnet_hidmask
        nnet_hidmask[0].Feedforward(nnet_noisy_in[0], &(nnet_noisy_in[1]));
        backend_in.CopyFromMat(nnet_noisy_in[1]);

        if (!cross_validate) {
          nnet_hidmask[0].Feedforward(nnet_clean_in[0], &(nnet_clean_in[1]));

          // compute hid masks
          hid_masks[1].CopyFromMat(nnet_noisy_in[1]);
          hid_masks[1].AddMat(-1.0, nnet_clean_in[1], 1.0);
          hid_masks[1].Power(2.0);
          hid_masks[1].ApplyExp();

          // apply masks to hidden acts
          backend_in.MulElements(hid_masks[1]);
        }

        // forward through backend nnet
        nnet_backend.Feedforward(backend_in, &backend_out);

        // evaluate
        xent.EvalVec(backend_out, nnet_labs, &glob_err);

        if (!cross_validate) {
          nnet_backend.Backpropagate(glob_err, &(layer_err[1]));

          layer_err[1].MulElements(hid_masks[1]);
          nnet_hidmask[0].Backpropagate(layer_err[1], NULL);

        }
        tot_t += nnet_labs.size();

      }

      // stop training when no more data
      if (noisyfeats_reader.Done() || cleanfeats_reader.Done())
        break;
    }

    if (!cross_validate) {
      nnet_hidmask[0].Write(target_hidmask_l1_filename, binary);
      nnet_backend.Write(target_backend_filename, binary);
    }

    std::cout << "\n" << std::flush;

  KALDI_LOG << (cross_validate? "CROSSVALIDATE":"TRAINING") << " FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
      << ", feature wait " << time_next << "s";

  KALDI_LOG << "Done " << num_done << " files, " << num_no_align
      << " with no alignments, " << num_other_error
      << " with other errors.";

  KALDI_LOG << xent.Report();

#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif


} catch (const std::exception &e) {
  std::cerr << e.what();
  return -1;
}

}

