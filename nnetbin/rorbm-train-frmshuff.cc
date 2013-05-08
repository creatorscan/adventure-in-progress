// rbmbin/rbm-train-cd1-frmshuff.cc

// Copyright 2012  Karel Vesely

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet/nnet-rorbm.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-rand.h"
#include "cudamatrix/cu-math.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform RoRbm training using SAP.\n"
            "Usage:  rorbm-train-frmshuff [options] <model-in> <feature-rspecifier> <model-out>\n"
            "e.g.: \n"
            " rorbm-train-frmshuff rorbm.init scp:train.scp rorbm.out\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    BaseFloat learn_rate = 0.008,
        init_momentum = 0.0, final_momentum = 0.0,
        l2_penalty = 0.0,
        z_momentum = 0.0;

    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("init-momentum", &init_momentum, "Initial momentum");
    po.Register("final-momentum", &final_momentum, "Final momentum");
    po.Register("l2-penalty", &l2_penalty, "L2 penalty (weight decay)");
    po.Register("z-momentum", &z_momentum, "Momentum for z");

    int32 final_momentum_start_iter = -1, num_gibbs_iters = 1,
        num_pos_iters = 1, z_start_iter = -1;

    po.Register("final-momentum-start-iter", &final_momentum_start_iter,
                "Final momentum starting iteration");
    po.Register("num-gibbs-iters", &num_gibbs_iters,
                "Number of Gibbs iterations");
    po.Register("num-pos-iters", &num_pos_iters,
                "Number of iterations for positive statistics accumulation");
    po.Register("z-start-iter", &z - start - iter,
                "Iteration to start z accumulation");

    int32 bunchsize = 512, cachesize = 32768;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling");

    int32 max_epoch = 100;
    po.Register("max-epoch", &max_epoch,
                "Maximum number of epochs for training");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2);

    std::string target_model_filename;
    target_model_filename = po.GetArg(3);

    Nnet nnet;
    nnet.Read(model_filename);
    KALDI_ASSERT(nnet.LayerCount()==1);
    KALDI_ASSERT(nnet.Layer(0)->GetType() == Component::kRoRbm);
    RoRbm &rbm = dynamic_cast<RoRbm&>(*nnet.Layer(0));

    rbm.SetL2Penalty(l2_penalty);
    rbm.SetZMomentum(z_momentum);

    rbm.SetNumGibbsIters(num_gibbs_iters);
    rbm.SetNumPosIters(num_pos_iters);
    rbm.SetZStartIter(z_start_iter);

    Timer train_timer;
    KALDI_LOG<< "RBM TRAINING STARTED";

    for (int32 iter = 0; iter < max_epoch; ++iter) {

      kaldi::int64 tot_t = 0;

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

      Cache cache;
      cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
      cache.Init(cachesize, bunchsize);

      MseProgress mse;

      CuMatrix<BaseFloat> feats, data_noisy, data_recon;
      CuMatrix<BaseFloat> dummy_mse_mat;
      std::vector<int32> dummy_cache_vec;

      Timer tim;
      double time_next = 0;

      BaseFloat dd = 0.99 * learn_rate / max_epoch;
      BaseFloat cur_lr = learn_rate - dd * iter;
      rbm.SetLearnRate(cur_lr);

      if (iter > final_momentum_start_iter)
        rbm.SetMomentum(final_momentum);
      else
        rbm.SetMomentum(init_momentum);

      KALDI_LOG<< "######################################################################";
      KALDI_LOG<< "#####   Epoch " << iter << " learn_rate: " << cur_lr;

      int32 num_done = 0, num_cache = 0;
      while (1) {
        // fill the cache
        while (!cache.Full() && !feature_reader.Done()) {
          // get feature matrix
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          // resize the dummy vector to fill Cache:: with
          dummy_cache_vec.resize(mat.NumRows());
          // push features to GPU
          feats.CopyFromMat(mat);
          // add to cache
          cache.AddData(feats, dummy_cache_vec);
          num_done++;
          // next feature file...
          Timer t_features;
          feature_reader.Next();
          time_next += t_features.Elapsed();
        }
        // randomize
        cache.Randomize();
        // report
        std::cerr << "Cache #" << ++num_cache << " "
            << (cache.Randomized() ? "[RND]" : "[NO-RND]")
            << " segments: " << num_done
            << " frames: " << tot_t << "\n";
        // train with the cache
        while (!cache.Empty()) {
          // get block of feature/target pairs
          cache.GetBunch(&data_noisy, &dummy_cache_vec);

          // TRAIN with CD1
          // forward pass
          rbm.Learn(data_noisy, data_recon);

          // evaluate mean square error
          mse.Eval(data_noisy, data_recon, &dummy_mse_mat);

          tot_t += data_noisy.NumRows();
        }

        // stop training when no more data
        if (feature_reader.Done())
          break;
      }

      feature_reader.Close();

      nnet.Write(target_model_filename, binary);

      std::cout << "\n" << std::flush;

      KALDI_LOG<< "RBM TRAINING FINISHED "
      << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
      << ", feature wait " << time_next << "s";

      KALDI_LOG<< "Done " << num_done << " files.";

      KALDI_LOG<< mse.Report();

    }

    KALDI_LOG<< "RBM TRAINING FINISHED "
    << train_timer.Elapsed() << "s, for " << max_epoch << " epochs.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
