/*
 * codeat-train-code.cc
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Learning the codes using stochastic gradients.
 */

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-codeat.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform one iteration of code learning by stochastic gradient descent.\n"
            "Usage:  codeat-train-code [options] <adapt-model-in> <back-model-in>"
            " <feature-rspecifier> <alignments-rspecifier> <set2utt-rspecifier> "
            "<code-rspecifier> [<code-wspecifier>]\n"
            "e.g.: \n"
            " codeat-train-code adapt.nnet back.nnet scp:train.scp ark:train.ali "
            "ark:set2utt.ark ark:code_init.ark ark:code_iter1.ark\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts, dummy_opts;
    trn_opts.Register(&po);

    bool binary = true,
        crossvalidate = false,
        randomize = true,
        shuffle = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");
    po.Register("shuffle", &shuffle,
                "Perform the utterance-level shuffling");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
                "Feature transform in Nnet format");

    int32 bunchsize = 512, cachesize = 32768, seed = 777;
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize,
                "Size of cache for frame level shuffling (max 8388479)");
    po.Register("seed", &seed,
                "Seed value for srand, sets fixed order of frame-shuffling");

    kaldi::int32 max_frames = 6000;  // Allow segments maximum of one minute by default
    po.Register("max-frames", &max_frames,
                "Maximum number of frames a segment can have to be processed");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id = 0;
    po.Register("use-gpu-id", &use_gpu_id,
                "Unused, kaldi is compiled w/o CUDA");
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 7 - (crossvalidate ? 1 : 0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string adapt_model_filename = po.GetArg(1),
        back_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        alignments_rspecifier = po.GetArg(4),
        set2utt_rspecifier = po.GetArg(5),
        code_rspecifier = po.GetArg(6);

    std::string code_wspecifier = "";
    if (!crossvalidate) {
      code_wspecifier = po.GetArg(4);
    }

    //set the seed to the pre-defined value
    srand(seed);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf, nnet_back;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    nnet_back.Read(back_model_filename);
    // just to disable the update of the back-nnet
    dummy_opts.learn_rate = 0.0;
    dummy_opts.momentum = 0.0;
    dummy_opts.l1_penalty = 0.0;
    dummy_opts.l2_penalty = 0.0;
    nnet_back.SetTrainOptions(dummy_opts);

    Nnet nnet;
    nnet.Read(adapt_model_filename);
    nnet.SetTrainOptions(trn_opts);

    /*
     * Find out all the <codeat> layers.
     */
    int32 num_codeat = 0, code_dim = 0;
    std::vector<CodeAT*> layers_codeat;
    for (int32 c = 0; c < nnet.NumComponents(); ++c) {
      if (nnet.GetComponent(c).GetType() == Component::kCodeAT) {
        layers_codeat.push_back(dynamic_cast<CodeAT*>(&(nnet.GetComponent(c))));
        // disable weight update, only update code
        layers_codeat[num_codeat]->DisableWeightUpdate();
        if (code_dim == 0) {
          code_dim = layers_codeat[num_codeat]->GetCodeDim();
        } else if (code_dim != layers_codeat[num_codeat]->GetCodeDim()) {
          KALDI_ERR<< "Inconsistent code dimensions for <codeat> layers in " << adapt_model_filename;
        }
        ++num_codeat;
      }
    }
    KALDI_LOG<< "Totally " << num_codeat << " among " << nnet.NumComponents() << " layers of the nnet are <codeat> layers.";

    kaldi::int64 total_frames = 0;

    SequentialTokenVectorReader set2utt_reader(set2utt_rspecifier);
    RandomAccessBaseFloatVectorReader code_reader(code_rspecifier);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    BaseFloatVectorWriter code_writer(code_wspecifier);

    Cache cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Xent xent;

    CuVector<BaseFloat> code_diff(code_dim);
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, back_out,
        obj_diff, back_diff, nnet_diff;
    std::vector<int32> targets;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0,
        num_cache = 0, num_set = 0;
    for (; !set2utt_reader.Done(); set2utt_reader.Next()) {
      std::string setkey = set2utt_reader.Key();
      if (!code_reader.HasKey(setkey)) {
        KALDI_ERR<< "No code for set " << setkey;
      }
      KALDI_LOG<< "Set # " << ++num_set << " - " << setkey << ":";
      // copy the code to GPU
      Vector<BaseFloat> code(code_reader.Value(setkey));
      // update the nnet's codeat layers with new code
      for (int32 c = 0; c < num_codeat; ++c) {
        layers_codeat[c]->SetCode(code);
        // also clean the code_corr so that the gradient doesn't accumulate
        // through different sets
        layers_codeat[c]->ZeroCodeCorr();
      }

      // all the utterances belong to this set
      std::vector<std::string> uttlst(set2utt_reader.Value());
      if (shuffle) {
        std::random_shuffle(uttlst.begin(), uttlst.end());
      }

      for (int32 uid = 0; uid < uttlst.size();) {

        // fill the cache
        while (!cache.Full() && uid < uttlst.size()) {
          std::string utt = uttlst[uid];
          KALDI_VLOG(2) << "Reading utt " << utt;
          // check that we have alignments
          if (!alignments_reader.HasKey(utt)) {
            num_no_alignment++;
            uid++;
            continue;
          }
          // get feature alignment pair
          const Matrix<BaseFloat> &mat = feature_reader.Value(utt);
          const std::vector<int32> &alignment = alignments_reader.Value(utt);
          // check maximum length of utterance
          if (mat.NumRows() > max_frames) {
            KALDI_WARN<< "Utterance " << utt << ": Skipped because it has " << mat.NumRows() <<
            " frames, which is more than " << max_frames << ".";
            num_other_error++;
            uid++;
            continue;
          }
            // check length match of features/alignments
          if ((int32) alignment.size() != mat.NumRows()) {
            KALDI_WARN<< "Alignment has wrong length, ali " << (alignment.size())
            << " vs. feats "<< (mat.NumRows()) << ", utt " << utt;
            num_other_error++;
            uid++;
            continue;
          }

            // All the checks OK,
            // push features to GPU
          feats.Resize(mat.NumRows(), mat.NumCols(), kUndefined);
          feats.CopyFromMat(mat);
          // possibly apply transform
          nnet_transf.Feedforward(feats, &feats_transf);
          // add to cache
          cache.AddData(feats_transf, alignment);
          num_done++;

          // measure the time needed to get next feature file
          Timer t_features;
          uid++;
          time_next += t_features.Elapsed();
          // report the speed
          if (num_done % 1000 == 0) {
            time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done
                << " utterances: time elapsed = "
                << time_now / 60 << " min; processed "
                << total_frames / time_now
                << " frames per second.";
          }

        }
        // randomize
        if (!crossvalidate && randomize) {
          cache.Randomize();
        }
        // report
        KALDI_VLOG(1) << "Cache #" << ++num_cache << " "
            << (cache.Randomized() ? "[RND]" : "[NO-RND]")
            << " segments: " << num_done
            << " frames: " << static_cast<double>(total_frames) / 360000 << "h";
        // train with the cache
        while (!cache.Empty()) {
          // get block of feature/target pairs
          cache.GetBunch(&nnet_in, &targets);
          // train
          nnet.Propagate(nnet_in, &nnet_out);
          nnet_back.Propagate(nnet_out, &back_out);

          xent.EvalVec(back_out, targets, &obj_diff);
          if (!crossvalidate) {
            nnet_back.Backpropagate(obj_diff, &back_diff);
            nnet.Backpropagate(back_diff, &nnet_diff);

            // accumulate code corr through different layers
            code_diff.SetZero();
            for (int32 c = 0; c < num_codeat; ++c) {
              code_diff.AddVec(1.0, layers_codeat[c]->GetCodeDiff(), 1.0);
            }
            code_diff.Scale(1.0 / num_codeat);
            // update the code
            for (int32 c = 0; c < num_codeat; ++c) {
              layers_codeat[c]->UpdateCode(code_diff);
            }
          }
          total_frames += nnet_in.NumRows();
        }

      }  // end for uttlst

      if (!crossvalidate) {
        (layers_codeat[0]->GetCode()).CopyToVec(&code);
        code_writer.Write(setkey, code);
      }

    }  // end for set2utt

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG << "Done " << num_set << " sets.";
    KALDI_LOG<< "Done " << num_done << " files, " << num_no_alignment
    << " with no alignments, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

