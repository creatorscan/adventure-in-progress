/*
 * codeat-train-parallel.cc
 *
 *  Created on: Nov. 5, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Learning the <codeat> layers using stochastic gradients,
 *      Using parallel data to guide the learning of noise parameters.
 *      Needs to specify which parameters to update.
 *      The data lists and the set2utt mapping should have the same utterance order.
 */

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
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
        "Perform one iteration of <codeat> learning to reduce parallel data hidden activation mismatches by stochastic gradient descent.\n"
            "Usage:  codeat-train [options] <ref-adapt-model> <ref-feature-rspecifier> <adapt-model-in> <feature-rspecifier>"
            "  <set2utt-rspecifier> <code-rspecifier>\n"
            "e.g.: \n"
            " codeat-train --update-weight=false --update-code-xform=true --update-code-vec=true "
            " --out-adapt-filename=adapt_iter1.nnet --code-vec-wspecifier=ark:code_iter1.ark "
            " adapt_ref.nnet ark:train_ref.scp adapt.nnet scp:train.scp ark:set2utt.ark ark:code_init.ark\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts, dummy_opts;
    trn_opts.Register(&po);

    bool binary = true,
        crossvalidate = false,
        randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate,
                "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize,
                "Perform the frame-level shuffling within the Cache::");

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

    bool update_weight = false,
         update_code_xform = false,
         update_code_vec = false;
    po.Register("update-weight", &update_weight, "Update the weight and bias of the layer");
    po.Register("update-code-xform", &update_code_xform, "Update the code transformation");
    po.Register("update-code-vec", &update_code_vec, "Update the code vector");

    std::string out_adapt_filename = "",
                code_vec_wspecifier = "";
    po.Register("out-adapt-filename", &out_adapt_filename, "Output adapt nnet file name");
    po.Register("code-vec-wspecifier", &code_vec_wspecifier, "Output code vector archive");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id = 0;
    po.Register("use-gpu-id", &use_gpu_id,
                "Unused, kaldi is compiled w/o CUDA");
#endif

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    if(!crossvalidate && !update_weight && !update_code_xform && !update_code_vec) {
      KALDI_ERR << "All the updates are disabled! Exiting ...";
    }
    if(!crossvalidate && (update_weight || update_code_xform) && out_adapt_filename == "") {
      KALDI_ERR << "No output adapt nnet file is specified for learning!";
    }
    if(!crossvalidate && update_code_vec && code_vec_wspecifier == "") {
      KALDI_ERR << "No output code archive is specified for learning";
    }

    std::string ref_adapt_model_filename = po.GetArg(1),
      ref_feature_rspecifier = po.GetArg(2),
      adapt_model_filename = po.GetArg(3),
      feature_rspecifier = po.GetArg(4),
      set2utt_rspecifier = po.GetArg(5),
      code_vec_rspecifier = po.GetArg(6);

    //set the seed to the pre-defined value
    srand(seed);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf, nnet_ref;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    nnet_ref.Read(ref_adapt_model_filename);

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
        layers_codeat[num_codeat]->ConfigureUpdate(update_weight, update_code_xform, update_code_vec);
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
    RandomAccessBaseFloatVectorReader code_vec_reader(code_vec_rspecifier);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader ref_feature_reader(ref_feature_rspecifier);

    BaseFloatVectorWriter code_vec_writer(code_vec_wspecifier);

    CacheTgtMat cache;
    cachesize = (cachesize / bunchsize) * bunchsize;  // ensure divisibility
    cache.Init(cachesize, bunchsize);

    Mse mse;

    Vector<BaseFloat> code(code_dim, kSetZero);

    CuMatrix<BaseFloat> code_diff;
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out,
        obj_diff, in_diff;
    CuMatrix<BaseFloat> ref_feats, ref_feats_transf, ref_in, ref_out;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_ref = 0, num_other_error = 0,
        num_cache = 0, num_set = 0;
    for (; !set2utt_reader.Done(); set2utt_reader.Next()) {
      std::string setkey = set2utt_reader.Key();
      if (!code_vec_reader.HasKey(setkey)) {
        KALDI_ERR<< "No code for set " << setkey;
      }
      KALDI_LOG<< "Set # " << ++num_set << " - " << setkey << ":";
      // update the nnet's codeat layers with new code
      for (int32 c = 0; c < num_codeat; ++c) {
        layers_codeat[c]->SetCode(code_vec_reader.Value(setkey));
        // also clean the code_corr so that the gradient doesn't accumulate
        // through different sets
        layers_codeat[c]->ZeroCodeCorr();
      }

      // all the utterances belong to this set
      std::vector<std::string> uttlst(set2utt_reader.Value());

      for (int32 uid = 0; uid < uttlst.size();) {

        // fill the cache
        while (!cache.Full() && uid < uttlst.size()) {
          std::string utt = uttlst[uid];
          KALDI_VLOG(2) << "Reading utt " << utt;
          // check that we have alignments
          std::string feat_key = feature_reader.Key();
          std::string ref_feat_key = ref_feature_reader.Key();
          if (feat_key!=utt || ref_feat_key!=utt) {
            num_no_ref++;
            uid++;
            continue;
          }
          // get feature alignment pair
          const Matrix<BaseFloat> &mat = feature_reader.Value();
          const Matrix<BaseFloat> &ref_mat = ref_feature_reader.Value();
          // check maximum length of utterance
          if (mat.NumRows() > max_frames) {
            KALDI_WARN<< "Utterance " << utt << ": Skipped because it has " << mat.NumRows() <<
            " frames, which is more than " << max_frames << ".";
            num_other_error++;
            uid++;
            continue;
          }
            // check length match of features/alignments
          if (ref_mat.NumRows()!= mat.NumRows() || ref_mat.NumCols() != mat.NumCols()) {
            KALDI_WARN<< "Reference feature has wrong size [" << ref_mat.NumRows() << "," << ref_mat.NumCols() 
            << "] vs. expected ["<< mat.NumRows() << "," << mat.NumCols() << "], for utt " << utt;
            num_other_error++;
            uid++;
            continue;
          }

            // All the checks OK,
            // push features to GPU
          if(feats.NumRows()!=mat.NumRows() || feats.NumCols()!=mat.NumCols()) {
            feats.Resize(mat.NumRows(), mat.NumCols());
          }
          feats.CopyFromMat(mat);
          if(ref_feats.NumRows()!=ref_mat.NumRows() || ref_feats.NumCols()!=ref_mat.NumCols()) {
            ref_feats.Resize(ref_mat.NumRows(), ref_mat.NumCols());
          }
          ref_feats.CopyFromMat(ref_mat);
          // possibly apply transform
          nnet_transf.Feedforward(feats, &feats_transf);
          nnet_transf.Feedforward(ref_feats, &ref_feats_transf);
          // add to cache
          cache.AddData(feats_transf, ref_feats_transf);
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
          cache.GetBunch(&nnet_in, &ref_in);

          // generate reference hidden activations
          nnet_ref.Feedforward(ref_in, &ref_out);

          // train
          nnet.Propagate(nnet_in, &nnet_out);

          mse.Eval(nnet_out, ref_out, &obj_diff);

          if (!crossvalidate) {
            // we need the nnet work to propagate throught the first layer
            nnet.Backpropagate(obj_diff, &in_diff);

            // accumulate code corr through different layers
            code_diff=layers_codeat[0]->GetCodeDiff();
            for (int32 c = 1; c < num_codeat; ++c) {
              code_diff.AddMat(1.0, layers_codeat[c]->GetCodeDiff(), 1.0);
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

      if (!crossvalidate && update_code_vec) {
        (layers_codeat[0]->GetCode()).CopyToVec(&code);
        code_vec_writer.Write(setkey, code);
        KALDI_LOG << "Write code vector for " << setkey;
      }

    }  // end for set2utt

    if(!crossvalidate && (update_weight || update_code_xform)) {
      nnet.Write(out_adapt_filename, binary);
      KALDI_LOG << "Write weight/xform to " << out_adapt_filename;
    }

    KALDI_LOG<< (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED "
    << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG << "Done " << num_set << " sets.";
    KALDI_LOG<< "Done " << num_done << " files, " << num_no_ref
    << " with no alignments, " << num_other_error
    << " with other errors.";

    KALDI_LOG<< mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

