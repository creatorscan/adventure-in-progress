/*
 * codemat-train.cc
 *
 *  Created on: Nov 8, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      Learning a code matrix and per-frame code for the input layer in a per-utterance 
 *      manner. The original nnet's input layer is not to be updated. 
 *
 *      The feature and code archieve should have the same utterance ordering.
 */

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform one iteration of code training by stochastic gradient descent.\n"
        "Usage:  codemat-train [options] <model-front> <model-code> <model-back>"
        " <feature-rspecifier> <code-rspecifier> <alignments-rspecifier>\n"
        "e.g.: \n"
        " codemat-train front.nnet code.nnet back.nnet scp:train.scp scp:code.scp ark:train.ali\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    bool update_code_xform = false,
      update_code_vec = false;
    po.Register("update-code-xform", &update_code_xform, "Update the code transformation");
    po.Register("update-code-vec", &update_code_vec, "Update the code vectors for each frame");

    std::string out_codennet_filename = "", 
      code_wspecifier = "";
    po.Register("out-codennet-filename", &out_codennet_filename, "Output code transfromation nnet");
    po.Register("code-wspecifier", &code_wspecifier, "Output code vector archieve");

    BaseFloat code_learn_rate = 0.0001;
    po.Register("code-learn-rate", &code_learn_rate, "Learning rate for the code vectors");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id=0;
    po.Register("use-gpu-id", &use_gpu_id, "Unused, kaldi is compiled w/o CUDA");
#endif
    
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    if(!crossvalidate && !update_code_xform && !update_code_vec) {
      KALDI_ERR << "Nothing to learn!";
    }

    if(!crossvalidate && update_code_xform && out_codennet_filename=="") {
      KALDI_ERR << "Output code transfromation nnet not specified!";
    }

    if(!crossvalidate && update_code_vec && code_wspecifier=="") {
      KALDI_ERR << "Output code vector archieve not specified!";
    }

    std::string nnet_front_filename = po.GetArg(1),
      nnet_code_filename = po.GetArg(2),
      nnet_back_filename = po.GetArg(3),
      feature_rspecifier = po.GetArg(4),
      code_rspecifier = po.GetArg(5),
      alignments_rspecifier = po.GetArg(6);
     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    // only forward is needed for nnet_transf and nnet_front
    Nnet nnet_transf, nnet_front;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    nnet_front.Read(nnet_front_filename);

    // dummy training options that disable the weight update
    NnetTrainOptions dummy_trn_opts;
    dummy_trn_opts.learn_rate=0.0;
    // backpropagation is needed for nnet_back and nnet_code
    Nnet nnet_back;
    nnet_back.Read(nnet_back_filename);
    // disable the update of this back end nnet
    nnet_back.SetTrainOptions(dummy_trn_opts);

    Nnet nnet_code;
    nnet_code.Read(nnet_code_filename);
    if(update_code_xform) {
      nnet_code.SetTrainOptions(trn_opts);
    } else {
      nnet_code.SetTrainOptions(dummy_trn_opts);
    }
    

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader code_reader(code_rspecifier);
    RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);

    BaseFloatMatrixWriter code_writer(code_wspecifier);

    Xent xent;

    Matrix<BaseFloat> codes_host;
    
    CuMatrix<BaseFloat> feats, feats_transf, front_out, nnet_out, obj_diff, back_diff, code_diff;
    CuMatrix<BaseFloat> codes, code_out;

    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_mismatched_code = 0, num_other_error = 0;
    while(!feature_reader.Done() && !code_reader.Done()) {
      std::string utt = feature_reader.Key();
      std::string code_key = code_reader.Key();

      if (!alignments_reader.HasKey(utt)) {
        num_no_alignment++;
      } else if (utt!=code_key){
        num_mismatched_code++;
      } else
      {
        
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Matrix<BaseFloat> &mat_code = code_reader.Value();
        const std::vector<int32> &alignment = alignments_reader.Value(utt);
         
        if ((int32)alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignment has wrong size "<< (alignment.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        if (mat.NumRows() != mat_code.NumRows()) {
          KALDI_WARN << "Feature and code has mismatched frame numbers: " << mat.NumRows() << " vs. " << mat_code.NumRows();
          num_other_error++;
          feature_reader.Next();
          code_reader.Next();
          continue;
        }

        // log
        KALDI_VLOG(2) << "utt " << utt << ", frames " << alignment.size();

        // push features to GPU
        if(feats.NumRows()!=mat.NumRows() || feats.NumCols()!=mat.NumCols()) {
          feats.Resize(mat.NumRows(), mat.NumCols());
        }
        feats.CopyFromMat(mat);
        if(codes.NumRows()!=mat_code.NumRows() || codes.NumCols()!=mat_code.NumCols()) {
          codes.Resize(mat_code.NumRows(), mat_code.NumCols());
        }
        codes.CopyFromMat(mat_code);

        // pre-transformation
        nnet_transf.Feedforward(feats, &feats_transf);

        // front nnet, simple forward, no updates needed
        nnet_front.Feedforward(feats_transf, &front_out);

        // code nnet
        nnet_code.Propagate(codes, &code_out);

        // add-in the shifting
        front_out.AddMat(1.0, code_out, 1.0);

        // propagate throught the back nnet
        nnet_back.Propagate(front_out, &nnet_out);
        
        xent.EvalVec(nnet_out, alignment, &obj_diff);
        
        if (!crossvalidate) {
          nnet_back.Backpropagate(obj_diff, &back_diff);
          nnet_code.Backpropagate(back_diff, &code_diff);

          // update the code 
          if(update_code_vec) {
            codes.AddMat(-1 * code_learn_rate, code_diff, 1.0);
            codes_host.Resize(codes.NumRows(), codes.NumCols());
            codes.CopyToMat(&codes_host);
            code_writer.Write(code_key, codes_host);
          }
        }

        total_frames += mat.NumRows();
      }
    
      num_done++;
      Timer t_features;
      feature_reader.Next();
      code_reader.Next();
      time_next += t_features.Elapsed();

      //report the speed
      if (num_done % 1000 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second.";
      }
    }

    if (!crossvalidate && update_code_xform) {
      nnet_code.Write(out_codennet_filename, binary);
    }
    
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_mismatched_code
              << " with mismatched codes, " << num_other_error
              << " with other errors.";

    KALDI_LOG << xent.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
