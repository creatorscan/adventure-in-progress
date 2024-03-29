/*
 * ideal-hidmask-forward.cc
 *
 *  Created on: Oct 9, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      This program is just for concept verification. We use parallel data to compute the ideal mask
 *      and then applied to the noisy hidden activations.
 */

#include <limits> 

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;

  try {
    const char *usage =
        "Perform forward pass through Neural Network with ideal hidden masking.\n"
        "Usage:  ideal-hidmask-forward [options] <l1-model-in> "
        "<feature-rspecifier> <ref-feat-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " ideal-hidmask-forward --backend-nnet=backend.nnet l1.nnet ark:features.ark "
        "ark:ref_feats.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

	bool binarize_mask = false;
	po.Register("binarize-mask", &binarize_mask, "Binarize the hidden mask");

	BaseFloat binarize_threshold = 0.5;
	po.Register("binarize-threshold", &binarize_threshold, "Threshold to binarize the hidden mask");

    BaseFloat alpha = 1.0;
    po.Register("alpha", &alpha, "Alpha value for the hidden mask compuation");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");

    std::string backend_nnet = "";
    po.Register("backend-nnet", &backend_nnet, "Backend Nnet");

    std::string class_frame_counts;
    po.Register("class-frame-counts", &class_frame_counts, "Counts of frames for posterior division by class-priors");

    BaseFloat prior_scale = 1.0;
    po.Register("prior-scale", &prior_scale, "scaling factor of prior log-probabilites given by --class-frame-counts");

    bool apply_log = false, silent = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output. The MLP outputs directly log-likelihoods, log-priors will be subtracted");

    po.Register("silent", &silent, "Don't print any messages");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string l1_model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        ref_feats_rspecifier = po.GetArg(3),
        feature_wspecifier = po.GetArg(4);
        
    using namespace kaldi;
    typedef kaldi::int32 int32;

    Nnet nnet_transf, nnet_backend;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    if(backend_nnet != ""){
      nnet_backend.Read(backend_nnet);
    }

    Nnet nnet;
    nnet.Read(l1_model_filename);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader ref_feats_reader(ref_feats_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, l1_out, nnet_out, hidmask;
    CuMatrix<BaseFloat> ref_feats, ref_feats_transf, ref_l1_out;
    Matrix<BaseFloat> nnet_out_host;

    // Read the class-counts, compute priors
    Vector<BaseFloat> tmp_priors;
    CuVector<BaseFloat> priors;
    if(class_frame_counts != "") {
      Input in;
      in.OpenTextMode(class_frame_counts);
      tmp_priors.Read(in.Stream(), false);
      in.Close();
      
      BaseFloat sum = tmp_priors.Sum();
      tmp_priors.Scale(1.0/sum);
      if (apply_log || no_softmax) {
        tmp_priors.ApplyLog();
        tmp_priors.Scale(-prior_scale);
      } else {
        tmp_priors.ApplyPow(-prior_scale);
      }

      // push priors to GPU
      priors=tmp_priors;
    }

    Timer tim;
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD STARTED";

    int32 num_done = 0;
    // iterate over all the feature files
    for (; !feature_reader.Done() && !ref_feats_reader.Done(); feature_reader.Next(), ref_feats_reader.Next()) {
      // read
      std::string key = feature_reader.Key();
      std::string ref_key = ref_feats_reader.Key();
      if(key != ref_key){
        KALDI_ERR << "Mismatched keys: " << key << " vs. " << ref_key;
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      const Matrix<BaseFloat> &ref_mat = ref_feats_reader.Value();
      if(mat.NumRows()!=ref_mat.NumRows() || mat.NumCols() != ref_mat.NumCols()){
        KALDI_ERR << "Feature dimension mismatch for " << key ;
      }
      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          BaseFloat ref_val = ref_mat(r,c);
          if(val != val || ref_val != ref_val) KALDI_ERR << "NaN in features of : " << key;
          if(val == std::numeric_limits<BaseFloat>::infinity() || ref_val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << key;
        }
      }
      // push it to gpu
      feats=mat;
      ref_feats=ref_mat;
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet_transf.Feedforward(ref_feats, &ref_feats_transf);

      nnet.Feedforward(feats_transf, &l1_out);
      nnet.Feedforward(ref_feats_transf, &ref_l1_out);

      /*
       * Do masking
       *
       */

      hidmask=l1_out;
      hidmask.AddMat(-1.0, ref_l1_out, 1.0);
      hidmask.ApplyPow(2.0);
      hidmask.Scale(-1.0*alpha);
      hidmask.ApplyExp();
      if(binarize_mask) hidmask.Binarize(binarize_threshold);

      l1_out.MulElements(hidmask);

      // forward through the backend dnn
      if(backend_nnet != ""){
        nnet_backend.Feedforward(l1_out, &nnet_out);
      }else{
        nnet_out=l1_out;
      }
      
      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // divide posteriors by priors to get quasi-likelihoods
      if(class_frame_counts != "") {
        if (apply_log || no_softmax) {
          nnet_out.AddVecToRows(1.0, priors, 1.0);
        } else {
          nnet_out.MulColsVec(priors);
        }
      }
     
      //download from GPU 
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);
      //check for NaN/inf
      for(int32 r=0; r<nnet_out_host.NumRows(); r++) {
        for(int32 c=0; c<nnet_out_host.NumCols(); c++) {
          BaseFloat val = nnet_out_host(r,c);
          if(val != val) KALDI_ERR << "NaN in NNet output of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in NNet coutput of : " << feature_reader.Key();
          /*if(val == kBaseLogZero)
            nnet_out_host(r,c) = -1e10;*/
        }
      }
      // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 1000 == 0) {
        if(!silent) KALDI_LOG << num_done << ", " << std::flush;
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    if(!silent) KALDI_LOG << "MLP FEEDFORWARD FINISHED " 
                          << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed(); 
    if(!silent) KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (!silent) CuDevice::Instantiate().PrintProfile();
#endif

    return ((num_done>0)?0:1);
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}

