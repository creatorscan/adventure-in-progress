/*
 *
 *  Troy Lee (troy.lee2008@gmail.com)
 *  Nov. 12, 2013
 *
 *  LIN Affine Transformation forward.
 */

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "nnet/nnet-linat.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through Neural Network with <linat> layers.\n"
        "\n"
        "Usage:  linat-forward [options] <model-in> <lin-weight-rspecifier> <lin-bias-rspecifier> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " linat-forward nnet ark:weight.ark ark:bias.ark ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

#if HAVE_CUDA==1
    int32 use_gpu_id=-2;
    po.Register("use-gpu-id", &use_gpu_id, "Manually select GPU by its ID (-2 automatic selection, -1 disable GPU, 0..N select GPU)");
#else
    int32 use_gpu_id=0;
    po.Register("use-gpu-id", &use_gpu_id, "Unused, kaldi is compiled w/o CUDA");
#endif

    std::string utt2xform; 
    po.Register("utt2xform", &utt2xform, "Utterance to LIN xform mapping");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        weight_rspecifier = po.GetArg(2),
        bias_rspecifier = po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        feature_wspecifier = po.GetArg(5);
        
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu_id);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    //optionally remove softmax
    if(no_softmax && nnet.GetComponent(nnet.NumComponents()-1).GetType() == Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveComponent(nnet.NumComponents()-1);
    }
    //check for some non-sense option combinations
    if(apply_log && no_softmax) {
      KALDI_ERR << "Nonsense option combination : --apply-log=true and --no-softmax=true";
    }
    if(apply_log && nnet.GetComponent(nnet.NumComponents()-1).GetType() != Component::kSoftmax) {
      KALDI_ERR << "Used --apply-log=true, but nnet " << model_filename 
                << " does not have <softmax> as last component!";
    }
    
    PdfPrior pdf_prior(prior_opts);
    if (prior_opts.class_frame_counts != "" && (!no_softmax && !apply_log)) {
      KALDI_ERR << "Option --class-frame-counts has to be used together with "
                << "--no-softmax or --apply-log";
    }

    if((nnet.GetComponent(0)).GetType() != Component::kLinAT) {
      KALDI_ERR << "The first layer is not <linat> layer!";
    }
    LinAT &lin = static_cast<LinAT&>(nnet.GetComponent(0));

    CuMatrix<BaseFloat> weight(lin.OutputDim(), lin.InputDim());
    CuVector<BaseFloat> bias(lin.OutputDim());


    kaldi::int64 tot_t = 0;

    RandomAccessTokenReader utt2xform_reader(utt2xform);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    RandomAccessBaseFloatMatrixReader weight_reader(weight_rspecifier);
    RandomAccessBaseFloatVectorReader bias_reader(bias_rspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;


    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    std::string cur_lin = "", new_lin = "";
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      std::string key = feature_reader.Key();

      if(utt2xform==""){
        new_lin = key;
      } else {
        if(!utt2xform_reader.HasKey(key)) {
          KALDI_ERR << "No mapping found for utterance " << key;
        }
        new_lin = utt2xform_reader.Value(key);
      }

      if(!weight_reader.HasKey(new_lin) || !bias_reader.HasKey(new_lin)) {
        KALDI_ERR << "No LIN weight/bias for the utterance " << key;
      }

      // update the LIN xform when necessary 
      if(new_lin != cur_lin) {
        const Matrix<BaseFloat> &weight_host = weight_reader.Value(new_lin);
        const Vector<BaseFloat> &bias_host = bias_reader.Value(new_lin);

        weight.CopyFromMat(weight_host);
        bias.CopyFromVec(bias_host);

        lin.SetLinearity(weight);
        lin.SetBias(bias);

        cur_lin = new_lin;
      }

      const Matrix<BaseFloat> &mat = feature_reader.Value();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << feature_reader.Key() 
                    << ", " << mat.NumRows() << "frm";

      //check for NaN/inf
      for(int32 r=0; r<mat.NumRows(); r++) {
        for(int32 c=0; c<mat.NumCols(); c++) {
          BaseFloat val = mat(r,c);
          if(val != val) KALDI_ERR << "NaN in features of : " << feature_reader.Key();
          if(val == std::numeric_limits<BaseFloat>::infinity())
            KALDI_ERR << "inf in features of : " << feature_reader.Key();
        }
      }

      // push it to gpu
      feats = mat;
      // fwd-pass
      nnet_transf.Feedforward(feats, &feats_transf);
      nnet.Feedforward(feats_transf, &nnet_out);
      
      // convert posteriors to log-posteriors
      if (apply_log) {
        nnet_out.ApplyLog();
      }
     
      // subtract log-priors from log-posteriors to get quasi-likelihoods
      if(prior_opts.class_frame_counts != "" && (no_softmax || apply_log)) {
        pdf_prior.SubtractOnLogpost(&nnet_out);
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
        }
      }

      // write
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return -1;
  }
}
