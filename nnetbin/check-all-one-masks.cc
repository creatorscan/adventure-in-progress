// nnetbin/check-all-one-masks.cc

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-cache-tgtmat.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Check whether the given input masks are all one.\n"
            "Usage:  check-all-one-masks [options] <mask-rspecifier>\n"
            "e.g.: \n"
            " check-all-one-masks scp:train.scp\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    CuMatrix<BaseFloat> feats;

    Timer tim;
    double time_next = 0;

    int32 num_done = 0, num_error = 0;

    while (!feature_reader.Done()) {
      // get the keys
      std::string fea_key = feature_reader.Key();
      // get feature tgt_mat pair
      const Matrix<BaseFloat> &fea_mat = feature_reader.Value();

      if( fea_mat.Sum() != fea_mat.NumRows() * fea_mat.NumCols() ){
        KALDI_LOG << fea_key << " have non-1 elements.";
        num_error++;
      }

      num_done++;

      feature_reader.Next();
    }

    std::cout << "\n" << std::flush;

    KALDI_LOG<< "COMPUTATION" << " FINISHED "
    << tim.Elapsed() << "s, fps" << tot_t/tim.Elapsed()
    << ", feature wait " << time_next << "s";

    KALDI_LOG<< "Done " << num_done << " files, " << num_error
    << " masks with none 1 elements.";

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
