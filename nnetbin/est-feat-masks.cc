// nnetbin/est-feat-masks.cc
//
// Estimate the masks for each utterance based on the NN posteriors and
// prior mask patterns.
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Estimate masks for each utterance.\n"
            "Usage:  est-feat-masks [options] <pat-wxfilename> <post-rspecifier> <mask-wspecifier>\n"
            "e.g.: \n"
            " est-feat-masks mask_patterns scp:post.scp ark:mask.ark\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string pat_wxfilename = po.GetArg(1),
        post_rspecifier = po.GetArg(2),
        mask_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader post_reader(post_rspecifier);
    BaseFloatMatrixWriter mask_writer(mask_wspecifier);

    Timer tim;

    int32 num_done = 0;
    Matrix<BaseFloat> patterns;
    {
      bool binary;
      Input ki(pat_wxfilename, &binary);
      patterns.Read(ki.Stream(), binary);
    }


    for ( ; !post_reader.Done(); post_reader.Next()) {
      // get the keys
      std::string utt = post_reader.Key();
      const Matrix<BaseFloat> &post = post_reader.Value();

      Matrix<BaseFloat> mask(post.NumRows(), patterns.NumCols());
      mask.AddMatMat(1.0, post, kNoTrans, patterns, kNoTrans, 0.0);

      mask_writer.Write(utt, mask);

      num_done++;
      if(num_done % 100 == 0){
        KALDI_LOG << "Done " << num_done << " files.";
      }

    }

    std::cout << "\n" << std::flush;
    KALDI_LOG<< "COMPUTATION" << " FINISHED " << tim.Elapsed() << "s";
    KALDI_LOG<< "Done " << num_done << " files.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
