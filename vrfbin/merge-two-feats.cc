/*
 * vrfbin/merge-two-feats.cc
 *
 *  Created on: Dec 17, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Merge two features frame by frame.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Merge two features frame by frame.\n"
            "Usage: merge-two-feats [options] feats1-rspecifier feats2-rspecifier "
            "out-feats-wspecifier\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats1_rspecifier = po.GetArg(1);
    std::string feats2_rspecifier = po.GetArg(2);
    std::string out_feats_wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feats1_reader(feats1_rspecifier);
    // to save the computational cost, we assume the two features are in the same key order
    //RandomAccessBaseFloatMatrixReader feats2_reader(feats2_rspecifier);
    SequentialBaseFloatMatrixReader feats2_reader(feats2_rspecifier);
    BaseFloatMatrixWriter out_feats_writer(out_feats_wspecifier);

    int32 num_done = 0;

    for (; !feats1_reader.Done() && !feats2_reader.Done(); feats1_reader.Next(), feats2_reader.Next()) {
      std::string key = feats1_reader.Key();
      Matrix<BaseFloat> feats1(feats1_reader.Value());

      if (feats2_reader.Key() != key) {
        KALDI_ERR<< "Key mismtach for the two features, "
        << "make sure they have the same ordering.";
      }

      Matrix<BaseFloat> feats2(feats2_reader.Value());

      KALDI_ASSERT(feats1.NumRows()==feats2.NumRows());

      Matrix<BaseFloat> out_feats(feats1.NumRows(), feats1.NumCols()+feats2.NumCols());
      SubMatrix<BaseFloat> left(out_feats, 0, feats1.NumRows(), 0, feats1.NumCols());
      left.CopyFromMat(feats1, kNoTrans);
      SubMatrix<BaseFloat> right(out_feats, 0, feats1.NumRows(), feats1.NumCols(), feats2.NumCols());
      right.CopyFromMat(feats2, kNoTrans);

      out_feats_writer.Write(key, out_feats);

      ++num_done;
      if (num_done % 1000 == 0) {
        KALDI_LOG<< "Done " << num_done << " utterances.";
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

