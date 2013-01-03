/*
 * vrfbin/merge-three-feats.cc
 *
 *  Created on: Dec 17, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Merge three features frame by frame.
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
        "Merge three features frame by frame.\n"
            "Usage: merge-three-feats [options] feats1-rspecifier feats2-rspecifier "
            "feats3-rspecifier out-feats-wspecifier\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats1_rspecifier = po.GetArg(1);
    std::string feats2_rspecifier = po.GetArg(2);
    std::string feats3_rspecifier = po.GetArg(3);
    std::string out_feats_wspecifier = po.GetArg(4);

    SequentialBaseFloatMatrixReader feats1_reader(feats1_rspecifier);
    // To save computation cost we assume the three features are in the same key order
    //RandomAccessBaseFloatMatrixReader feats2_reader(feats2_rspecifier);
    //RandomAccessBaseFloatMatrixReader feats3_reader(feats3_rspecifier);
    SequentialBaseFloatMatrixReader feats2_reader(feats2_rspecifier);
    SequentialBaseFloatMatrixReader feats3_reader(feats3_rspecifier);
    BaseFloatMatrixWriter out_feats_writer(out_feats_wspecifier);

    int32 num_done = 0;

    for (; !feats1_reader.Done() && !feats2_reader.Done() && !feats3_reader.Done(); feats1_reader.Next(), feats2_reader.Next(), feats3_reader.Next()) {
      std::string key = feats1_reader.Key();
      Matrix<BaseFloat> feats1(feats1_reader.Value());

      if ((feats2_reader.Key() != key) || (feats3_reader.Key() != key) ) {
        KALDI_ERR<< "Feature order mismatch!";
      }

      Matrix<BaseFloat> feats2(feats2_reader.Value());
      Matrix<BaseFloat> feats3(feats3_reader.Value());

      KALDI_ASSERT(feats1.NumRows()==feats2.NumRows() && feats1.NumRows()==feats3.NumRows());

      Matrix<BaseFloat> out_feats(feats1.NumRows(),
                                  feats1.NumCols() + feats2.NumCols() + feats3.NumCols());

      SubMatrix<BaseFloat> left(out_feats, 0, feats1.NumRows(), 0,
                                feats1.NumCols());
      left.CopyFromMat(feats1, kNoTrans);

      SubMatrix<BaseFloat> mid(out_feats, 0, feats1.NumRows(), feats1.NumCols(), feats2.NumCols());
      mid.CopyFromMat(feats2, kNoTrans);

      SubMatrix<BaseFloat> right(out_feats, 0, feats1.NumRows(),
                                 feats1.NumCols()+feats2.NumCols(), feats3.NumCols());
      right.CopyFromMat(feats3, kNoTrans);

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

