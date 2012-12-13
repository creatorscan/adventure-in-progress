/*
 * vrfbin/segment-feats.cc
 *
 *  Created on: Dec 13, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * Generate segment based features for SVM based verification.
 * The output format is binary for SVMLight, each line corresponds
 * to a sample of the ascii form:
 *  label 1:v1 2:v2 ... # [comments, not compulsory]
 * or binary form:
 *  label v1 v2 ...
 *
 * The segmentation is based on the input frame label sequences, i.e.
 * adjacent frames with the same label are grouped together as one segment.
 *
 * The features are averaged across the segment.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#define MAX_PHONEME_LENGTH 10

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Segment features for SVMLight based verification.\n"
            "Usage: segment-feats [options] in-feats-rspecifier in-labels-rspecifier out-feats-wfilename\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary,
                "Write the output feature file in binary format");

    bool normalize = false;
    po.Register("normalize", &normalize, "Whether to normalize the feature to sum to 1");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feats_rspecifier = po.GetArg(1);
    std::string labels_rspecifier = po.GetArg(2);
    std::string out_wfilename = po.GetArg(3);

    FILE *fout;
    if(binary){
      fout=fopen(out_wfilename.c_str(), "wb");
    }else{
      fout=fopen(out_wfilename.c_str(), "w");
    }

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    RandomAccessTokenVectorReader labels_reader(labels_rspecifier);

    int32 num_done=0;
    bool first=true;
    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string key = feats_reader.Key();
      Matrix<BaseFloat> feats(feats_reader.Value());

      if(first){ // write out the dimension of the feature vectors
        if(binary){
          int dim=feats.NumCols();
          fwrite(&dim, 1, sizeof(dim), fout);
        }else{
          fprintf(fout, "%d\n", feats.NumCols());
        }
        first=false;
      }

      if (!labels_reader.HasKey(key)) {
        KALDI_WARN << "No labels available for key "
                   << key << ", producing no output for this utterance";
        continue;
      }

      std::vector<std::string> labels(labels_reader.Value(key));

      Vector<BaseFloat> segment(feats.NumCols(), kSetZero);
      std::string pre_label="";
      int32 seglen=0;
      for(int32 i=0; i<feats.NumRows(); ++i){
        std::string cur_label=labels[i];
        if (cur_label == pre_label){
          segment.AddVec(1.0, feats.Row(i));
          seglen+=1;
        }else {
          if(seglen > 0){
            if (normalize) { // do simple normalization
              BaseFloat sum=segment.Sum();
              segment.Scale(1.0/sum);
            }else{ // do averaging
              segment.Scale(1.0/seglen);
            }
            // write out the previous segment
            if(binary){
              // write the label
              fwrite(pre_label.c_str(), MAX_PHONEME_LENGTH, sizeof(char), fout);
              // write the values
              for (int32 j=0; j<feats.NumCols(); ++j){
                float val=segment(j); // ensure use float point numbers
                fwrite(&val, 1, sizeof(val), fout);
              }
            }else{
              fprintf(fout, "%s", pre_label.c_str());
              for(int32 j=0; j<feats.NumCols(); ++j){
                fprintf(fout, " %d:%f", j+1, segment(j));
              }
              fprintf(fout, "\n");
            }
          }
          // start new segment
          pre_label=cur_label;
          segment.CopyFromVec(feats.Row(i));
          seglen=1;
        }
      }

      ++num_done;
      if(num_done%100){
        KALDI_LOG << "Done " << num_done << "utterances.";
      }
    }

    fclose(fout);

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

