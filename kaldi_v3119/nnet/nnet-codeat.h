/*
 * nnet-codebl.h
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      The code vector is placed after the features.
 *      In this new implementation, we store the code related transform separately.
 *      code_vec_ is shared among different layers, hence cannot directly updated in each batch;
 *      code_xform_ is specific to each layer, thus can be updated similarly to the linearity.
 */

#ifndef NNET_CODEBL_H_
#define NNET_CODEBL_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-various.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-affine-transform.h"

namespace kaldi {
namespace nnet1 {

class CodeAT : public AffineTransform {
 public:
  CodeAT(int32 dim_in, int32 dim_out)
      : AffineTransform(dim_in, dim_out),
        update_weight_(false), 
        update_code_xform_(false),
        update_code_vec_(false),
        code_dim_(0), 
        code_vec_(0), 
        code_vec_corr_(0),
        code_xform_(0,0),
        code_xform_corr_(0,0)
  {
  }

  ~CodeAT()
  {
  }

  Component* Copy() const {
    return new CodeAT(*this);
  }
  ComponentType GetType() const {
    return kCodeAT;
  }

  /*
   * The read and write functions are overwrite due to the code
   * vector.
   */
  void ReadData(std::istream &is, bool binary) {
    ReadBasicType(is, binary, &code_dim_);
    KALDI_ASSERT(code_dim_ >= 0);

    AffineTransform::ReadData(is, binary);

    code_xform_.Read(is, binary);

    KALDI_ASSERT(code_xform_.NumRows() == output_dim_);
    KALDI_ASSERT(code_xform_.NumCols() == code_dim_);

    // resize the necessary data structure
    code_vec_.Resize(code_dim_);
    code_vec_corr_.Resize(code_dim_);

    code_xform_corr_.Resize(output_dim_, code_dim_);

  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, code_dim_);
    if (!binary)
      os << "\n";

    AffineTransform::WriteData(os, binary);

    code_xform_.Write(os, binary);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    AffineTransform::PropagateFnc(in, out);

    // add the code shifting if needed
    if (code_dim_ > 0) {
      if(in.NumRows()!=code_vec_expd_.NumRows()) {
        code_vec_expd_.Resize(in.NumRows(), code_dim_);
        code_vec_expd_.AddVecToRows(1.0, code_vec_, 0.0);
      }
      out->AddMatMat(1.0, code_vec_expd_, kNoTrans, code_xform_, kTrans, 1.0);
    }
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in,
                        const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff,
                        CuMatrix<BaseFloat> *in_diff) {
    AffineTransform::BackpropagateFnc(in, out, out_diff, in_diff);

    // compute the code diff
    code_vec_diff_.Resize(out_diff.NumRows(), code_dim_);
    code_vec_diff_.AddMatMat(1.0, out_diff, kNoTrans, code_xform_, kNoTrans, 0.0);
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &diff) {
    if(update_weight_){
      AffineTransform::Update(input, diff);
    }

    if(update_code_xform_) {
      // we use following hyperparameters from the option class
      const BaseFloat lr = opts_.learn_rate;
      const BaseFloat mmt = opts_.momentum;
      const BaseFloat l2 = opts_.l2_penalty;
      const BaseFloat l1 = opts_.l1_penalty;
      // we will also need the number of frames in the mini-batch
      const int32 num_frames = input.NumRows();
      // compute gradient (incl. momentum)
      code_xform_corr_.AddMatMat(1.0, diff, kTrans, code_vec_expd_, kNoTrans, mmt);
      
      // l2 regularization
      if (l2 != 0.0) {
        code_xform_.AddMat(-lr * l2 * num_frames, code_xform_);
      }
      // l1 regularization
      if (l1 != 0.0) {
        cu::RegularizeL1(&code_xform_, &code_xform_corr_, lr * l1 * num_frames, lr);
      }
      // update
      code_xform_.AddMat(-lr, code_xform_corr_);
    }

    // the code vector is update in the main tool due to its inter-layer sharing
    // the code_vec_diff_ is computed in the BackpropagateFnc.
  }

  // The diff will be averaged across layers beforing accumulated into corr_
  const CuMatrix<BaseFloat>& GetCodeDiff() {
    return code_vec_diff_;
  }

  void UpdateCode(const CuMatrix<BaseFloat>& diff) {
    if(update_code_vec_) {
      code_vec_corr_.AddRowSumMat(1.0, diff, opts_.momentum);
      code_vec_.AddVec(-1 * opts_.learn_rate, code_vec_corr_);
    }
  }

  void SetCode(const CuVector<BaseFloat> &code) {
    code_vec_.Resize(code.Dim());
    code_vec_.CopyFromVec(code);
  }

  void SetCode(const Vector<BaseFloat> &code) {
    code_vec_.Resize(code.Dim());
    code_vec_.CopyFromVec(code);
  }

  const CuVector<BaseFloat>& GetCode() {
    return code_vec_;
  }

  void ZeroCode() {
    code_vec_.SetZero();
  }

  void ZeroCodeCorr() {
    code_vec_corr_.SetZero();
  }

  int32 GetCodeDim() {
    return code_dim_;
  }

  void ConfigureUpdate(bool weight, bool code_xform, bool code_vec) {
    update_weight_ = weight;
    update_code_xform_ = code_xform;
    update_code_vec_ = code_vec;
  }

 protected:
  bool update_weight_;
  bool update_code_xform_;
  bool update_code_vec_;

  int32 code_dim_;  // dimensionality of the code vector

  CuVector<BaseFloat> code_vec_;  // code vector
  CuVector<BaseFloat> code_vec_corr_;  // correction of the code vector

  CuMatrix<BaseFloat> code_vec_expd_; // expanded code vector
  CuMatrix<BaseFloat> code_vec_diff_;  // intermediate for code_

  CuMatrix<BaseFloat> code_xform_; // code transformation matrix
  CuMatrix<BaseFloat> code_xform_corr_; // correction of the code transformation matrix
};
}
}

#endif /* NNET_CODEBL_H_ */
