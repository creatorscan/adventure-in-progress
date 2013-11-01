/*
 * nnet-codebl.h
 *
 *  Created on: Oct 31, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *      The code vector is placed after the features.
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
        code_dim_(0)
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
    code_vec_.Resize(code_dim_);
    code_corr_.Resize(code_dim_);
    code_diff_.Resize(code_dim_);

    linearity_.Read(is, binary);
    bias_.Read(is, binary);

    KALDI_ASSERT(linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(linearity_.NumCols() == input_dim_ + code_dim_);
    KALDI_ASSERT(bias_.Dim() == output_dim_);

    linearity_corr_.Resize(linearity_.NumRows(), linearity_.NumCols());
    bias_corr_.Resize(bias_.Dim());
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteBasicType(os, binary, code_dim_);
    if (!binary)
      os << "\n";
    linearity_.Write(os, binary);
    bias_.Write(os, binary);
  }

  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    // augment the input if needed
    if (code_dim_ > 0) {
      if (aug_in_.NumRows() != in.NumRows()
          || aug_in_.NumCols() != in.NumCols() + code_dim_) {
        aug_in_.Resize(in.NumRows(), in.NumCols() + code_dim_);
      }
      (CuSubMatrix<BaseFloat>(aug_in_, 0, aug_in_.NumRows(), 0,
                              aug_in_.NumCols())).CopyFromMat(in);
      (CuSubMatrix<BaseFloat>(aug_in_, 0, aug_in_.NumRows(), aug_in_.NumCols(),
                              code_dim_)).AddVecToRows(1.0, code_vec_, 0.0);
    } else {
      aug_in_ = in;
    }

    // precopy bias
    out->AddVecToRows(1.0, bias_, 0.0);
    // multiply by weights^t
    out->AddMatMat(1.0, aug_in_, kNoTrans, linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in,
                        const CuMatrix<BaseFloat> &out,
                        const CuMatrix<BaseFloat> &out_diff,
                        CuMatrix<BaseFloat> *in_diff) {
    // compute the augmented errors
    if (aug_err_.NumRows() != out_diff.NumRows()
        || aug_err_.NumCols() != linearity_.NumCols()) {
      aug_err_.Resize(out_diff.NumRows(), linearity_.NumCols());
    }
    aug_err_.AddMatMat(1.0, out_diff, kNoTrans, linearity_, kNoTrans, 0.0);

    if (code_dim_ > 0) {
      code_diff_.AddRowSumMat(
          1.0,
          (CuSubMatrix<BaseFloat>(aug_err_, 0, aug_err_.NumRows(),
                                  aug_err_.NumCols() - code_dim_,
                                  code_dim_)),
          0.0);
      in_diff->CopyFromMat(
          CuSubMatrix<BaseFloat>(aug_err_, 0, aug_err_.NumRows(), 0,
                                 aug_err_.NumCols() - code_dim_));
    } else {
      in_diff->CopyFromMat(aug_err_);
    }
  }

  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &diff) {
    if (!update_weight_)
      return;

    // we use following hyperparameters from the option class
    const BaseFloat lr = opts_.learn_rate;
    const BaseFloat mmt = opts_.momentum;
    const BaseFloat l2 = opts_.l2_penalty;
    const BaseFloat l1 = opts_.l1_penalty;
    // we will also need the number of frames in the mini-batch
    const int32 num_frames = input.NumRows();
    // compute gradient (incl. momentum)
    linearity_corr_.AddMatMat(1.0, diff, kTrans, aug_in_, kNoTrans, mmt);
    bias_corr_.AddRowSumMat(1.0, diff, mmt);

    // l2 regularization
    if (l2 != 0.0) {
      linearity_.AddMat(-lr * l2 * num_frames, linearity_);
    }
    // l1 regularization
    if (l1 != 0.0) {
      cu::RegularizeL1(&linearity_, &linearity_corr_, lr * l1 * num_frames, lr);
    }
    // update
    linearity_.AddMat(-lr, linearity_corr_);
    bias_.AddVec(-lr, bias_corr_);

    // the code vector is update in the main tool due to its cluster based
    // update, here only the correction for this batch is computed.
  }

  const CuVector<BaseFloat>& GetCodeDiff() {
    return code_diff_;
  }

  void UpdateCode(const CuVector<BaseFloat>& diff) {
    code_corr_.AddVec(1.0, diff, opts_.momentum);
    code_vec_.AddVec(-1 * opts_.learn_rate, code_corr_);
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
    code_corr_.SetZero();
  }

  int32 GetCodeDim() {
    return code_dim_;
  }

  void EnableWeightUpdate() {
    update_weight_ = true;
  }

  void DisableWeightUpdate() {
    update_weight_ = false;
  }

 protected:
  bool update_weight_;

  int32 code_dim_;  // dimensionality of the code vector

  CuVector<BaseFloat> code_vec_;  // code vector
  CuVector<BaseFloat> code_corr_;  // correction of the code vector
  CuVector<BaseFloat> code_diff_;  // intermediate for code_

  CuMatrix<BaseFloat> aug_in_;  // input features augmented with code vec
  CuMatrix<BaseFloat> aug_err_;  // the error includes the code err
};
}
}

#endif /* NNET_CODEBL_H_ */
