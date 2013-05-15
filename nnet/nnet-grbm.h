/*
 * nnet/nnet-grbm.h
 *
 *  Created on: May 9, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef KALDI_NNET_GRBM_H_
#define KALDI_NNET_GRBM_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-rbm.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi {

/*
 * Gaussian visible units and binary hidden units.
 * Learn input variance.
 *
 */
class GRbm : public RbmBase {
 public:
  GRbm(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : RbmBase(dim_in, dim_out, nnet),
        std_learn_rate_(0.001),
        apply_sparsity_(true),
        sparsity_lambda_(0.01),
        sparsity_p_(0.2)
  {
  }
  ~GRbm()
  {
  }

  ComponentType GetType() const {
    return kGRbm;
  }

  void ReadData(std::istream &is, bool binary) {
    std::string vis_node_type, hid_node_type;
    ReadToken(is, binary, &vis_node_type);
    ReadToken(is, binary, &hid_node_type);

    KALDI_ASSERT(vis_node_type == "gauss");
    KALDI_ASSERT(hid_node_type == "bern");

    vis_hid_.Read(is, binary);
    vis_bias_.Read(is, binary);
    hid_bias_.Read(is, binary);
    fstd_.Read(is, binary);

    // compute variance
    fvar_.CopyFromVec(fstd_);
    fvar_.Power(2.0);

    KALDI_ASSERT(vis_hid_.NumRows() == output_dim_);
    KALDI_ASSERT(vis_hid_.NumCols() == input_dim_);
    KALDI_ASSERT(vis_bias_.Dim() == input_dim_);
    KALDI_ASSERT(hid_bias_.Dim() == output_dim_);
    KALDI_ASSERT(fstd_.Dim() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {

    WriteToken(os, binary, "gauss");
    WriteToken(os, binary, "bern");

    vis_hid_.Write(os, binary);
    vis_bias_.Write(os, binary);
    hid_bias_.Write(os, binary);
    fstd_.Write(os, binary);
  }

  /*
   * UpdatableComponent API
   *
   * Generate hidden probabilities given the input visible states.
   */
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    //KALDI_LOG << "In PropagateFnc...";
    // precopy bias
    out->AddVecToRows(1.0, hid_bias_, 0.0);
    // copy input to data
    data_.CopyFromMat(in);
    // divide std
    data_.DivColsVec(fstd_);
    // multiply by weights^t
    out->AddMatMat(1.0, data_, kNoTrans, vis_hid_, kTrans, 1.0);
    // apply sigmoid
    cu::Sigmoid(*out, out);
  }

  /*
   * RBM training API
   *
   * Generate vis probabilities from the hidden states.
   */
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state,
                   CuMatrix<BaseFloat> *vis_probs) {
    //KALDI_LOG << "In Reconstruct ... ";
    // check the dim
    if (output_dim_ != hid_state.NumCols()) {
      KALDI_ERR<< "Nonmatching dims, component:" << output_dim_ << " data:" << hid_state.NumCols();
    }
    // optionally allocate buffer
    if (input_dim_ != vis_probs->NumCols() || hid_state.NumRows() != vis_probs->NumRows()) {
      vis_probs->Resize(hid_state.NumRows(), input_dim_);
    }

    // multiply by weights
    vis_probs->AddMatMat(1.0, hid_state, kNoTrans, vis_hid_, kNoTrans, 0.0);
    // multiply std
    vis_probs->MulColsVec(fstd_);
    // add bias
    vis_probs->AddVecToRows(1.0, vis_bias_, 1.0);
  }

  /*
   * Add Gaussian noise to convert the input visible probability to visible samples.
   */
  void SampleVisible(CuRand<BaseFloat> &rand, CuMatrix<BaseFloat> *vis_probs) {
    if(data_.NumRows()!=vis_probs->NumRows() || data_.NumCols() != vis_probs->NumCols()) {
      data_.Resize(vis_probs->NumRows(), vis_probs->NumCols());
      data_.SetZero();
    }
    // generate standard Gaussian random samples
    rand.RandGaussian(&data_);
    // scale to the desired standard deviation
    data_.MulColsVec(fstd_);
    // shift to the desired mean
    vis_probs->AddMat(1.0, data_, 1.0);
  }

  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid,
      const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid,
      CuVector<BaseFloat> *avg_hid_probs, bool first_bunch) {
    //KALDI_LOG << "In RbmUpdate ...";

    assert(pos_vis.NumRows() == pos_hid.NumRows() &&
        pos_vis.NumRows() == neg_vis.NumRows() &&
        pos_vis.NumRows() == neg_hid.NumRows() &&
        pos_vis.NumCols() == neg_vis.NumCols() &&
        pos_hid.NumCols() == neg_hid.NumCols() &&
        pos_vis.NumCols() == input_dim_ &&
        pos_hid.NumCols() == output_dim_);

    //lazy initialization of buffers and will set to zero for the first time
    if(first_bunch) {
      vis_hid_corr_.Resize(output_dim_,input_dim_);
      vis_bias_corr_.Resize(input_dim_);
      hid_bias_corr_.Resize(output_dim_);
      log_fstd_corr_.Resize(input_dim_);

      vis_hid_grad_.Resize(output_dim_, input_dim_);
      hid_bias_grad_.Resize(output_dim_);
      log_fstd_grad_.Resize(input_dim_);
    }

    //  UPDATE vishid matrix
    //
    //  vishidinc = momentum*vishidinc + ...
    //              epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    //
    //  vishidinc[t] = -(epsilonw/numcases)*negprods + momentum*vishidinc[t-1]
    //                 +(epsilonw/numcases)*posprods
    //                 -(epsilonw*weightcost)*vishid[t-1]
    //
    BaseFloat N = static_cast<BaseFloat>(pos_vis.NumRows());
    data_.CopyFromMat(neg_vis);// neg_vis
    data_.MulColsVec(fstd_);// neg_vis ./ fstd
    vis_hid_corr_.AddMatMat(-learn_rate_/N, neg_hid, kTrans, data_, kNoTrans, momentum_);
    data_.CopyFromMat(pos_vis);// pos_vis
    data_.MulColsVec(fstd_);// pos_vis ./ fstd
    vis_hid_corr_.AddMatMat(+learn_rate_/N, pos_hid, kTrans, data_, kNoTrans, 1.0);
    vis_hid_corr_.AddMat(-learn_rate_*l2_penalty_, vis_hid_, 1.0);

    //  UPDATE visbias vector
    //
    //  visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    //
    tmp_vec_vis_.Resize(pos_vis.NumCols());
    tmp_vec_vis_.AddRowSumMat(-1.0, neg_vis, 0.0);// -neg_vis
    tmp_vec_vis_.AddRowSumMat(1.0, pos_vis, 1.0);// pos_vis - neg_vis
    tmp_vec_vis_.DivElements(fvar_);// (pos_vis / (fstd.^2)) - (neg_vis / (fstd.^2))
    vis_bias_corr_.AddVec(learn_rate_/N, tmp_vec_vis_, momentum_);

    //  UPDATE hidbias vector
    //
    // hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    //
    hid_bias_corr_.AddRowSumMat(-learn_rate_/N, neg_hid, momentum_);
    hid_bias_corr_.AddRowSumMat(+learn_rate_/N, pos_hid, 1.0);

    // UPDATE visible variance
    //
    //
    //
    tmp_mat_n_vis_.CopyFromMat(pos_vis);// data
    tmp_mat_n_vis_.AddVecToRows(-1.0, vis_bias_, 1.0);// data - vb
    tmp_mat_n_vis_.Power(2.0);// (data - vb).^2
    tmp_mat_n_vis_.DivColsVec(fvar_);// (data - vb).^2 ./ fvar
    log_fstd_grad_.AddRowSumMat(1.0, tmp_mat_n_vis_, 0.0);
    tmp_mat_n_vis_.AddMatMat(1.0, pos_hid, kNoTrans, vis_hid_, kNoTrans, 0.0);// (vhw * pos_hidprobs')
    tmp_mat_n_vis_.MulElements(pos_vis);// data .* (vhw * pos_hidprobs')
    tmp_mat_n_vis_.DivColsVec(fstd_);// data .* (vhw * pos_hidprobs') ./ fstd
    log_fstd_grad_.AddRowSumMat(-1.0, tmp_mat_n_vis_, 1.0);
    tmp_mat_n_vis_.CopyFromMat(neg_vis);// negdata
    tmp_mat_n_vis_.AddVecToRows(-1.0, vis_bias_, 1.0);// negdata - vb
    tmp_mat_n_vis_.Power(2.0);// (negdata - vb).^2
    tmp_mat_n_vis_.DivColsVec(fvar_);// (negdata -vb).^2 ./ fvar_
    log_fstd_grad_.AddRowSumMat(-1.0, tmp_mat_n_vis_, 1.0);
    tmp_mat_n_vis_.AddMatMat(1.0, neg_hid, kNoTrans, vis_hid_, kNoTrans, 0.0);// (vhw * neg_hidprobs')
    tmp_mat_n_vis_.MulElements(neg_vis);// negdata .* (vhw * neg_hidprobs')
    tmp_mat_n_vis_.DivColsVec(fstd_);// negdata .* (vhw * neg_hidprobs') ./ fstd
    log_fstd_grad_.AddRowSumMat(1.0, tmp_mat_n_vis_, 1.0);
    // correction
    log_fstd_corr_.AddVec(-std_learn_rate_/N, log_fstd_grad_, momentum_);
    // constrain the updates
    log_fstd_corr_.ApplyTruncate(-1.0, 1.0);

    if ( apply_sparsity_) {
      if(first_bunch) {
        avg_hid_probs->Resize(output_dim_);
        avg_hid_probs->AddRowSumMat(1.0/N, pos_hid, 0.0);  // q = mean(pos_hidprobs)
      } else {
        avg_hid_probs->AddRowSumMat(0.1/N, pos_hid, 0.9);  // q = 0.9 * q + 0.1 * mean(pos_hidprobs)
      }

      /* prepare temporal variables */
      tmp_mat_n_hid_.Resize(pos_hid.NumRows(), output_dim_);
      tmp_mat_n_hid_.Set(1.0);
      tmp_mat_n_hid_.AddMat(-1.0, pos_hid, 1.0);  // 1 - pos_hidprobs
      tmp_mat_n_hid_.MulElements(pos_hid);// pos_hidprobs .* (1 - pos_hidprobs)

      tmp_vec_hid_.Resize(output_dim_);
      tmp_vec_hid_.Set(1.0);
      tmp_vec_hid_.AddVec(-1.0, *avg_hid_probs, 1.0);// 1 - q
      tmp_vec_hid_.MulElements(*avg_hid_probs);// q .* (1 - q)

      tmp_vec_hid_2_.Resize(output_dim_);
      tmp_vec_hid_2_.Set(sparsity_p_);
      tmp_vec_hid_2_.AddVec(-1.0, *avg_hid_probs, 1.0);// p - q
      tmp_vec_hid_2_.DivElements(tmp_vec_hid_);// (p - q) ./ (q .* (1 - q))

      /* compute hidden bias gradient */
      hid_bias_grad_.AddRowSumMat(0.1 * sparsity_lambda_ / N, tmp_mat_n_hid_, 0.0);  // 0.1 * sparse_lambda / n * sum(pos_hidprobs .* (1-pos_hidprobs))
      hid_bias_grad_.MulElements(tmp_vec_hid_2_);// 0.1 * sparse_lambda / n * sum(pos_hidprobs .* (1-pos_hidprobs)) .* (p - q) ./ (q .* (1 - q))

      /* compute weight gradient */
      vis_hid_grad_.AddMatMat(0.1 * sparsity_lambda_ / N, tmp_mat_n_hid_, kTrans, data_, kNoTrans, 0.0);  // 0.1 * sparse_lambda / n * (data' ./ Fstd' * (pos_hidprobs .* (1-pos_hidprobs)))
      vis_hid_grad_.MulRowsVec(tmp_vec_hid_2_);

      /* update the correction */
      hid_bias_corr_.AddVec(learn_rate_, hid_bias_grad_, 1.0);
      vis_hid_corr_.AddMat(learn_rate_, vis_hid_grad_, 1.0);
    }

    // do the update
    vis_hid_.AddMat(1.0, vis_hid_corr_, 1.0);
    vis_bias_.AddVec(1.0, vis_bias_corr_, 1.0);
    hid_bias_.AddVec(1.0, hid_bias_corr_, 1.0);

    fvar_.CopyFromVec(log_fstd_corr_);// log_fstd
    fvar_.ApplyExp();// exp(log_fstd)
    fstd_.MulElements(fvar_);
    fstd_.ApplyFloor(0.1);

    // update variance
    fvar_.CopyFromVec(fstd_);
    fvar_.Power(2.0);
  }

  RbmNodeType VisType() const {
    return vis_type_;
  }

  RbmNodeType HidType() const {
    return hid_type_;
  }

  void SetVarianceLearnRate(BaseFloat value) {
    std_learn_rate_ = value;
  }

  void EnableSparsity() {
    apply_sparsity_ = true;
  }

  void DisableSparsity() {
    apply_sparsity_ = false;
  }

  void ConfigSparsity(BaseFloat lambda, BaseFloat p) {
    sparsity_lambda_ = lambda;
    sparsity_p_ = p;
  }

  void WriteAsNnet(std::ostream& os, bool binary) const {
    //header
    WriteToken(os,binary,Component::TypeToMarker(Component::kBiasedLinearity));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,InputDim());
    if(!binary) os << "\n";
    //data
    /* apply the variance to the weight matrix */
    CuMatrix<BaseFloat> mat(output_dim_, input_dim_);
    mat.CopyFromMat(vis_hid_);
    mat.DivColsVec(fstd_);

    mat.Write(os,binary);
    hid_bias_.Write(os,binary);

    // sigmoid activation
    WriteToken(os,binary,Component::TypeToMarker(Component::kSigmoid));
    WriteBasicType(os,binary,OutputDim());
    WriteBasicType(os,binary,OutputDim());

    if(!binary) os << "\n";
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ERR<< "Not implemented for GRbm!";
  }
  virtual void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &err) {
    KALDI_ERR << "Not implemented for GRbm!";
  }

  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {
    KALDI_ERR << "Not implemented for GRbm!";
  }

  void WriteAsAutoEncoder(std::ostream& os, bool isEncoder, bool binary) const {
    KALDI_ERR << "Not implemented for GRbm!";
  }
protected:
  CuMatrix<BaseFloat> vis_hid_;        ///< Matrix with neuron weights
  CuVector<BaseFloat> vis_bias_;///< Vector with biases
  CuVector<BaseFloat> hid_bias_;///< Vector with biases
  CuVector<BaseFloat> fstd_;///< visible unit standard deviation
  CuVector<BaseFloat> fvar_;///< visible unit variance = fstd_.^2

  CuMatrix<BaseFloat> vis_hid_corr_;///< Matrix for linearity updates
  CuVector<BaseFloat> vis_bias_corr_;///< Vector for bias updates
  CuVector<BaseFloat> hid_bias_corr_;///< Vector for bias updates
  CuVector<BaseFloat> log_fstd_corr_;///< Vector for input std updates, use log to ensure std to be positive

  RbmNodeType vis_type_;
  RbmNodeType hid_type_;

  CuMatrix<BaseFloat> data_;///< input data bunch
  CuMatrix<BaseFloat> tmp_mat_n_vis_;///< temporary data matrix of the size [N, vis_dim]
  CuMatrix<BaseFloat> tmp_mat_n_hid_;///< temporary data matrix of the size [N, hid_dim]
  CuVector<BaseFloat> tmp_vec_vis_;///< temporary visible vector of the size [1, vis_dim]
  CuVector<BaseFloat> tmp_vec_hid_;///< temporary hidden vector of the size [1, hid_dim]
  CuVector<BaseFloat> tmp_vec_hid_2_;///< temporary hidden vector of the size [1, hid_dim]

  CuMatrix<BaseFloat> vis_hid_grad_;///< Weight matrix gradient for sparsity
  CuVector<BaseFloat> hid_bias_grad_;///< Hidden bias vector gradient for sparsity
  CuVector<BaseFloat> log_fstd_grad_;///< Vector for input std gradient

  BaseFloat std_learn_rate_;///< learning rate for standard deviation

  bool apply_sparsity_;///< imply whether to apply sparsity to the weights
  BaseFloat sparsity_lambda_;
  BaseFloat sparsity_p_;
};

}  // end namespace

#endif /* KALDI_NNET_GRBM_H_ */
