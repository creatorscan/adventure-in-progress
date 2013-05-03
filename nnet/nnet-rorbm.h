/*
 * nnet/nnet-rorbm.h
 *
 *  Created on: Apr 30, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef KALDI_NNET_RORBM_H_
#define KALDI_NNET_RORBM_H_


#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "nnet/nnet-rbm.h"
#include "cudamatrix/cu-rand.h"

namespace kaldi{
/*
 * Robust RBM
 *
 * The clean GRBM model parameters are pre-trained separately as a conventional
 * GRBM and simply copied here.
 *
 */
class RoRbm : public RbmBase {
 public:
  /*
   * dim_in: the input dimension of the noisy signal;
   * dim_out: the hidden dimension of the clean speech RBM;
   */
  RoRbm(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : Rbm(dim_in, dim_out, nnet)
  {
  }

  ~RoRbm() {
  }

  ComponentType GetType() const {
    return kRoRbm;
  }

  void ReadData(std::istream &is, bool binary) {
    //TODO::

  }

  void WriteData(std::ostream &os, bool binary) const {
    //TODO::

  }

  // UpdatableComponent API
  void PropagateFnc(const CuMatrix<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void BackpropagateFnc(const CuMatrix<BaseFloat> &in,
                        CuMatrix<BaseFloat> *out) {
    KALDI_ERR<< "Cannot backpropagate through RBM!"
    << "Better convert it to <BiasedLinearity>";
  }

  // RBM training API
  void Reconstruct(const CuMatrix<BaseFloat> &hid_state, CuMatrix<BaseFloat> *vis_probs) {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void RbmUpdate(const CuMatrix<BaseFloat> &pos_vis, const CuMatrix<BaseFloat> &pos_hid, const CuMatrix<BaseFloat> &neg_vis, const CuMatrix<BaseFloat> &neg_hid) {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  /*
   * Visible unit type for the clean speech RBM.
   */
  RbmNodeType VisType() const {
    return vis_type_;
  }

  /*
   * Hidden unit type for the clean speech RBM.
   */
  RbmNodeType HidType() const {
    return hid_type_;
  }

  void WriteAsNnet(std::ostream& os, bool binary) const {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  void WriteAsAutoEncoder(std::ostream& os, bool isEncoder, bool binary) const {
    KALDI_ERR << "Not implemented for RoRbm!";
  }

  /*
   * To do posterior inference in a single object RoRbm conditioned on a vt_cn image
   */
  void Infer(const CuMatrix<BaseFloat> &vt_cn, CuMatrix<BaseFloat> *v,
             CuMatrix<BaseFloat> *ha,
             CuMatrix<BaseFloat> *s,
             CuMatrix<BaseFloat> *hs, CuMatrix<BaseFloat> *v_condmean,
             CuMatrix<BaseFloat> *z,
             int32 nIters, int32 start_z, BaseFloat z_momentum);

  void Learn(const CuMatrix<BaseFloat> &batchdata, int32 posPhaseInters, int32 nGibbsIters);

private:
  // Model parameters for the noisy input
  CuVector<BaseFloat> bt_; ///< Input bias vector \tilde{b}
  CuVector<BaseFloat> lamt2_; ///< Input variance vector 1.0/{\tilde{\sigma}^2}

  // Model parameters for the gating function
  CuVector<BaseFloat> gamma2_; ///< Gating vector gamma_square

  // Model parameters for the clean GRBM
  CuMatrix<BaseFloat> clean_vis_hid_;        ///< Matrix with neuron weights, size [clean_hid_dim_, vis_dim_]
  CuVector<BaseFloat> clean_vis_bias_;///< Vector with biases
  CuVector<BaseFloat> clean_hid_bias_;///< Vector with biases
  CuVector<BaseFloat> clean_vis_inv_sigma2_; ///< Inverse standard deviation of the clean GRBM inputs, 1.0/{\sigma}^2
  CuVector<BaseFloat> clean_vis_sigma_; ///< Standard deviation of the clean GRM inputs, \sigma
  CuVector<BaseFloat> clean_vis_sigma2_;

  // Model parameters for the noise indicator RBM
  CuMatrix<BaseFloat> U_; // noise_vis_hid_; ///< Weight matrix U, size [noise_hid_dim_, vis_dim_]
  CuVector<BaseFloat> d_; // noise_vis_bias_; ///< Visible bias vector d
  CuVector<BaseFloat> e_; // noise_hid_bias_; ///< Hidden bias vector e
  CuVector<BaseFloat> ee_; // noise_hid_bias_ori_; ///< Hidden bias vector ee

  // Variables for parameter updates
  CuMatrix<BaseFloat> U_corr_; ///< Matrix for noise RBM weight updates
  CuVector<BaseFloat> d_corr_; ///< Vector for noise visible bias updates
  CuVector<BaseFloat> ee_corr_; ///< Vector for noise hidden bias updates
  CuVector<BaseFloat> bt_corr_; ///< Vector for input bias updates
  CuVector<BaseFloat> lamt2_corr_; ///< Vector for input variance updates
  CuVector<BaseFloat> gamma2_corr_; ///< Vector for gamm2 updates


  CuMatrix<BaseFloat> U_pos_, U_neg_;
  CuVector<BaseFloat> d_pos_, d_neg_;
  CuVector<BaseFloat> ee_pos_, ee_neg_;
  CuVector<BaseFloat> bt_pos_, bt_neg_;  // visible node bias
  CuVector<BaseFloat> lamt2_pos_, lamt2_neg_;
  CuVector<BaseFloat> gamma2_pos_, gamma2_neg_;

  // fantasy particles (needed for negative phase of SAP)
  CuMatrix<BaseFloat> fp_vt_;
  CuMatrix<BaseFloat> ha_, haprob_, fp_ha_; /// for clean RBM hidden activations
  CuMatrix<BaseFloat> hs_, hsprob_, fp_hs_; /// for noise RBM hidden activations
  CuMatrix<BaseFloat> fp_v; /// for clean RBM input
  CuMatrix<BaseFloat> fp_s; /// for noise RBM input


  RbmNodeType vis_type_;
  RbmNodeType hid_type_;



  int32 vis_dim_; ///< visible layer dim, same for \tilde{v}, v and s
  int32 clean_hid_dim_; ///< hidden layer dim for clean GRBM
  int32 noise_hid_dim_; ///< hidden layer dim for noise indicator RBM

  CuRand<BaseFloat> cu_rand_;

  BaseFloat weight_cost_;

  int32 batch_size_;

  int32 num_gibbs_iters_; ///< number of gibbs iterations to perform
  int32 num_pos_iters_;

};

}  // namespace


#endif /* KALDI_NNET_RORBM_H_ */
