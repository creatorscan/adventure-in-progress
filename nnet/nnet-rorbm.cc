/*
 * nnet/nnet-robm.cc
 *
 *  Created on: Apr 30, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "nnet/nnet-rorbm.h"

namespace kaldi {

void RoRbm::Infer(const CuMatrix<BaseFloat> &vt_cn, CuMatrix<BaseFloat> *v,
                  CuMatrix<BaseFloat> *ha,
                  CuMatrix<BaseFloat> *s,
                  CuMatrix<BaseFloat> *hs,
                  CuMatrix<BaseFloat> *v_condmean,
                  CuMatrix<BaseFloat> *z,
                  int32 nIters, int32 start_z, BaseFloat z_momentum) {

  int32 n = vt_cn.NumRows();  // number of instances in current bunch
  KALDI_ASSERT(vt_cn.NumCols()==vis_dim_);
  CuMatrix<BaseFloat> vt_cn_0(vt_cn.NumRows(), vt_cn.NumCols());
  vt_cn_0.CopyFromMat(vt_cn);  // save for future use

  CuMatrix<BaseFloat> mu(n, vis_dim_);
  CuMatrix<BaseFloat> phi_s(n, vis_dim_);
  CuMatrix<BaseFloat> mu_hat(n, vis_dim_);
  CuMatrix<BaseFloat> log_sprob_1(n, vis_dim_);
  CuMatrix<BaseFloat> log_sprob_0(n, vis_dim_);
  CuMatrix<BaseFloat> mat_tmp(n, vis_dim_);
  CuMatrix<BaseFloat> v_condstd(n, vis_dim_);

  CuVector<BaseFloat> std_hat(vis_dim_);
  CuVector<BaseFloat> inv_gamma2_tmp(gamma2_.Dim());
  CuVector<BaseFloat> vec_tmp(vis_dim_);

  for (int32 k = 0; k < nIters; ++k) {
    // downsample - from hidden to visible
    /* needed for sprob_0, clean GRBM */
    mu.AddMatMat(1.0, *ha, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // ha * W
    mu.MulColsVec(clean_vis_sigma2_);  // sigma2 * (ha * W)
    mu.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // b + sigma2 * (ha * W)
    /* needed for sprob_1, noise RBM */
    phi_s.AddVecToRows(1.0, noise_vis_bias_, 0.0);  // d
    phi_s.AddMatMat(1.0, *hs, kNoTrans, noise_vis_hid_, kNoTrans, 1.0);  // d + hs * U

    /* needed for sprob_1, noisy input */
    mu_hat.CopyFromMat(vt_cn);
    mu_hat.MulColsVec(gamma2_);  // gamma2 .* vt_cn
    mu_hat.AddMat(1.0, mu, 1.0);  // mu + gamma2 .* vt_cn
    inv_gamma2_tmp.CopyFromVec(gamma2_);  // gamma2
    inv_gamma2_tmp.Add(1.0);  // gamma2 + 1
    inv_gamma2_tmp.InvertElements();  // 1.0 / (gamma2 + 1)
    mu_hat.MulColsVec(inv_gamma2_tmp);  // (mu + gamma2 .* vt_cn) ./ (gamma2 + 1)
    /* needed for sprob_1 */
    inv_gamma2_tmp.Power(0.5);  // 1.0 ./ sqrt(gamma2 + 1)
    inv_gamma2_tmp.MulElements(clean_vis_sigma_);  // std_vec ./ sqrt(gamma2 + 1)
    std_hat.CopyFromVec(inv_gamma2_tmp);

    /* compute log_sprob_1 */
    log_sprob_1.CopyFromMat(phi_s);  // phi_s

    mat_tmp.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp.Power(2.0);  // vt_cn.^2
    vec_tmp.CopyFromVec(clean_vis_inv_sigma2_);  // 1.0 ./ var_vec
    vec_tmp.MulElements(gamma2_);  // gamma2 ./ var_vec
    mat_tmp.MulColsVec(vec_tmp);  // vt_cn.^2 .* gamma2 ./ var_vec
    log_sprob_1.AddMat(-0.5, mat_tmp, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec

    mat_tmp.CopyFromMat(mu_hat);  // mu_hat
    vec_tmp.CopyFromVec(std_hat);  // std_hat
    vec_tmp.InvertElements();  // 1.0 ./std_hat
    mat_tmp.MulColsVec(vec_tmp);  // mu_hat ./ std_hat
    mat_tmp.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1.AddMat(0.5, mat_tmp, 1.0);  // phi_s - 0.5 * vt_cn.^2 .* gamma2 ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp.CopyFromVec(std_hat);  // std_hat
    vec_tmp.ApplyLog();  // log(std_hat)
    log_sprob_1.AddVecToRows(1.0, vec_tmp, 1.0);

    /* compute log_sprob_0 */
    log_sprob_0.SetZero();

    mat_tmp.CopyFromMat(mu);  // mu
    mat_tmp.Power(2.0);  // mu.^2
    mat_tmp.MulColsVec(clean_vis_inv_sigma2_);  // mu.^2 ./ var_vec
    log_sprob_0.AddMat(0.5, mat_tmp, 0.0);  // mu.^2 ./ var_vec

    vec_tmp.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp.ApplyLog();  // log(std_vec)
    log_sprob_0.AddVecToRows(1.0, vec_tmp, 1.0);  // mu.^2 ./ var_vec + log(std_vec)

    /* log(exp(log_sprob_0) + exp(log_sprob_1)) */
    log_sprob_0.LogAddExpMat(log_sprob_1);

    /* compute sprob (saved in log_sprob_1) */
    log_sprob_1.AddMat(-1.0, log_sprob_0);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /* compute s */
    cu_rand_.BinarizeProbs(log_sprob_1, s);

    /* compute v_condmean */
    v_condmean->CopyFromMat(mu);  // mu

    mat_tmp.CopyFromMat(vt_cn);  // vt_cn
    mat_tmp.MulElements(*s);  // s .* vt_cn
    mat_tmp.MulColsVec(gamma2_);  // gamma2 .* s.* vt_cn
    v_condmean->AddMat(1.0, mat_tmp, 1.0);  // gamma2 .* s.* vt_cn + mu

    mat_tmp.CopyFromMat(*s);  // s
    mat_tmp.MulColsVec(gamma2_);  // gamma2 .* s
    mat_tmp.Add(1.0);  // gamma2 .* s + 1
    v_condmean->DivElements(mat_tmp);  // (gamma2 .* s.* vt_cn + mu) ./ (gamma2 .* s + 1)

    /* compute v_condstd */
    v_condstd.AddVecToRows(1.0, clean_vis_sigma_, 0.0);  // std_vec
    mat_tmp.Power(0.5);  // sqrt(gamma2 .* s + 1)
    v_condstd.DivElements(mat_tmp);  // std_vec ./ sqrt(gamma2 .* s + 1)

    /* sample from v */
    cu_rand_.RandGaussian(v);
    v->MulElements(v_condstd);
    v->AddMat(1.0, *v_condmean, 1.0);

    /* normalise the masked vt_cn */
    // NOT implemented yet, as may not be compulsory
    /* sample the hidden variables */
    mat_tmp.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
    mat_tmp.AddMatMat(1.0, *v, kNoTrans, clean_vis_hid_, kTrans, 1.0);  // v*W + c
    cu::Sigmoid(mat_tmp, &mat_tmp);  // 1.0 ./ (1.0 + exp(v*W + c))
    cu_rand_.BinarizeProbs(mat_tmp, ha);  // binarize

    mat_tmp.AddVecToRows(1.0, noise_hid_bias_, 0.0);  // e
    mat_tmp.AddMatMat(1.0, *s, kNoTrans, noise_vis_hid_, kTrans, 1.0);  // s*U + e
    cu::Sigmoid(mat_tmp, &mat_tmp);  // 1.0 ./ (1.0 + exp(s*U + e))
    cu_rand_.BinarizeProbs(mat_tmp, hs);  // binarize

    /* collect smooth estimates */
    if (start_z >= 0) {  // negative z indicates no collection
      if (k == start_z) {
        z->CopyFromMat(*v_condmean);
      } else if (k > start_z) {
        z->AddMat(1 - z_momentum, *v_condmean, z_momentum);
      }
    }

  }  // end iteration k
}  // end Infer()

void RoRbm::Learn(const CuMatrix<BaseFloat> &batchdata, int32 posPhaseInters,
                  int32 nGibbsIters) {
  int32 n = batchdata.NumRows();
  KALDI_ASSERT(batchdata.NumCols()==vis_dim_);

  CuMatrix<BaseFloat> vt(n, vis_dim_);
  CuMatrix<BaseFloat> vt_cn(n, vis_dim_);
  CuMatrix<BaseFloat> mat_tmp(n, vis_dim_);
  CuMatrix<BaseFloat> v(n, vis_dim_);
  CuMatrix<BaseFloat> s(n, vis_dim_);
  CuMatrix<BaseFloat> v_condmean(n, vis_dim_), fp_vt_condmean(n, vis_dim_);
  CuMatrix<BaseFloat> z(n, vis_dim_);

  CuMatrix<BaseFloat> ha(n, clean_hid_dim_);
  CuMatrix<BaseFloat> haprob(n, clean_hid_dim_);

  CuMatrix<BaseFloat> hs(n, noise_hid_dim_);
  CuMatrix<BaseFloat> hsprob(n, noise_hid_dim_);

  CuVector<BaseFloat> vec_tmp(vis_dim_), vec_tmp2(vis_dim_);
  CuVector<BaseFloat> s_mu(vis_dim_);
  s_mu.Set(0.9);  // moving average of the mean of the layer s



  CuMatrix<BaseFloat> mu(n, vis_dim_);
  CuMatrix<BaseFloat> phi_s(n, vis_dim_);
  CuMatrix<BaseFloat> mu_hat(n, vis_dim_), mu_t_hat(n, vis_dim_);
  CuVector<BaseFloat> std_hat(vis_dim_);
  CuMatrix<BaseFloat> log_sprob_1(n, vis_dim_);
  CuMatrix<BaseFloat> log_sprob_0(n, vis_dim_);

  CuMatrix<BaseFloat> fp_s(n, vis_dim_);
  CuMatrix<BaseFloat> v_condstd(n, vis_dim_), fp_vt_condstd(n, vis_dim_);
  CuMatrix<BaseFloat> fp_v(n, vis_dim_);

  CuVector<BaseFloat> lamt2_hat(vis_dim_);

  CuVector<BaseFloat> bt_inc(vis_dim_);
  CuVector<BaseFloat> lamt2_inc(vis_dim_);
  CuVector<BaseFloat> gamma2_inc(vis_dim_);
  CuVector<BaseFloat> d_inc(vis_dim_);
  CuVector<BaseFloat> ee_inc(noise_hid_dim_);
  CuMatrix<BaseFloat> U_inc(noise_vis_hid_.NumRows(), noise_vis_hid_.NumCols());

  /* add noise to the training data */
  // NOT implemented
  vt.CopyFromMat(batchdata);

  /* normalize the data */
  // NOT implemented
  vt_cn.CopyFromMat(vt);

  /* noise RBM hidden bias conversion, e = ee - s_mu * U */
  noise_hid_bias_.CopyFromVec(noise_hid_bias_ori_);  // ee
  mat_tmp.CopyFromMat(noise_vis_hid_);  // U
  mat_tmp.MulColsVec(s_mu);  // s_mu * U
  noise_hid_bias_.AddRowSumMat(-1.0, mat_tmp, 1.0);  // ee - s_mu * U

  /* initialize the clean RBM hidden states */
  haprob.AddVecToRows(1.0, clean_hid_bias_, 0.0);  // c
  haprob.AddMatMat(1.0, vt_cn, kNoTrans, clean_vis_hid_, kTrans, 1.0);
  cu::Sigmoid(haprob, &haprob);
  cu_rand_.BinarizeProbs(haprob, &ha);

  /* initialize the noise RBM hidden states */
  cu_rand_.RandUniform(&hs);

  Infer(vt_cn, &v, &ha, &s, &hs, &v_condmean, &z, posPhaseInters, -1, 0.0);

  /* use more smoother version */
  v.CopyFromMat(v_condmean);

  s_mu.AddColSumMat(0.05, s, 0.95);

  /* positive phase gradient */
  mat_tmp.CopyFromMat(vt_cn);  // vt_cn
  mat_tmp.MulColsVec(lamt2_);  // vt_cn .* lamt2
  bt_pos.AddColSumMat(1.0, mat_tmp, 0.0);  // sum(vt_cn .* lamt2)

  mat_tmp.CopyFromMat(vt_cn);  // vt_cn
  mat_tmp.MulColsVec(bt_);  // vt_cn .* bt
  vt.CopyFromMat(vt_cn);  // vt_cn
  vt.Power(2.0);  // vt_cn.^2
  mat_tmp.AddMat(-0.5, vt, 1.0);  // -0.5 * vt_cn.^2 + vt_cn .* bt
  lamt2_pos.AddColSumMat(1.0, mat_tmp, 0.0);  // sum(-0.5 * vt_cn.^2 + vt_cn .* bt)

  mat_tmp.CopyFromMat(vt_cn);  // vt_cn
  mat_tmp.AddMat(1.0, v, -1.0);  // v - vt_cn
  mat_tmp.Power(2.0);  // (v - vt_cn).^2
  mat_tmp.MulElements(s);  // s .* (v - vt_cn).^2
  mat_tmp.Scale(-0.5);  // -0.5 * s.* (v - vt_cn).^2
  gamma2_pos.AddColSumMat(1.0, mat_tmp, 0.0);  // sum(-0.5 * s.* (v - vt_cn).^2)
  gamma2_pos.DivElements(clean_vis_sigma2_);  // sum(-0.5 * s.* (v - vt_cn).^2) ./ var_vec

  mat_tmp.CopyFromMat(s);  // s
  mat_tmp.AddVecToRows(-1.0, s_mu, 1.0);  // s - s_mu
  U_pos.AddMatMat(1.0, hs, kTrans, mat_tmp, kNoTrans, 0.0);
  d_pos.AddColSumMat(1.0, mat_tmp, 0.0);
  ee_pos.AddColSumMat(1.0, hs, 0.0);

  /* update using SAP */
  for (int32 kk = 0; kk < nGibbsIters; ++kk) {

    /* #1. p(s|hs, ha, vt) */
    mu.AddMatMat(1.0, fp_ha_, kNoTrans, clean_vis_hid_, kNoTrans, 0.0);  // fp_ha * W
    mu.MulColsVec(clean_vis_sigma2_);  // (fp_ha * W) .* var_vec
    mu.AddVecToRows(1.0, clean_vis_bias_, 1.0);  // (fp_ha * W) .* var_vec + b

    phi_s.AddVecToRows(1.0, noise_vis_bias_, 0.0);  // d
    phi_s.AddMatMat(1.0, fp_hs_, kNoTrans, noise_vis_hid_, kNoTrans, 1.0);  // fp_hs * U + d

    mu_hat.CopyFromMat(fp_vt_);  // fp_vt
    mu_hat.MulColsVec(gamma2_);  // fp_vt .* gamma2
    mu_hat.AddMat(1.0, mu, 1.0);  // mu + fp_vt .* gamma2
    vec_tmp.CopyFromVec(gamma2_);  // gamma2
    vec_tmp.Add(1.0);  // gamma2 + 1
    vec_tmp.InvertElements();  // 1.0 / (gamma2 + 1)
    mu_hat.MulColsVec(vec_tmp);  // (mu + fp_vt .* gamma2) ./ (gamma2 + 1)

    std_hat.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp.Power(0.5);  // 1.0 / sqrt(gamma2 + 1)
    std_hat.DivElements(vec_tmp);  // std_vec ./ sqrt(gamma2 + 1)

    /** compute log_sprob_1 **/
    log_sprob_1.CopyFromMat(phi_s);  // phi_s

    mat_tmp.CopyFromMat(fp_vt_);  // fp_vt
    mat_tmp.Power(2);  // fp_vt.^2
    mat_tmp.MulColsVec(gamma2_);  // gamma2 .* (fp_vt.^2)
    mat_tmp.MulColsVec(clean_vis_inv_sigma2_);  // gamma2 .* (fp_vt.^2) ./ var_vec
    log_sprob_1.AddMat(-0.5, mat_tmp, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec

    mat_tmp.CopyFromMat(mu_hat);  // mu_hat
    mat_tmp.DivColsVec(std_hat);  // mu_hat ./ std_hat
    mat_tmp.Power(2.0);  // mu_hat.^2 ./ std_hat.^2
    log_sprob_1.AddMat(0.5, mat_tmp, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2

    vec_tmp.CopyFromVec(std_hat);  // std_hat
    vec_tmp.ApplyLog();  // log(std_hat)
    log_sprob_1.AddVecToRows(1.0, vec_tmp, 1.0);  //  phi_s - 0.5 * gamma2 .* (fp_vt.^2) ./ var_vec + 0.5 * mu_hat.^2 ./ std_hat.^2 + log(std_hat)

    /** compute log_sprob_0 **/
    log_sprob_0.CopyFromMat(mu);  // mu
    log_sprob_0.Power(2.0);  // mu.^2
    log_sprob_0.MulColsVec(clean_vis_inv_sigma2_);  // mu.^2 ./ var_vec
    log_sprob_0.Scale(0.5);  // 0.5 * mu.^2 ./ var_vec
    vec_tmp.CopyFromVec(clean_vis_sigma_);  // std_vec
    vec_tmp.ApplyLog();  // log(std_vec)
    log_sprob_0.AddVecToRows(1.0, vec_tmp, 1.0);

    /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
    log_sprob_0.LogAddExpMat(log_sprob_1);

    /** compute sprob (saved in log_sprob_1) **/
    log_sprob_1.AddMat(-1.0, log_sprob_0);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    /** compute s **/
    cu_rand_.BinarizeProbs(log_sprob_1, &fp_s);

    /* #2. p(v|s, ha, vt) */
    mat_tmp.CopyFromMat(fp_s);  // fp_s
    mat_tmp.MulColsVec(gamma2_);  // gamma2 .* fp_s
    v_condmean.CopyFromMat(mat_tmp);  // gamma2 .* fp_s
    v_condmean.MulElements(fp_vt_);  // gamma2 .* fp_s .* fp_vt
    v_condmean.AddMat(1.0, mu, 1.0);  // gamma2 .* fp_s .* fp_vt + mu
    mat_tmp.Add(1.0);  // gamma2 .* fp_s + 1.0
    v_condmean.DivElements(mat_tmp);  // (gamma2 .* fp_s .* fp_vt + mu) ./ (gamma2 .* fp_s + 1.0)

    v_condstd.CopyFromMat(mat_tmp);  // gamma2 .* fp_s + 1.0
    v_condstd.Power(0.5);  // sqrt(gamma2 .* fp_s + 1.0)
    v_condstd.InvertElements();  // 1.0 ./ sqrt(gamma2 .* fp_s + 1.0)
    v_condstd.MulColsVec(clean_vis_sigma_);  // std_vec ./ sqrt(gamma2 .* fp_s + 1.0)

    /** sample from v **/
    cu_rand_.RandGaussian(&fp_v);  // random
    fp_v.MulElements(v_condstd);  // fp_v .* v_condstd
    fp_v.AddMat(1.0, v_condmean, 1.0);  // fp_v .* v_condstd + v_condmean

    /* #3. p(s|v, hs) */
    vec_tmp.CopyFromVec(clean_vis_sigma2_);  // var_vec
    vec_tmp.MulElements(bt_);  // var_vec .* bt
    vec_tmp2.CopyFromVec(gamma2_);  // gamma2
    vec_tmp2.DivElements(lamt2_);  // gamma2 ./ lamt2
    mu_t_hat.CopyFromMat(fp_v);  // fp_v
    mu_t_hat.MulColsVec(vec_tmp2);  // (gamma2 ./ lamt2) .* fp_v
    mu_t_hat.AddVecToRows(1.0, vec_tmp, 1.0);  // var_vec .* bt + (gamma2 ./ lamt2) .* fp_v
    vec_tmp2.AddVec(1.0, clean_vis_sigma2_, 1.0);  // var_vec + gamma2 ./ lamt2
    mu_t_hat.DivColsVec(vec_tmp2);  // (var_vec .* bt + (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + gamma2 ./ lamt2)

    lamt2_hat.CopyFromVec(vec_tmp2);  // var_vec + gamma2 ./ lamt2
    lamt2_hat.DivElements(clean_vis_sigma2_);  // (var_vec + gamma2 ./ lamt2) ./ var_vec
    lamt2_hat.MulElements(lamt2_);  // (var_vec + gamma2 ./ lamt2) ./ (var_vec ./ lamt2)

    /** compute log_sprob_1 **/
    log_sprob_1.CopyFromMat(phi_s);  // phi_s

    mat_tmp.CopyFromMat(fp_v);  // fp_v
    mat_tmp.Power(2);  // fp_v.^2
    mat_tmp.MulColsVec(gamma2_);  // gamma2 .* (fp_v.^2)
    mat_tmp.MulColsVec(clean_vis_inv_sigma2_);  // gamma2 .* (fp_v.^2) ./ var_vec
    log_sprob_1.AddMat(-0.5, mat_tmp, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec

    mat_tmp.CopyFromMat(mu_t_hat);  // mu_t_hat
    mat_tmp.Power(2.0); // mu_t_hat.^2
    mat_tmp.MulColsVec(lamt2_hat); // (mu_t_hat.^2) .* lamt2_hat
    log_sprob_1.AddMat(0.5, mat_tmp, 1.0);  // phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat

    vec_tmp.CopyFromVec(lamt2_hat);  // lamt2_hat
    vec_tmp.ApplyLog();  // log(lamt2_hat)
    log_sprob_1.AddVecToRows(-0.5, vec_tmp, 1.0);  //  phi_s - 0.5 * gamma2 .* (fp_v.^2) ./ var_vec + 0.5 * mu_t_hat.^2 .* lamt2_hat - log(sqrt(lamt2_hat))

    /** compute log_sprob_0 **/
    vec_tmp.CopyFromVec(bt_); // bt
    vec_tmp.Power(2.0); // bt.^2
    vec_tmp.MulElements(lamt2_); // bt.^2 .* lamt2
    vec_tmp2.CopyFromVec(lamt2_); // lamt2
    vec_tmp2.ApplyLog(); // log(lamt2)
    vec_tmp.AddVec(-0.5, vec_tmp2, 0.5); // 0.5 * bt.^2 .* lamt2 - log(sqrt(lmat2))
    log_sprob_0.AddVecToRows(1.0, vec_tmp, 0.0);

    /** log(exp(log_sprob_0) + exp(log_sprob_1)) **/
    log_sprob_0.LogAddExpMat(log_sprob_1);

    /** compute sprob (saved in log_sprob_1) **/
    log_sprob_1.AddMat(-1.0, log_sprob_0);  // log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1))
    log_sprob_1.ApplyExp();  // exp(log_sprob_1 - log(exp(log_sprob_0) + exp(log_sprob_1)))

    cu_rand_.BinarizeProbs(log_sprob_1, &fp_s);

    /* #4. p(vt | s, v) */
    vec_tmp.CopyFromVec(gamma2_); // gamma2
    vec_tmp.DivElements(lamt2_);// gamma2 ./ lamt2
    mat_tmp.CopyFromMat(fp_s); // fp_s
    mat_tmp.MulColsVec(vec_tmp);// fp_s .* (gamma2 ./ lamt2)
    vec_tmp.CopyFromVec(clean_vis_sigma2_); // var_vec
    vec_tmp.MulElements(bt_); // var_vec .* bt
    fp_vt_condmean.CopyFromMat(mat_tmp); // fp_s .* (gamma2 ./ lamt2)
    fp_vt_condmean.MulElements(fp_v); // fp_s .* (gamma2 ./ lamt2) .* fp_v
    fp_vt_condmean.AddVecToRows(1.0, vec_tmp, 1.0); // var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v
    mat_tmp.AddVecToRows(1.0, clean_vis_sigma2_, 1.0); // var_vec + fp_s .* (gamma2 ./ lamt2)
    fp_vt_condmean.DivElements(mat_tmp); // (var_vec .* bt + fp_s .* (gamma2 ./ lamt2) .* fp_v) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))

    vec_tmp.CopyFromVec(clean_vis_sigma2_); // var_vec
    vec_tmp.DivElements(bt_); // var_vec ./ bt
    fp_vt_condstd.AddVecToRows(1.0, vec_tmp, 0.0); // var_vec ./ bt
    fp_vt_condstd.DivElements(mat_tmp); // (var_vec ./ bt) ./ (var_vec + fp_s .* (gamma2 ./ lamt2))
    fp_vt_condstd.Power(0.5);// sqrt((var_vec ./ bt) ./ (var_vec + fp_s .* (gamma2 ./ lamt2)))

    /** sample from vt **/
    cu_rand_.RandGaussian(&fp_vt_);
    fp_vt_.MulElements(fp_vt_condstd); // fp_vt .* fp_vt_condstd
    fp_vt_.AddMat(1.0, fp_vt_condmean, 1.0); // fp_vt .* fp_vt_condstd + fp_vt_condmean

    /* #5. p(hs|s); p(ha|v) */
    haprob.AddVecToRows(1.0, clean_hid_bias_, 0.0); // c
    haprob.AddMatMat(1.0, fp_v, kNoTrans, clean_vis_hid_, kTrans, 1.0); // fp_v * W' + c
    cu::Sigmoid(haprob, &haprob);
    cu_rand_.BinarizeProbs(haprob, &fp_ha_);

    hsprob.AddVecToRows(1.0, noise_hid_bias_, 0.0); // e
    hsprob.AddMatMat(1.0, fp_s, kNoTrans, noise_vis_hid_, kTrans, 1.0); // fp_s * U' + e
    cu::Sigmoid(hsprob, &hsprob);
    cu_rand_.BinarizeProbs(hsprob, &fp_hs_);

  }  // iteration kk, end SAP

  /* save temporal results (vt_cn, v, s, fp_s) here if needed */

  /* negative phase gradients */
  mat_tmp.CopyFromMat(fp_vt_); // fp_vt
  mat_tmp.MulColsVec(lamt2_); // fp_vt .* lamt2
  bt_neg.AddColSumMat(1.0, mat_tmp, 0.0);

  mat_tmp.AddVecToRows(1.0, bt_, 0.0); // bt
  mat_tmp.AddMat(-0.5, fp_vt_, 1.0); // -0.5 * fp_vt + bt
  mat_tmp.MulElements(fp_vt_); // -0.5 * fp_vt.^2 + fp_vt .* bt
  lamt2_neg.AddColSumMat(1.0, mat_tmp, 0.0);

  mat_tmp.CopyFromMat(fp_v); // fp_v
  mat_tmp.AddMat(-1.0, fp_vt_, 1.0); // fp_v - fp_vt
  mat_tmp.Power(2.0); // (fp_v - fp_vt).^2
  mat_tmp.MulElements(fp_s); // fp_s .* (fp_v - fp_vt).^2
  mat_tmp.MulColsVec(clean_vis_inv_sigma2_); // fp_s .* (fp_v - fp_vt).^2 ./ var_vec
  gamma2_neg.AddColSumMat(-0.5, mat_tmp, 0.0); // -0.5 * fp_s .* (fp_v - fp_vt).^2 ./ var_vec

  mat_tmp.CopyFromMat(fp_s); // fp_s
  mat_tmp.AddVecToRows(-1.0, s_mu, 1.0); // fp_s - s_mu
  U_neg.AddMatMat(1.0, fp_hs_, kTrans, mat_tmp, kNoTrans, 0.0); // (fp_s - s_mu)' * fp_hs
  d_neg.AddColSumMat(1.0, mat_tmp, 0.0);
  ee_neg.AddColSumMat(1.0, fp_hs_, 0.0);

  /////////////////////////////////////////////////////////////////
  BaseFloat lr = learn_rate_ / n;
  BaseFloat wc = -learn_rate_ * weight_cost_;

  bt_pos.AddVec(-1.0, bt_neg, 1.0); // bt_pos - bt_neg
  bt_inc.AddVec(lr, bt_pos, momentum_ ); // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg)
  bt_inc.AddVec(wc, bt_, 1.0); // momentum * bt_inc + epsilon/n * (bt_pos - bt_neg) - epsilon * wtcost * bt

  lamt2_pos.AddVec(-1.0, lamt2_neg, 1.0); // lamt2_pos - lamt2_neg
  lamt2_inc.AddVec(lr, lamt2_pos, momentum_);
  lamt2_inc.AddVec(wc, lamt2_, 1.0);

  gamma2_pos.AddVec(-1.0, gamma2_neg, 1.0);
  gamma2_inc.AddVec(lr, gamma2_pos, momentum_);
  gamma2_inc.AddVec(wc, gamma2_, 1.0);

  d_pos.AddVec(-1.0, d_neg, 1.0);
  d_inc.AddVec(lr, d_pos, momentum_);

  ee_pos.AddVec(-1.0, ee_neg, 1.0);
  ee_inc.AddVec(lr, ee_pos, momentum_);

  U_pos.AddMat(-1.0, U_neg, 1.0);
  U_inc.AddMat(lr, U_pos, momentum_);
  U_inc.AddMat(wc, noise_vis_hid_, 1.0);

  bt_.AddVec(1.0, bt_inc, 1.0);
  lamt2_.AddVec(1.0, lamt2_inc, 1.0);
  gamma2_.AddVec(1.0, gamma2_inc, 1.0);
  noise_vis_bias_.AddVec(1.0, d_inc, 1.0);
  noise_hid_bias_ori_.AddVec(1.0, ee_inc, 1.0);
  noise_vis_hid_.AddMat(1.0, U_inc, 1.0);

  gamma2_.ApplyFloor(0.0);
  lamt2_.ApplyFloor(0.0);

}  // end Learn()

}  // end namespace

