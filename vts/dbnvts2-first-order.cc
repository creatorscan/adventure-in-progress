/*
 * dbnvts2-first-order.cc
 *
 *  Created on: Nov 8, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-am-diag-gmm.h"
#include "util/timer.h"
#include "gmm/diag-gmm-normal.h"
#include "nnet/nnet-component.h"

#include "vts/dbnvts2-first-order.h"

namespace kaldi {

/*
 * weight and bias are initialised as the DBN input layer weights,
 * i.e. in the normalized feature space.
 *
 * w_dis = w ./ norm_std;
 * b_dis = b - w .* norm_mean ./ norm_std;
 *
 */
void ConvertWeightToOriginalSpace(int32 num_frames,
                                  const Vector<double> &norm_mean,
                                  const Vector<double> &norm_std,
                                  Matrix<BaseFloat> &weight,
                                  Vector<BaseFloat> &bias) {
  int32 feat_dim = norm_mean.Dim();
  double tmp = 0.0;
  KALDI_ASSERT(num_frames * feat_dim == weight.NumCols());

  for (int32 r = 0; r < weight.NumRows(); ++r) {
    tmp = 0.0;
    for (int32 c = 0; c < weight.NumCols(); ++c) {
      tmp += weight(r, c) * norm_mean(c % feat_dim) / norm_std(c % feat_dim);
      weight(r, c) = weight(r, c) / norm_std(c % feat_dim);
    }
    bias(r) = bias(r) - tmp;
  }

}

/*
 * The negative mean is the reflection of the positive mean against the
 * decision hyperplane.
 *
 * The weight and bias are in the original feature space.
 *
 * mu_neg = mu_pos - w.* var_pos
 *
 * var_pos is the shared var for both negative and positive classes.
 *
 * The param neg_am_gmm must be initialized the same as pos_am_gmm.
 *
 * llh_scale to make sure not only the sign but also the value is the same
 * as the logistic regression.
 *
 */
void ComputeNegativeReflectionGmm(const Matrix<BaseFloat> &weight,
                                  const Vector<BaseFloat> &bias,
                                  const Vector<double> &pos2neg_log_prior_ratio,
                                  AmDiagGmm &neg_am_gmm,
                                  Vector<double> &var_scale,
                                  Vector<double> &llr_scale) {

  int32 feat_dim = weight.NumCols();
  Vector<double> w_mu(feat_dim, kSetZero);
  Vector<double> w_w(feat_dim, kSetZero);
  double coef = 0.0, w2, wpos;

  // iterate all the GMMs
  int32 num_pdf = neg_am_gmm.NumPdfs();
  KALDI_ASSERT(num_pdf == weight.NumRows());
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {

    // iterate all the Gaussians
    DiagGmm *gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    w_w.SetZero();
    w_w.AddVec2(1.0, weight.Row(pdf));
    w2 = w_w.Sum();

    KALDI_ASSERT(gmm->NumGauss()==1);
    // compute the var scale first

    w_mu.CopyRowFromMat(weight, pdf);
    w_mu.MulElements(ngmm.means_.Row(0));  // w .* mu_pos
    wpos = w_mu.Sum() + bias(pdf);  // w .* mu_pos + b

    w_mu.CopyFromVec(w_w);
    w_mu.MulElements(ngmm.vars_.Row(0));  // w_T * var_pos * w

    var_scale(pdf) = ((wpos * wpos / w2) - 0.5 * pos2neg_log_prior_ratio(pdf))
        / ((wpos * wpos * w_mu.Sum()) / (w2 * w2));

    llr_scale(pdf) = 0.5 * w2 / wpos;  // w_T * w / (2 * (w_T * mu_pos + b))

    coef = 2 * var_scale(pdf) * wpos / w2;
    w_mu.CopyRowFromMat(weight, pdf);  // w
    w_mu.MulElements(ngmm.vars_.Row(0));  // w .* var_pos
    (ngmm.means_.Row(0)).AddVec(-coef, w_mu);  // mu_pos - w .* var_pos

    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();

  }

}

///*
// * The negative mean is the reflection of the positive mean against the
// * decision hyperplane.
// *
// * The weight and bias are in the original feature space.
// *
// * mu_neg = mu_pos - 2 * w.* var_pos * (w_T * mu_pos + b) / (w_T * w)
// *
// * var_pos is the shared var for both negative and positive classes.
// *
// * The param neg_am_gmm must be initialized the same as pos_am_gmm.
// *
// */
//void ComputeNegativeReflectionGmm(
//    const Matrix<BaseFloat> &weight, const Vector<BaseFloat> &bias,
//    const Vector<BaseFloat> &pos2neg_log_prior_ratio, AmDiagGmm &neg_am_gmm,
//    Vector<double> &var_scales) {
//
//  int32 feat_dim = weight.NumCols();
//  Vector<double> cur_weight(feat_dim, kSetZero);
//  Vector<double> w_mu(feat_dim, kSetZero), pos_mean(feat_dim, kSetZero);
//  Vector<double> w_w(feat_dim, kSetZero);
//  double coef = 0.0;
//  double cur_bias = 0.0;
//
//  // iterate all the GMMs
//  int32 num_pdf = neg_am_gmm.NumPdfs();
//  KALDI_ASSERT(num_pdf == weight.NumRows());
//  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
//    cur_weight.CopyRowFromMat(weight, pdf);
//    cur_bias = bias(pdf);
//
//    /* Normalize the original weights, is it necessary?
//     * double w_norm = cur_weight.Norm(2.0);
//     cur_weight.Scale(1.0 / w_norm);
//     cur_bias = cur_bias / w_norm;*/
//
//    // each pdf corresponds to one hidden unit
//    w_w.AddVecVec(1.0, cur_weight, cur_weight, 0.0);
//
//    // iterate all the Gaussians
//    DiagGmm *gmm = &(neg_am_gmm.GetPdf(pdf));
//    DiagGmmNormal ngmm(*gmm);
//
//    int32 num_gauss = gmm->NumGauss();
//    for (int32 g = 0; g < num_gauss; ++g) {
//
//      pos_mean.CopyRowFromMat(ngmm.means_, g);  // keep a copy of the pos_mean
//
//      w_mu.AddVecVec(1.0, cur_weight, ngmm.means_.Row(g), 0.0);  // w_T * mu_pos
//      coef = -2.0 * (w_mu.Sum() + cur_bias) / w_w.Sum();  // -2.0 * (w_T * mu_pos + b) / (w_T * w)
//      w_mu.AddVecVec(coef, cur_weight, ngmm.vars_.Row(g), 0.0);  // - 2 * w.* var_pos * (w_T * mu_pos + b) / (w_T * w)
//      (ngmm.means_.Row(g)).AddVec(1.0, w_mu);
//
//      // using the current neg_mean estimation to find the scale for the shared covariance
//      Vector<double> tmp(pos_mean.Dim(), kSetZero);
//      tmp.AddVec2(1.0, pos_mean);  // pos_mean .* pos_mean
//      tmp.AddVec2(-1.0, ngmm.means_.Row(g));  // pos_mean .* pos_mean - neg_mean .* neg_mean
//      tmp.DivElements(ngmm.vars_.Row(g));  // (pos_mean .* pos_mean - neg_mean .* neg_mean) ./ var
//
//      var_scales(pdf) = (-coef + 0.5 * tmp.Sum())
//          / pos2neg_log_prior_ratio(pdf);  // w_norm * cur_bias /scale = b_prior - b_gauss / scale
//
//    }
//    ngmm.CopyToDiagGmm(gmm);
//    gmm->ComputeGconsts();
//
//  }
//
//}

/*
 * Interpolate the positive and negative covariance to make them same.
 *
 */
void InterpolateVariance(BaseFloat pos_weight, AmDiagGmm &pos_am_gmm,
                         AmDiagGmm &neg_am_gmm) {

  KALDI_ASSERT(pos_weight >= 0.0 && pos_weight<=1.0);

  // iterate all the GMMs
  int32 num_pdf = pos_am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);
    KALDI_ASSERT(pos_gmm->NumGauss()==1);

    DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);
    KALDI_ASSERT(neg_gmm->NumGauss()==1);

    (pos_ngmm.vars_.Row(0)).Scale(pos_weight);
    (pos_ngmm.vars_.Row(0)).AddVec(1 - pos_weight, neg_ngmm.vars_.Row(0));

    neg_ngmm.vars_.CopyRowFromVec(pos_ngmm.vars_.Row(0), 0);

    pos_ngmm.CopyToDiagGmm(pos_gmm);
    pos_gmm->ComputeGconsts();
    neg_ngmm.CopyToDiagGmm(neg_gmm);
    neg_gmm->ComputeGconsts();
  }
}

/*
 * Scale the covariance
 *
 */
void ScaleVariance(const Vector<double> &var_scale, AmDiagGmm &pos_am_gmm,
                   AmDiagGmm &neg_am_gmm) {

  KALDI_ASSERT(
      var_scale.Dim() == pos_am_gmm.NumPdfs() && neg_am_gmm.NumPdfs() == pos_am_gmm.NumPdfs());

  // iterate all the GMMs
  int32 num_pdf = pos_am_gmm.NumPdfs();
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // iterate all the Gaussians
    DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);
    KALDI_ASSERT(pos_gmm->NumGauss()==1);

    DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);
    KALDI_ASSERT(neg_gmm->NumGauss()==1);

    KALDI_ASSERT(var_scale(pdf));
    (pos_ngmm.vars_.Row(0)).Scale(var_scale(pdf));
    (neg_ngmm.vars_.Row(0)).Scale(var_scale(pdf));

    pos_ngmm.CopyToDiagGmm(pos_gmm);
    pos_gmm->ComputeGconsts();
    neg_ngmm.CopyToDiagGmm(neg_gmm);
    neg_gmm->ComputeGconsts();
  }
}

/*
 * Compute the log likelihood ratio of two single Gaussian based AM models.
 *
 * Assumptions:
 *  1. Pos and neg share the same diagonal covariance.
 *
 */
void ComputeGaussianLogLikelihoodRatio(
    const Matrix<BaseFloat> &feats, const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm, const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &llr_scale, Matrix<BaseFloat> &llr) {
  int32 num_pdfs = pos_am_gmm.NumPdfs();
  Vector<double> tmp(pos_am_gmm.Dim());

  // compute the weight and bias from Gaussians

  Matrix<BaseFloat> weights(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
  Vector<BaseFloat> bias(pos_am_gmm.NumPdfs(), kSetZero);

  for (int32 pdf = 0; pdf < num_pdfs; ++pdf) {
    const DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);
    KALDI_ASSERT(pos_gmm->NumGauss()==1);

    const DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);
    KALDI_ASSERT(neg_gmm->NumGauss()==1);

    SubVector<BaseFloat> w(weights, pdf);
    w.CopyRowFromMat(pos_ngmm.means_, 0);
    w.AddVec(-1.0, neg_ngmm.means_.Row(0));
    w.DivElements(pos_ngmm.vars_.Row(0));

    tmp.SetZero();
    tmp.AddVec2(1.0, pos_ngmm.means_.Row(0));
    tmp.AddVec2(-1.0, neg_ngmm.means_.Row(0));
    tmp.DivElements(pos_ngmm.vars_.Row(0));

    bias(pdf) = -0.5 * tmp.Sum() + pos2neg_log_prior_ratio(pdf);
  }

  llr.SetZero();
  llr.AddVecToRows(1.0, bias);
  llr.AddMatMat(1.0, feats, kNoTrans, weights, kTrans, 1.0);

  llr.MulColsVec(Vector<BaseFloat>(llr_scale));

//  for (int32 t = 0; t < num_frames; ++t) {
//    for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
//      llr(t, pdf) = pos_am_gmm.LogLikelihood(pdf, feats.Row(t))
//          - neg_am_gmm.LogLikelihood(pdf, feats.Row(t))
//          + pos2neg_log_prior_ratio(pdf);
//
//      /*
//       const DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
//       DiagGmmNormal pos_ngmm(*pos_gmm);
//       KALDI_ASSERT(pos_gmm->NumGauss()==1);
//
//       const DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
//       DiagGmmNormal neg_ngmm(*neg_gmm);
//       KALDI_ASSERT(neg_gmm->NumGauss()==1);
//
//       tmp.CopyRowFromMat(pos_ngmm.means_, 0);
//       tmp.AddVec(-1.0, neg_ngmm.means_.Row(0));
//       tmp.MulElements(feats.Row(t));
//       tmp.AddVec2(-0.5, pos_ngmm.means_.Row(0));
//       tmp.AddVec2(0.5, neg_ngmm.means_.Row(0));
//       tmp.DivElements(pos_ngmm.vars_.Row(0));
//
//       llr(t, pdf) = tmp.Sum() / var_scale(pdf) + pos2neg_log_prior_ratio(pdf);*/
//    }
//  }

}

/*
 * Compute the log likelihood ratio of two single Gaussian based AM models.
 *
 */
void ComputeGaussianLogLikelihoodRatio_General(
    const Matrix<BaseFloat> &feats, const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm, const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &llr_scale, Matrix<BaseFloat> &llr) {
  int32 num_pdfs = pos_am_gmm.NumPdfs();
  Vector<double> tmp(pos_am_gmm.Dim());

  // compute the weight and bias from Gaussians

  Matrix<BaseFloat> weights(pos_am_gmm.NumPdfs(), pos_am_gmm.Dim(), kSetZero);
  Vector<BaseFloat> bias(pos_am_gmm.NumPdfs(), kSetZero);

  for (int32 pdf = 0; pdf < num_pdfs; ++pdf) {
    const DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);
    KALDI_ASSERT(pos_gmm->NumGauss()==1);

    const DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);
    KALDI_ASSERT(neg_gmm->NumGauss()==1);

    double pconst = pos2neg_log_prior_ratio(pdf);
    tmp.CopyRowFromMat(pos_ngmm.vars_, 0);
    tmp.DivElements(neg_ngmm.vars_.Row(0));
    pconst -= (0.5 * tmp.SumLog());

    for (int32 t = 0; t < feats.NumRows(); ++t) {
      llr(t, pdf) = pconst;

      tmp.CopyRowFromMat(feats, t);
      tmp.AddVec(-1.0, pos_ngmm.means_.Row(0));
      tmp.ApplyPow(2.0);
      tmp.DivElements(pos_ngmm.vars_.Row(0));
      llr(t, pdf) -= (0.5 * tmp.Sum());

      tmp.CopyRowFromMat(feats, t);
      tmp.AddVec(-1.0, neg_ngmm.means_.Row(0));
      tmp.ApplyPow(2.0);
      tmp.DivElements(neg_ngmm.vars_.Row(0));
      llr(t, pdf) += (0.5 * tmp.Sum());

      llr(t, pdf) *= llr_scale(pdf);
    }
  }
}

}

