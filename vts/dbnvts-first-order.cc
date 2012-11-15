/*
 * dbnvts-first-order.cc
 *
 *  Created on: Oct 26, 2012
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
#include "lat/kaldi-lattice.h" // for CompactLatticeArc
#include "gmm/diag-gmm-normal.h"
#include "nnet/nnet-component.h"

#include "vts/dbnvts-first-order.h"

namespace kaldi {


/*
 * The negative mean is the reflection of the positive mean against the
 * decision hyperplane.
 *
 * mu_neg = mu_pos - 2 * w.* var_pos * (w_T * mu_pos + b) / (w_T * w)
 *
 * var_pos is the shared var for both negative and positive classes.
 *
 * The param neg_am_gmm must be initialized the same as pos_am_gmm.
 *
 */
void ComputeNegativeGmm(const Matrix<BaseFloat> &weight,
                        const Vector<BaseFloat> &bias,
                        const Vector<BaseFloat> &pos2neg_log_prior_ratio,
                        AmDiagGmm &neg_am_gmm, Vector<double> &var_scales) {

  int32 feat_dim = weight.NumCols();
  Vector<double> cur_weight(feat_dim, kSetZero);
  Vector<double> w_mu(feat_dim, kSetZero), pos_mean(feat_dim, kSetZero);
  Vector<double> w_w(feat_dim, kSetZero);
  double w_norm = 0.0, coef = 0.0, b_prior, b_gauss;
  double cur_bias = 0.0;

  // iterate all the GMMs
  int32 num_pdf = neg_am_gmm.NumPdfs();
  KALDI_ASSERT(num_pdf == weight.NumRows());
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    cur_weight.CopyRowFromMat(weight, pdf);
    cur_bias = bias(pdf);

    w_norm = cur_weight.Norm(2.0);
    cur_weight.Scale(1.0 / w_norm);
    cur_bias = cur_bias / w_norm;

    // each pdf corresponds to one hidden unit
    w_w.AddVecVec(1.0, cur_weight, cur_weight, 0.0);

    // iterate all the Gaussians
    DiagGmm *gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal ngmm(*gmm);

    int32 num_gauss = gmm->NumGauss();
    for (int32 g = 0; g < num_gauss; ++g) {

      pos_mean.CopyRowFromMat(ngmm.means_, g);  // keep a copy of the pos_mean

      w_mu.AddVecVec(1.0, cur_weight, ngmm.means_.Row(g), 0.0);  // w_T * mu_pos
      coef = -2.0 * (w_mu.Sum() + cur_bias) / w_w.Sum();  // -2.0 * (w_T * mu_pos + b) / (w_T * w)
      w_mu.AddVecVec(coef, cur_weight, ngmm.vars_.Row(g), 0.0);  // - 2 * w.* var_pos * (w_T * mu_pos + b) / (w_T * w)
      (ngmm.means_.Row(g)).AddVec(1.0, w_mu);

      // using the current neg_mean estiamtion to find the scale for the shared covariance
      cur_weight.CopyFromVec(pos_mean);  // estimate the new boundary weights
      cur_weight.AddVec(-1.0, ngmm.means_.Row(g));  // pos_mean - neg_mean
      cur_weight.DivElements(ngmm.vars_.Row(g));  // (pos_mean - neg_mean)./var_shared

      w_norm = cur_weight.Norm(2.0);
      b_prior = pos2neg_log_prior_ratio(pdf);
      w_mu.AddVecVec(1.0, pos_mean, pos_mean, 0.0);  // pos_mean .* pos_mean
      w_mu.AddVecVec(-1.0, ngmm.means_.Row(g), ngmm.means_.Row(g), 1.0);  // (pos_mean .* pos_mean - neg_mean .* neg_mean)
      w_mu.DivElements(ngmm.vars_.Row(g));  // (pos_mean .* pos_mean - neg_mean .* neg_mean) ./ var_shared
      b_gauss = 0.5 * w_mu.Sum();

      var_scales(pdf) = (w_norm * cur_bias + b_gauss) / b_prior;  // w_norm * cur_bias /scale = b_prior - b_gauss / scale

    }
    ngmm.CopyToDiagGmm(gmm);
    gmm->ComputeGconsts();

  }

  // generate the scaled GMMs

}

void ComputeGaussianBoundary(
    const AmDiagGmm &pos_am_gmm, const AmDiagGmm &neg_am_gmm,
    const Vector<BaseFloat> &pos2neg_log_prior_ratio,  // prior pos/neg ratio
    const Vector<double> &var_scale, Matrix<BaseFloat> &weight,
    Vector<BaseFloat> &bias) {

  Vector<BaseFloat> new_w(weight.NumCols()), new_b(weight.NumCols());
  Vector<double> var_shared(weight.NumCols());

  // iterate all the GMMs
  int32 num_pdf = pos_am_gmm.NumPdfs();
  KALDI_ASSERT(pos_am_gmm.NumPdfs() == neg_am_gmm.NumPdfs());
  KALDI_ASSERT(num_pdf == weight.NumRows());
  for (int32 pdf = 0; pdf < num_pdf; ++pdf) {
    // each pdf corresponds to one hidden unit

    // iterate all the Gaussians
    const DiagGmm *pos_gmm = &(pos_am_gmm.GetPdf(pdf));
    DiagGmmNormal pos_ngmm(*pos_gmm);

    const DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    DiagGmmNormal neg_ngmm(*neg_gmm);

    KALDI_ASSERT(pos_gmm->NumGauss() == 1 && neg_gmm->NumGauss() == 1);

    var_shared.CopyRowFromMat(pos_ngmm.vars_, 0);
    var_shared.Scale(var_scale(pdf));

    // weight is the vector from negative mean to positive mean
    new_w.CopyRowFromMat(pos_ngmm.means_, 0);
    new_w.AddVec(-1.0, neg_ngmm.means_.Row(0));  // new_w = mu_pos - mu_neg
    new_w.DivElements(var_shared);  // new_w = (mu_pos - mu_neg) / var_shared
    /*
     new_w.Scale((weight.Row(pdf)).Norm(2.0)/new_w.Norm(2.0));  // scale the norm to be similar to original weight
     */

    weight.CopyRowFromVec(new_w, pdf);

    new_b.CopyRowFromMat(pos_ngmm.means_, 0);  // new_b = mu_pos
    new_b.ApplyPow(2.0);  // new_b = mu_pos * mu_pos
    new_b.AddVec2(-1.0, neg_ngmm.means_.Row(0));  // new_b = mu_pos * mu_pos - mu_neg * mu_neg
    new_b.DivElements(var_shared);  // new_b = (mu_pos * mu_pos - mu_neg * mu_neg) / var_shared

    bias(pdf) = pos2neg_log_prior_ratio(pdf) - 0.5 * new_b.Sum();  // new_b = -0.5 * sum((mu_pos * mu_pos - mu_neg * mu_neg) / var_shared)

    if (var_scale(pdf) < 0) {
      (weight.Row(pdf)).Scale(-1.0);
      bias(pdf) = -bias(pdf);
    }

    /*
     // bias is computed using the middle point
     new_w.CopyRowFromMat(pos_ngmm.means_, 0);
     new_w.AddVec(1.0, neg_ngmm.means_.Row(0));  // new_w = mu_pos + mu_neg
     new_w.Scale(0.5);  // new_w = 0.5 * (mu_pos + mu_neg), i.e. the middle point
     new_w.MulElements(weight.Row(pdf));
     bias(pdf) = -new_w.Sum();
     */
  }

}


}

