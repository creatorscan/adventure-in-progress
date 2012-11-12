/*
 * vtsbnd-first-order.cc
 *
 *  Created on: Nov 7, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 * 1st order VTS compensation for the DBN input layer.
 *
 * General steps:
 * 1. Estimate the positive and negative sample distribution, using a single Gaussian;
 * 2. Compensate the two Gaussians;
 * 3. Estimate the new linear boundary using 1st order VTS approximation of the original
 *  quadratic boundary;
 * 4. Use the linear boundary as the new DBN input layer weights.
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

#include "vts/vts-first-order.h"

namespace kaldi {

void ComputeFirstOrderVTSGaussianBoundary(
    const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm,
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
    KALDI_ASSERT(pos_gmm->NumGauss() == 1);
    DiagGmmNormal pos_ngmm(*pos_gmm);
    Vector<double> pos_inv_var(pos_ngmm.vars_.Row(0));
    pos_inv_var.Scale(var_scale(pdf)); // scale the covariance
    pos_inv_var.InvertElements(); // inverse covariance

    const DiagGmm *neg_gmm = &(neg_am_gmm.GetPdf(pdf));
    KALDI_ASSERT(neg_gmm->NumGauss() == 1);
    DiagGmmNormal neg_ngmm(*neg_gmm);
    Vector<double> neg_inv_var(neg_ngmm.vars_.Row(0));
    neg_inv_var.Scale(var_scale(pdf));
    neg_inv_var.InvertElements();

    Vector<double> delta_inv_var(pos_inv_var);
    delta_inv_var.AddVec(-1.0, neg_inv_var); // pos_inv_var - neg_inv_var

    double neg_ratio = 1.0 / (1.0 + exp(pos2neg_log_prior_ratio(pdf)));
    double pos_ratio = 1.0 - neg_ratio;

    // expansion point
    Vector<double> x0(pos_ngmm.means_.Row(0));
    x0.Scale(pos_ratio);
    x0.AddVec(neg_ratio, neg_ngmm.means_.Row(0));

    Vector<double> tmp(x0.Dim());

    Vector<double> dfx0(delta_inv_var);
    dfx0.MulElements(x0); // (pos_inv_var - neg_inv_var) * x0
    dfx0.Scale(-1.0);

    double fx0 = pos2neg_log_prior_ratio(pdf); // log ( theta / (1-theta) )
    // - 0.5 * log (det_pos/det_neg)
    for(int32 i=0; i<pos_inv_var.Dim(); ++i){
      fx0 = fx0 - 0.5 * (log(pos_ngmm.vars_(0,i)) - log(neg_ngmm.vars_(0,i)));
    }
    tmp.SetZero();
    tmp.AddVec2(-0.5, x0); // -0.5 * x0 .* x0
    tmp.MulElements(delta_inv_var);
    fx0 = fx0 + tmp.Sum(); // -0.5 * x0_T * (pos_inv_var - neg_inv_var) * x0
    tmp.CopyFromVec(pos_ngmm.means_.Row(0));
    tmp.MulElements(pos_inv_var);
    dfx0.AddVec(1.0, tmp); // pos_mu_T * pos_inv_var
    tmp.MulElements(x0);
    fx0 = fx0 + tmp.Sum(); // pos_mu_T * pos_inv_var * x0
    tmp.CopyFromVec(neg_ngmm.means_.Row(0));
    tmp.MulElements(neg_inv_var);
    dfx0.AddVec(-1.0, tmp); // - neg_mu_T * neg_inv_var
    tmp.MulElements(x0);
    fx0 = fx0 - tmp.Sum(); // - neg_mu_T * neg_inv_var * x0
    tmp.SetZero();
    tmp.AddVec2(-0.5, pos_ngmm.means_.Row(0));
    tmp.MulElements(pos_inv_var);
    fx0 = fx0 + tmp.Sum(); // -0.5 * pos_mu_T * pos_inv_var * pos_mu
    tmp.SetZero();
    tmp.AddVec2(0.5, neg_ngmm.means_.Row(0));
    tmp.MulElements(neg_inv_var);
    fx0 = fx0 + tmp.Sum(); // 0.5 * neg_mu_T * neg_inv_var * neg_mu

    Vector<BaseFloat> new_w(dfx0); // convert from double to float
    weight.CopyRowFromVec(new_w, pdf);

    tmp.CopyFromVec(dfx0);
    tmp.MulElements(x0);
    bias(pdf) = fx0 - tmp.Sum();

  }

}

}

