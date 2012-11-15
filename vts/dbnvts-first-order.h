/*
 * dbnvts-first-order.h
 *
 *  Created on: Oct 26, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  First order VTS model compensation for DBN's first layer.
 *
 *  General steps:
 *  1. Estimate the positive sample distribution using a Gaussian;
 *  2. Using Geometry to find the relection Gaussian as the negative sample distribution;
 *  3. Compensated the two Gaussians using 1st order VTS;
 *  4. Re-estimate the boundary of the compensated Gaussians;
 *  5. Use the boundary as the DBN first layer's weights.
 *
 */

#ifndef KALDI_VTS_DBNVTS_FIRST_ORDER_H_
#define KALDI_VTS_DBNVTS_FIRST_ORDER_H_

#include <vector>

#include "base/kaldi-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

namespace kaldi {





/*
 * Compute the negative GMM based on the positive GMM and the decision boundary.
 *
 * Assumption: Postive and negative are conjugate to each other with respect to the
 * decision boundary.
 *
 * Thus the covariances will be the same, only the means are different.
 *
 * The parameter neg_am_gmm is assumed to be a copy of pos_am_gmm initially.
 *
 */
void ComputeNegativeGmm(const Matrix<BaseFloat> &weight,
                        const Vector<BaseFloat> &bias,
                        const Vector<BaseFloat> &pos2neg_log_prior_ratio,
                        AmDiagGmm &neg_am_gmm, Vector<double> &var_scales);

/*
 * Based on the positive and negative GMMs, compute the decision boundaries.
 *
 * To avoid the weights going too large, we use the unit normalized weights.
 *
 * Assuming each GMM is a single Gaussian.
 *
 */
void ComputeGaussianBoundary(const AmDiagGmm &pos_am_gmm,
                             const AmDiagGmm &neg_am_gmm,
                             const Vector<BaseFloat> &pos2neg_log_prior_ratio,
                             const Vector<double> &var_scale,
                             Matrix<BaseFloat> &weight,
                             Vector<BaseFloat> &bias);



}

#endif /* KALDI_VTS_DBNVTS_FIRST_ORDER_H_ */
