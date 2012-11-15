/*
 * dbnvts2-first-order.h
 *
 *  Created on: Nov 8, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef VTS_DBNVTS2_FIRST_ORDER_H_
#define VTS_DBNVTS2_FIRST_ORDER_H_

#include "base/kaldi-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

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
                                  Vector<BaseFloat> &bias);

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
 */
void ComputeNegativeReflectionGmm(const Matrix<BaseFloat> &weight,
                                  const Vector<BaseFloat> &bias,
                                  const Vector<double> &pos2neg_log_prior_ratio,
                                  AmDiagGmm &neg_am_gmm,
                                  Vector<double> &var_scale,
                                  Vector<double> &llr_scale);

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
//    Vector<double> &var_scales);

/*
 * Interpolate the positive and negative covariance to make them same.
 *
 */
void InterpolateVariance(BaseFloat pos_weight, AmDiagGmm &pos_am_gmm,
                         AmDiagGmm &neg_am_gmm);

/*
 * Scale the covariance
 *
 */
void ScaleVariance(const Vector<double> &var_scale, AmDiagGmm &pos_am_gmm,
                   AmDiagGmm &neg_am_gmm);

/*
 * Compute the log likelihood ratio of two single Gaussian based AM models.
 *
 */
void ComputeGaussianLogLikelihoodRatio(
    const Matrix<BaseFloat> &feats, const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm, const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &llr_scale,
    Matrix<BaseFloat> &llr);

/*
 * with interpolation .
 */
void ComputeGaussianLogLikelihoodRatio_Interpolate(
    const Matrix<BaseFloat> &feats, const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm, const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &llr_scale, const Matrix<BaseFloat> &ori_weights,
    const Vector<BaseFloat> &ori_bias, BaseFloat new_ratio, Matrix<BaseFloat> &llr);

/*
 * Compute the log likelihood ratio of two single Gaussian based AM models.
 *
 * Gaussians can have different variances.
 *
 */
void ComputeGaussianLogLikelihoodRatio_General(
    const Matrix<BaseFloat> &feats, const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm, const Vector<double> &pos2neg_log_prior_ratio,
    const Vector<double> &llr_scale,
    Matrix<BaseFloat> &llr);

}

#endif /* VTS_DBNVTS2_FIRST_ORDER_H_ */
