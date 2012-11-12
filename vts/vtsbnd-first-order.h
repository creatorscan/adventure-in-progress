/*
 * vtsbnd-first-order.h
 *
 *  Created on: Nov 7, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#ifndef VTS_VTSBND_FIRST_ORDER_H_
#define VTS_VTSBND_FIRST_ORDER_H_

#include "base/kaldi-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"

namespace kaldi {

void ComputeFirstOrderVTSGaussianBoundary(
    const AmDiagGmm &pos_am_gmm,
    const AmDiagGmm &neg_am_gmm,
    const Vector<BaseFloat> &pos2neg_log_prior_ratio,  // prior pos/neg ratio
    const Vector<double> &var_scale, Matrix<BaseFloat> &weight,
    Vector<BaseFloat> &bias);

}

#endif /* VTS_VTSBND_FIRST_ORDER_H_ */
