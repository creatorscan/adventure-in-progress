// nnet/nnet-maskedbl.h

#ifndef KALDI_NNET_MASKEDBL_H
#define KALDI_NNET_MASKEDBL_H

#include "nnet/nnet-biasedlinearity.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {

class MaskedBL : public BiasedLinearity {
 public:
  MaskedBL(MatrixIndexT dim_in, MatrixIndexT dim_out, Nnet *nnet)
      : BiasedLinearity(dim_in, dim_out, nnet),
        mask_(dim_out, dim_in)
  {
  }
  ~MaskedBL()
  {
  }

  ComponentType GetType() const {
    return kMaskedBL;
  }

  void ReadData(std::istream &is, bool binary) {

    BiasedLinearity::ReadData(is, binary);

    mask_.Read(is, binary);

    KALDI_ASSERT(mask_.NumRows() == output_dim_);
    KALDI_ASSERT(mask_.NumCols() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {

    BiasedLinearity::WriteData(os, binary);

    mask_.Write(os, binary);
  }

  /*
   * Only the updat is different from the biased linearity layer
   */
  void Update(const CuMatrix<BaseFloat> &input,
              const CuMatrix<BaseFloat> &err) {

    BiasedLinearity::Update(input, err);

    // apply mask to the new weight
    linearity_.MulElements(mask_);

  }

  void SetMask(const Matrix<BaseFloat> &mask){
    KALDI_ASSERT(mask.NumRows() == output_dim_);
    KALDI_ASSERT(mask.NumCols() == input_dim_);

    mask_.CopyFromMat(mask);
  }

 private:
  CuMatrix<BaseFloat> mask_;
};

}  // namespace

#endif
