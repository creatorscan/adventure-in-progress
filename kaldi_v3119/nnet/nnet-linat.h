/*
 *
 *	Troy Lee (troy.lee2008@gmail.com)
 *  Nov. 12, 2013
 *
 *	LIN Affine Transformation layer.
 */
 #ifndef KALDI_NNET_NNET_LINAT_H_
 #define KALDI_NNET_NNET_LINAT_H_

 #include "nnet/nnet-component.h"
 #include "cudamatrix/cu-math.h"
 #include "nnet/nnet-affine-transform.h"

 namespace kaldi {
 namespace nnet1 {
 	class LinAT : public AffineTransform {
 		public:
 			LinAT(int32 dim_in, int32 dim_out)
 			: AffineTransform(dim_in, dim_out),
 			lin_type_(0), num_blks_(0), blk_dim_(0),
 			mask_(dim_out, dim_in),
 			bias_cpu_(dim_out)
 			{
 			}

 			~LinAT() {
 			}

 			Component* Copy() const { return new LinAT(*this); }
  			ComponentType GetType() const { return kLinAT; }

  			std::string Info() const {
			    return std::string("\n  linearity") + MomentStatistics(linearity_) +
			           "\n  bias" + MomentStatistics(bias_);
			}

  			void SetLinType(int32 tid, int32 num_blks = 0, int32 blk_dim = 0) {
  				KALDI_ASSERT(tid >= 0 && tid <= 3);
  				lin_type_ = tid;

  				if(tid == 1) { /* diagonal LIN */
  					Matrix<BaseFloat> mat(output_dim_, input_dim_, kSetZero);
  					mat.SetUnit();
  					mask_.CopyFromMat(mat);
  				} else if (tid==2 || tid==3) {
  					KALDI_ASSERT(num_blks > 0 && blk_dim > 0 && num_blks * blk_dim == input_dim_);
  					num_blks_ = num_blks;
  					blk_dim_ = blk_dim;
  					Matrix<BaseFloat> mat(output_dim_, input_dim_, kSetZero);
  					for(int32 i=0; i<num_blks_; ++i) {
  						int32 offset = i * blk_dim_; 
  						for(int32 r=0; r<blk_dim_; ++r) {
  							for(int32 c=0; c<blk_dim_; ++c) {
  								mat(r+offset, c+offset)=1.0;
  							}
  						}
  					}
  					mask_.CopyFromMat(mat);
  				}
  			}

  			int32 GetLinType() const {
  				return lin_type_;
  			}

  			void ReadData(std::istream &is, bool binary) {
  				// read in LinAT type first
  				ReadBasicType(is, binary, &lin_type_);
  				if(lin_type_ == 2 || lin_type_ == 3) {
  					ReadBasicType(is, binary, &num_blks_);
  					ReadBasicType(is, binary, &blk_dim_);
  					KALDI_ASSERT(num_blks_ * blk_dim_ == input_dim_);
  				}

  				AffineTransform::ReadData(is, binary);

  				SetLinType(lin_type_, num_blks_, blk_dim_);

  			}

  			void WriteData(std::ostream &os, bool binary) const {
  				WriteBasicType(os, binary, lin_type_);
  				if(lin_type_ == 2 || lin_type_ == 3) { 
  					WriteBasicType(os, binary, num_blks_);
  					WriteBasicType(os, binary, blk_dim_);
  				}

  				AffineTransform::WriteData(os, binary);
  			}

  			void Update(const CuMatrix<BaseFloat> &input, const CuMatrix<BaseFloat> &diff) {
  				// update as original AT first 
  				AffineTransform::Update(input, diff);
  				// constrain the weights after each update
  				switch (lin_type_) {
  					case 1: /* diagonal AT */
  					case 2: /* block diagonal AT */
  						linearity_.MulElements(mask_);
  						break;
  					case 3: /* constrained block diagonal AT */
  						linearity_.MulElements(mask_);
  						/* average all the blocks */
  						bias_.CopyToVec(&bias_cpu_);
  						CuSubMatrix<BaseFloat> blk0(linearity_, 0, blk_dim_, 0, blk_dim_);
  						SubVector<BaseFloat> vec0(bias_cpu_, 0, blk_dim_);
  						int32 offset, i;
  						for(i=1; i<num_blks_; ++i) {
  							offset=i*blk_dim_;
  							blk0.AddMat(1.0, CuSubMatrix<BaseFloat>(linearity_, offset, blk_dim_, offset, blk_dim_));
  							vec0.AddVec2(1.0, SubVector<BaseFloat>(bias_cpu_, offset, blk_dim_));
  						}
  						blk0.Scale(1.0 / num_blks_);
  						vec0.Scale(1.0 / num_blks_);
  						/* copy back */
  						for(i=1; i<num_blks_; ++i) {
  							offset = i * blk_dim_;
  							(CuSubMatrix<BaseFloat>(linearity_, offset, blk_dim_, offset, blk_dim_)).CopyFromMat(blk0);
  							(SubVector<BaseFloat>(bias_cpu_, offset, blk_dim_)).CopyFromVec(vec0);
  						}
  						bias_.CopyFromVec(bias_cpu_);
  						break;
  				}
  			}
  

 		protected:
 			/*
 			 * LinAT type code: 0 - standard AT
 			 					1 - diagonal AT
 			 					2 - block diagonal AT, requires num_blks_ and blk_dim_;
 			 					3 - shared block diagonal AT, requires num_blks_ and blk_dim_;
 			 */
 			 int32 lin_type_;

 			 int32 num_blks_;
 			 int32 blk_dim_;

 			 /* for weight structure constraints */
 			 CuMatrix<BaseFloat> mask_;

 			 Vector<BaseFloat> bias_cpu_;


 	};

 } // namespace nnet1
 } // namespace kaldi

 #endif