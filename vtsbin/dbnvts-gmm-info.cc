/*
 * dbnvts-gmm-info.cc
 *
 *  Created on: Oct 29, 2012
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Write to standard output various properties of DBNVTS postive/negative GMM-based model\n"
        "Usage:  dbnvts-gmm-info [options] <model-in>\n"
        "e.g.:\n"
        " dbnvts-gmm-info pos.mdl\n";

    ParseOptions po(usage);

    bool detailed = false;
    po.Register("detailed", &detailed, "Print detail Gaussian information for each PDF");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);

    AmDiagGmm am_gmm;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      am_gmm.Read(ki.Stream(), binary_read);
    }

    std::cout << "number of pdfs " << am_gmm.NumPdfs() << '\n';
    std::cout << "feature dimension " << am_gmm.Dim() << '\n';
    std::cout << "number of gaussians " << am_gmm.NumGauss() << '\n';
    if (detailed){
      std::cout<< "number of gaussians per pdf: [\n";
      for(int32 i=0; i<am_gmm.NumPdfs(); ++i){
        std::cout<< am_gmm.NumGaussInPdf(i) << ", ";
      }
      std::cout<< "]\n";
    }
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}




