#include <iostream>
#include <vector>
#include <utility>  // for std::pair

#include "Landscape.h"
#include "Persistence_image.h"
#include "../utils.h"

std::vector<std::vector<double> > compute_ls(const std::vector<std::pair<double, double> >& diag, int nb_ls, double min_x, double max_x, int res_x) {
  Landscape L(diag, nb_ls, min_x, max_x, res_x);
  return L.vectorize();
}

std::vector<std::vector<double> > compute_pim(const std::vector<std::pair<double, double> >& diag, double min_x, double max_x, int res_x, double min_y, double max_y, int res_y, std::string kernel, std::string weight, double sigma = 1.0, double c = 1.0, double d = 1.0) {

  Weight weight_fn;
  if(weight.compare("linear") == 0)  weight_fn = linear_weight;
  if(weight.compare("arctan") == 0)  weight_fn = arctan_weight(1.0,1.0);
  if(weight.compare("const")  == 0)  weight_fn = const_weight;
  if(weight.compare("pss")  == 0)    weight_fn = pss_weight;

  Kernel kernel_fn;
  if(kernel.compare("rbf") == 0)  kernel_fn = rbf_kernel(sigma);
  if(kernel.compare("poly") == 0) kernel_fn = poly_kernel(c,d);

  Persistence_image P(diag, min_x, max_x, res_x, min_y, max_y, res_y, kernel_fn, weight_fn);
  return P.vectorize();
}
