#include <iostream>
#include <vector>
#include <utility>  // for std::pair

#include "Sliced_Wasserstein.h"
#include "Persistence_weighted_gaussian.h"
#include "../utils.h"

// *******************
// Kernel evaluations.
// *******************

double sw(const std::vector<std::pair<double, double>>& diag1, const std::vector<std::pair<double, double>>& diag2, double sigma, int N) {
  Sliced_Wasserstein sw1(diag1, sigma, N);
  Sliced_Wasserstein sw2(diag2, sigma, N);
  return sw1.compute_scalar_product(sw2);
}

double pwg(const std::vector<std::pair<double, double>>& diag1, const std::vector<std::pair<double, double>>& diag2, std::string kernel, std::string weight,
           double sigma = 1.0, double c = 1.0, double d = 1.0) {

  Weight weight_fn;
  if(weight.compare("linear") == 0)  weight_fn = linear_weight;
  if(weight.compare("arctan") == 0)  weight_fn = arctan_weight(1.0,1.0);
  if(weight.compare("const")  == 0)  weight_fn = const_weight;
  if(weight.compare("pss")  == 0)    weight_fn = pss_weight;

  Kernel kernel_fn;
  if(kernel.compare("rbf") == 0)  kernel_fn = rbf_kernel(sigma);
  if(kernel.compare("poly") == 0) kernel_fn = poly_kernel(c,d);

  Persistence_weighted_gaussian pwg1(diag1, kernel_fn, weight_fn);
  Persistence_weighted_gaussian pwg2(diag2, kernel_fn, weight_fn);
  return pwg1.compute_scalar_product(pwg2);
}


// ****************
// Kernel matrices.
// ****************

std::vector<std::vector<double> > sw_matrix(const std::vector<std::vector<std::pair<double, double> > >& s1, const std::vector<std::vector<std::pair<double, double> > >& s2, double sigma, int N){
  std::vector<std::vector<double> > matrix;
  std::vector<Sliced_Wasserstein> ss1;
  int num_diag_1 = s1.size(); for(int i = 0; i < num_diag_1; i++){Sliced_Wasserstein sw1(s1[i], sigma, N); ss1.push_back(sw1);}
  std::vector<Sliced_Wasserstein> ss2;
  int num_diag_2 = s2.size(); for(int i = 0; i < num_diag_2; i++){Sliced_Wasserstein sw2(s2[i], sigma, N); ss2.push_back(sw2);}
  for(int i = 0; i < num_diag_1; i++){
    std::cout << 100.0*i/num_diag_1 << " %" << std::endl;
    std::vector<double> ps; for(int j = 0; j < num_diag_2; j++) ps.push_back(ss1[i].compute_scalar_product(ss2[j])); matrix.push_back(ps);
  }
  return matrix;
}

std::vector<std::vector<double> > pwg_matrix(const std::vector<std::vector<std::pair<double, double> > >& s1, const std::vector<std::vector<std::pair<double, double> > >& s2, std::string kernel, std::string weight, double sigma = 1.0, double c = 1.0, double d = 1.0){
  std::vector<std::vector<double> > matrix; int num_diag_1 = s1.size(); int num_diag_2 = s2.size();
  for(int i = 0; i < num_diag_1; i++){
    std::cout << 100.0*i/num_diag_1 << " %" << std::endl;
    std::vector<double> ps; for(int j = 0; j < num_diag_2; j++) ps.push_back(pwg(s1[i], s2[j], kernel, weight, sigma, c, d)); matrix.push_back(ps);
  }
  return matrix;
}
