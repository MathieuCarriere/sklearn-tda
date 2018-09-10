#include "Sliced_Wasserstein.h"
#include "Persistence_weighted_gaussian.h"
#include "../utils.h"

std::vector<std::vector<double> > swk_matrix(const std::vector<std::vector<std::pair<double, double> > >& s1, const std::vector<std::vector<std::pair<double, double> > >& s2, double sigma, int N){

  int num_diag_1 = s1.size();
  std::vector<std::vector<double> > matrix;
  std::vector<Sliced_Wasserstein> ss1;
  for(int i = 0; i < num_diag_1; i++){Sliced_Wasserstein sw1(s1[i], sigma, N); ss1.push_back(sw1);}

  if(s1 == s2){
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_1);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = i; j < num_diag_1; j++){  matrix[i][j] = ss1[i].compute_scalar_product(ss1[j]); matrix[j][i] = matrix[i][j];  }
    }
  } else {
    int num_diag_2 = s2.size();
    std::vector<Sliced_Wasserstein> ss2;
    for(int i = 0; i < num_diag_2; i++){Sliced_Wasserstein sw2(s2[i], sigma, N); ss2.push_back(sw2);}
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_2);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = 0; j < num_diag_2; j++)  matrix[i][j] = ss1[i].compute_scalar_product(ss2[j]);
    }
  }
  return matrix;
}

std::vector<std::vector<double> > sw_matrix(const std::vector<std::vector<std::pair<double, double> > >& s1, const std::vector<std::vector<std::pair<double, double> > >& s2, int N){

  int num_diag_1 = s1.size();
  std::vector<std::vector<double> > matrix;
  std::vector<Sliced_Wasserstein> ss1;
  for(int i = 0; i < num_diag_1; i++){Sliced_Wasserstein sw1(s1[i], 1.0, N); ss1.push_back(sw1);}

  if(s1 == s2){
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_1);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = i; j < num_diag_1; j++){  matrix[i][j] = ss1[i].compute_sliced_wasserstein_distance(ss1[j]); matrix[j][i] = matrix[i][j];  }
    }
  } else {
    int num_diag_2 = s2.size();
    std::vector<Sliced_Wasserstein> ss2;
    for(int i = 0; i < num_diag_2; i++){Sliced_Wasserstein sw2(s2[i], 1.0, N); ss2.push_back(sw2);}
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_2);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = 0; j < num_diag_2; j++)  matrix[i][j] = ss1[i].compute_sliced_wasserstein_distance(ss2[j]);
    }
  }
  return matrix;
}

std::vector<std::vector<double> > pwgk_matrix(const std::vector<std::vector<std::pair<double, double> > >& s1, const std::vector<std::vector<std::pair<double, double> > >& s2, double sigma, const Weight & weight){

  int num_diag_1 = s1.size();
  std::vector<std::vector<double> > matrix;
  std::vector<Persistence_weighted_gaussian> ss1;
  for(int i = 0; i < num_diag_1; i++){Persistence_weighted_gaussian pwg1(s1[i], sigma, weight); ss1.push_back(pwg1);}

  if(s1 == s2){
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_1);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = i; j < num_diag_1; j++){  matrix[i][j] = ss1[i].compute_scalar_product(ss1[j]); matrix[j][i] = matrix[i][j];  }
    }
  } else {
    int num_diag_2 = s2.size();
    std::vector<Persistence_weighted_gaussian> ss2;
    for(int i = 0; i < num_diag_2; i++){Persistence_weighted_gaussian pwg2(s2[i], sigma, weight); ss2.push_back(pwg2);}
    for(int i = 0; i < num_diag_1; i++)  matrix.emplace_back(num_diag_2);
    for(int i = 0; i < num_diag_1; i++){
      std::cout << 100.0*i/num_diag_1 << "\r" << std::flush;
      for(int j = 0; j < num_diag_2; j++)  matrix[i][j] = ss1[i].compute_scalar_product(ss2[j]);
    }
  }
  return matrix;
}
