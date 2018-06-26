// standard include
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <utility>
#include <functional>
#include <random>
#include "utils.h"

class Persistence_weighted_gaussian{

 protected:
    std::vector<std::pair<double,double> > diagram;
    Weight weight;
    Kernel kernel;
    std::vector<double> w;

 public:

  Persistence_weighted_gaussian(const std::vector<std::pair<double,double> > & _diagram, const Kernel & _kernel = rbf_kernel(1.0), const Weight & _weight = linear_weight){
      diagram = _diagram; weight = _weight; kernel = _kernel;
      for(size_t i = 0; i < this->diagram.size(); i++)  this->w.push_back(this->weight(this->diagram[i]));
  }

  double compute_scalar_product(const Persistence_weighted_gaussian & second) const {
    std::vector<std::pair<double,double> > diagram1 = this->diagram; std::vector<std::pair<double,double> > diagram2 = second.diagram;
    int num_pts1 = diagram1.size(); int num_pts2 = diagram2.size(); double k = 0;
    for(int i = 0; i < num_pts1; i++)
      for(int j = 0; j < num_pts2; j++)
        k += this->w[i] * second.w[j] * this->kernel(diagram1[i], diagram2[j]);
    return k;
  }

}; // class Persistence_weighted_gaussian
