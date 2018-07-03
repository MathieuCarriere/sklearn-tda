#include "../utils.h"

class Persistence_weighted_gaussian{

 protected:
    std::vector<std::pair<double,double> > diagram;
    Weight weight;
    double sigma;
    std::vector<double> w;

 public:

  Persistence_weighted_gaussian(const std::vector<std::pair<double,double> > & _diagram, double _sigma, const Weight& _weight){
      diagram = _diagram; weight = _weight; sigma = _sigma;
      for(size_t i = 0; i < this->diagram.size(); i++)  this->w.push_back(this->weight(this->diagram[i]));
  }

  double compute_scalar_product(const Persistence_weighted_gaussian & second) const {
    std::vector<std::pair<double,double> > diagram1 = this->diagram; std::vector<std::pair<double,double> > diagram2 = second.diagram;
    int num_pts1 = diagram1.size(); int num_pts2 = diagram2.size(); double k = 0;
    for(int i = 0; i < num_pts1; i++)
      for(int j = 0; j < num_pts2; j++)
        k += this->w[i] * second.w[j] * (1/(this->sigma*std::sqrt(2*pi))) * std::exp(  -((diagram1[i].first-diagram2[j].first)*(diagram1[i].first-diagram2[j].first)+(diagram1[i].second-diagram2[j].second)*(diagram1[i].second-diagram2[j].second)) / (2*this->sigma*this->sigma)  );
    return k;
  }

}; // class Persistence_weighted_gaussian
