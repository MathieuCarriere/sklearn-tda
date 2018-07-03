#include "../utils.h"

class Sliced_Wasserstein {

 protected:
    std::vector<std::pair<double,double> > diagram;
    int approx;
    double sigma;
    std::vector<std::vector<double> > projections, projections_diagonal;

 public:

  void build_rep(){

    if(approx > 0){

      double step = pi/this->approx;
      int n = diagram.size();

      for (int i = 0; i < this->approx; i++){
        std::vector<double> l,l_diag;
        for (int j = 0; j < n; j++){

          double px = diagram[j].first; double py = diagram[j].second;
          double proj_diag = (px+py)/2;

          l.push_back         (   px         * cos(-pi/2+i*step) + py         * sin(-pi/2+i*step)   );
          l_diag.push_back    (   proj_diag  * cos(-pi/2+i*step) + proj_diag  * sin(-pi/2+i*step)   );
        }

        std::sort(l.begin(), l.end()); std::sort(l_diag.begin(), l_diag.end());
        projections.push_back(l); projections_diagonal.push_back(l_diag);

      }

    }

  }

  Sliced_Wasserstein(const std::vector<std::pair<double, double> > & _diagram, double _sigma, int _approx){diagram = _diagram; approx = _approx; sigma = _sigma; build_rep();}

  double compute_sliced_wasserstein_distance(const Sliced_Wasserstein & second) const {

    std::vector<std::pair<double,double> > diagram1 = this->diagram; std::vector<std::pair<double,double> > diagram2 = second.diagram; double sw = 0;

    double step = pi/this->approx;
    for (int i = 0; i < this->approx; i++){

      std::vector<double> v1; std::vector<double> l1 = this->projections[i]; std::vector<double> l1bis = second.projections_diagonal[i]; std::merge(l1.begin(), l1.end(), l1bis.begin(), l1bis.end(), std::back_inserter(v1));
      std::vector<double> v2; std::vector<double> l2 = second.projections[i]; std::vector<double> l2bis = this->projections_diagonal[i]; std::merge(l2.begin(), l2.end(), l2bis.begin(), l2bis.end(), std::back_inserter(v2));
      int n = v1.size(); double f = 0;
      for (int j = 0; j < n; j++)  f += std::abs(v1[j] - v2[j]);
      sw += f*step;

    }

    return sw/pi;
  }

  double compute_scalar_product(const Sliced_Wasserstein & second) const {
    return std::exp(-compute_sliced_wasserstein_distance(second)/(2*this->sigma*this->sigma));
  }

}; // class Sliced_Wasserstein
