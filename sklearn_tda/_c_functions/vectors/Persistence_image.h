#include "../utils.h"

class Persistence_image {

 protected:
  std::vector<std::pair<double,double> > diagram;
  int res_x, res_y;
  double min_x, max_x, min_y, max_y;
  double sigma;
  Weight weight;

 public:

  Persistence_image(const std::vector<std::pair<double,double> > & _diagram, double _min_x, double _max_x, int _res_x, double _min_y, double _max_y, int _res_y, double _sigma, const Weight & _weight){
      diagram = _diagram; min_x = _min_x; max_x = _max_x; res_x = _res_x; min_y = _min_y; max_y = _max_y; res_y = _res_y, weight = _weight; sigma = _sigma;
  }

  std::vector<std::vector<double> > vectorize() const {
    std::vector<std::vector<double> > im; for(int i = 0; i < this->res_y; i++)  im.emplace_back();
    double step_x = (this->max_x - this->min_x)/(this->res_x - 1); double step_y = (this->max_y - this->min_y)/(this->res_y - 1);
    int num_pts = this->diagram.size();
    std::vector<double> weights(num_pts); for (int i = 0; i < num_pts; i++)  weights[i] = this->weight(this->diagram[i]);

    for(int i = 0; i < this->res_y; i++){
      double y = this->min_y + i*step_y;
      for(int j = 0; j < this->res_x; j++){
        double x = this->min_x + j*step_x;

        double pixel_value = 0;
        for(int k = 0; k < num_pts; k++){
          std::pair<double,double> point = this->diagram[k]; std::pair<double,double> grid_point(x,y);
          pixel_value += weights[k] * (1/(sigma*std::sqrt(2*pi))) * std::exp(  -((point.first-grid_point.first)*(point.first-grid_point.first)+(point.second-grid_point.second)*(point.second-grid_point.second)) / (2*sigma*sigma) );
        }
        im[i].push_back(pixel_value);

      }
    }

    return im;

  }

}; // class Persistence_image
