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
#include "utils.h"

class Persistence_image {

 protected:
  std::vector<std::pair<double,double> > diagram;
  int res_x, res_y;
  double min_x, max_x, min_y, max_y;
  Kernel kernel;
  Weight weight;

 public:

  Persistence_image(const std::vector<std::pair<double,double> > & _diagram, double _min_x = 0.0, double _max_x = 1.0, int _res_x = 10, double _min_y = 0.0, double _max_y = 1.0, int _res_y = 10, const Kernel & _kernel = rbf_kernel(1.0), const Weight & _weight = linear_weight){
      diagram = _diagram; min_x = _min_x; max_x = _max_x; res_x = _res_x; min_y = _min_y; max_y = _max_y; res_y = _res_y, weight = _weight; kernel = _kernel;
  }

  std::vector<std::vector<double> > vectorize() const {
    std::vector<std::vector<double> > im; for(int i = 0; i < res_y; i++)  im.emplace_back();
    double step_x = (max_x - min_x)/(res_x - 1); double step_y = (max_y - min_y)/(res_y - 1);

    int num_pts = diagram.size();

    for(int i = 0; i < res_y; i++){
      double y = min_y + i*step_y;
      for(int j = 0; j < res_x; j++){
        double x = min_x + j*step_x;

        double pixel_value = 0;
        for(int k = 0; k < num_pts; k++){
          double px = diagram[k].first; double py = diagram[k].second;
          std::pair<double,double> point(px,py); std::pair<double,double> grid_point(x,y);
          pixel_value += this->weight(point) * this->kernel(point, grid_point);
        }
        im[i].push_back(pixel_value);

      }
    }

    return im;

  }

}; // class Persistence_image
