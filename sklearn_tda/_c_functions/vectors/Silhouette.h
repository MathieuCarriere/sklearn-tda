#include "../utils.h"

class Silhouette {

  protected:
   std::vector<std::pair<double,double> > diagram;
   int res_x;
   double min_x, max_x;
   Weight weight;

  public:

   Silhouette(const std::vector<std::pair<double,double> > & _diagram, double _min_x, double _max_x, int _res_x, const Weight & _weight){diagram = _diagram; weight = _weight; min_x = _min_x; max_x = _max_x; res_x = _res_x;}

   std::vector<double> vectorize() const {
     std::vector<double> sh(this->res_x);
     int num_pts = this->diagram.size(); double step = (this->max_x - this->min_x)/(this->res_x - 1);
     std::vector<double> weights(num_pts); for (int i = 0; i < num_pts; i++)  weights[i] = this->weight(this->diagram[i]);

     for(int j = 0; j < num_pts; j++){
       double px = this->diagram[j].first; double py = this->diagram[j].second; double mid = (px+py)/2;
       int first  = std::min(this->res_x, std::max(0, (int) std::ceil((px-this->min_x)/step)));
       int middle = std::min(this->res_x, std::max(0, (int) std::ceil((mid-this->min_x)/step)));
       int last   = std::min(this->res_x, std::max(0, (int) std::ceil((py-this->min_x)/step)));
       if(first < this->res_x && last > 0){
         double x = this->min_x + first*step;
         for(int i = first; i < middle; i++){  double value = std::sqrt(2)*(x-px)*weights[j]; sh[i] += value; x += step;  }
         for(int i = middle; i < last; i++){   double value = std::sqrt(2)*(py-x)*weights[j]; sh[i] += value; x += step;  }
       }
     }
     return sh;
   }

}; // class Silhouette
