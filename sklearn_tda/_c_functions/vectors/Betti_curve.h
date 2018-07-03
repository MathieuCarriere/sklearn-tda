#include "../utils.h"

class Betti_curve {

  protected:
   std::vector<std::pair<double,double> > diagram;
   int res_x;
   double min_x, max_x;

  public:

   Betti_curve(const std::vector<std::pair<double,double> > & _diagram, double _min_x, double _max_x, int _res_x){diagram = _diagram; min_x = _min_x; max_x = _max_x; res_x = _res_x;}

   std::vector<int> vectorize() const {
     std::vector<int> bc(this->res_x, 0);
     int num_pts = this->diagram.size(); double step = (this->max_x - this->min_x)/(this->res_x - 1);

     for(int j = 0; j < num_pts; j++){
       double px = this->diagram[j].first; double py = this->diagram[j].second;
       int first  = std::min(this->res_x, std::max(0, (int) std::ceil((px-this->min_x)/step)));
       int last   = std::min(this->res_x, std::max(0, (int) std::ceil((py-this->min_x)/step)));
       if(first < this->res_x && last > 0){
         double x = this->min_x + first*step;
         for(int i = first; i < last; i++){  bc[i] += 1; x += step;  }
       }
     }
     return bc;
   }

}; // class Betti_curve
