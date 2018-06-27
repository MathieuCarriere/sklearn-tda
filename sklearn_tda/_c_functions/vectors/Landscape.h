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

class Landscape {

  protected:
   std::vector<std::pair<double,double> > diagram;
   int res_x, nb_ls;
   double min_x, max_x;

  public:

   Landscape(const std::vector<std::pair<double,double> > & _diagram, int _nb_ls = 5, double _min_x = 0.0, double _max_x = 1.0, int _res_x = 10){diagram = _diagram; nb_ls = _nb_ls; min_x = _min_x; max_x = _max_x; res_x = _res_x;}

   std::vector<std::vector<double> > vectorize() const {
     std::vector<std::vector<double> >  ls; for(int i = 0; i < nb_ls; i++)  ls.emplace_back();
     int num_pts = diagram.size(); double step = (max_x - min_x)/(res_x - 1);

     std::vector<std::vector<double> > ls_t; for(int i = 0; i < res_x; i++)  ls_t.emplace_back();
     for(int j = 0; j < num_pts; j++){
       double px = diagram[j].first; double py = diagram[j].second; double mid = (px+py)/2;
       int first  = std::min(this->res_x, std::max(0, (int) std::ceil((px-min_x)/step)));
       int middle = std::min(this->res_x, std::max(0, (int) std::ceil((mid-min_x)/step)));
       int last   = std::min(this->res_x, std::max(0, (int) std::ceil((py-min_x)/step)));
       if(first < this->res_x && last > 0){
         double x = min_x + first*step;
         for(int i = first; i < middle; i++){  double value = std::sqrt(2)*(x-px); ls_t[i].push_back(value); x += step;  }
         for(int i = middle; i < last; i++){   double value = std::sqrt(2)*(py-x); ls_t[i].push_back(value); x += step;  }
       }
     }

     for(int i = 0; i < res_x; i++){
       std::sort(ls_t[i].begin(), ls_t[i].end(), [](const double & a, const double & b){return a > b;});
       int nb_events_i = ls_t[i].size();
       for (int j = 0; j < nb_ls; j++){  if(j < nb_events_i)  ls[j].push_back(ls_t[i][j]);  else  ls[j].push_back(0);  }
     }

     return ls;
   }

}; // class Landscape
