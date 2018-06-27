#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <limits>
#include <utility>
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>

#include "geom_bottleneck/include/bottleneck.h"

using namespace std;
using PD = vector<pair<double,double> >;

double bottleneck_dist(PD & diag1, PD & diag2, double delta = 0.01){
    if(delta == 0.0)  return hera::bottleneckDistExact(diag1, diag2);
    else return hera::bottleneckDistApprox(diag1, diag2, delta);
}


