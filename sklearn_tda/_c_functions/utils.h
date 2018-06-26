#ifndef UTILS_H
#define UTILS_H

#include <utility>
#include <string>
#include <cmath>

using Weight = std::function<double (std::pair<double,double>) >;
using Kernel = std::function<double (std::pair<double,double>, std::pair<double,double>) >;
double pi = 3.1415;

double pss_weight(std::pair<double,double> p)     {if(p.second > p.first)  return 1; else return -1;}
double linear_weight(std::pair<double,double> p)  {return std::abs(p.second);}
double const_weight(std::pair<double,double> p)   {return 1;}
std::function<double (std::pair<double,double>) > arctan_weight(double C, double alpha)  {return [=](std::pair<double,double> p){return C * atan(std::pow(std::abs(p.second), alpha));};}

std::function<double (std::pair<double,double>, std::pair<double,double>) > rbf_kernel(double sigma)  {return [=](std::pair<double,double> p, std::pair<double, double> q){return std::exp( -((p.first-q.first)*(p.first-q.first) + (p.second-q.second)*(p.second-q.second)) / (2*sigma*sigma) ) / (sigma*std::sqrt(2*pi));};}
std::function<double (std::pair<double,double>, std::pair<double,double>) > poly_kernel(double c, double d)  {return [=](std::pair<double,double> p, std::pair<double, double> q){return std::pow(p.first*q.first + p.second*q.second + c, d);};}

#endif
