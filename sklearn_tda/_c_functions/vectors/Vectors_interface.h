#include "Landscape.h"
#include "Persistence_image.h"
#include "Silhouette.h"
#include "Betti_curve.h"
#include "../utils.h"

std::vector<std::vector<double> > compute_ls(const std::vector<std::pair<double, double> >& diag, int nb_ls, double min_x, double max_x, int res_x) {
  Landscape L(diag, nb_ls, min_x, max_x, res_x);
  return L.vectorize();
}

std::vector<std::vector<double> > compute_pim(const std::vector<std::pair<double, double> >& diag, double min_x, double max_x, int res_x, double min_y, double max_y, int res_y, double sigma, const Weight& weight) {
  Persistence_image P(diag, min_x, max_x, res_x, min_y, max_y, res_y, sigma, weight);
  return P.vectorize();
}

std::vector<double> compute_sh(const std::vector<std::pair<double, double> >& diag, double min_x, double max_x, int res_x, const Weight& weight) {
  Silhouette S(diag, min_x, max_x, res_x, weight);
  return S.vectorize();
}

std::vector<int> compute_bc(const std::vector<std::pair<double, double> >& diag, double min_x, double max_x, int res_x) {
  Betti_curve B(diag, min_x, max_x, res_x);
  return B.vectorize();
}
