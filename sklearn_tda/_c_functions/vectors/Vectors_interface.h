#include "Landscape.h"
#include "Persistence_image.h"
#include "../utils.h"

std::vector<std::vector<double> > compute_ls(const std::vector<std::pair<double, double> >& diag, int nb_ls, double min_x, double max_x, int res_x) {
  Landscape L(diag, nb_ls, min_x, max_x, res_x);
  return L.vectorize();
}

std::vector<std::vector<double> > compute_pim(const std::vector<std::pair<double, double> >& diag, double min_x, double max_x, int res_x, double min_y, double max_y, int res_y, const Kernel& kernel, const Weight& weight) {
  Persistence_image P(diag, min_x, max_x, res_x, min_y, max_y, res_y, kernel, weight);
  return P.vectorize();
}
