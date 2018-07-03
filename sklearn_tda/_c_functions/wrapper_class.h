#include "utils.h"
#include "wrapper.h"

class WeightWrapper {
public:

    WeightWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    WeightWrapper(const WeightWrapper& rhs): WeightWrapper(rhs.held) {}

    WeightWrapper(WeightWrapper&& rhs): held(rhs.held) {rhs.held = 0;}

    WeightWrapper(): WeightWrapper(nullptr) {}

    ~WeightWrapper() {Py_XDECREF(held);}

    WeightWrapper& operator=(const WeightWrapper& rhs) {
        WeightWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    WeightWrapper& operator=(WeightWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    double operator()(std::pair<double,double> p) {
        if (held) return call_wei(held,p); else return 0;
    }

private:
    PyObject* held;
};
