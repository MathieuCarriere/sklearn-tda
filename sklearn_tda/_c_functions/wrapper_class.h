#include "utils.h"
#include "wrapper.h"

class KernelWrapper {
public:

    KernelWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    KernelWrapper(const KernelWrapper& rhs): KernelWrapper(rhs.held) {}

    KernelWrapper(KernelWrapper&& rhs): held(rhs.held) {rhs.held = 0;}

    KernelWrapper(): KernelWrapper(nullptr) {}

    ~KernelWrapper() {Py_XDECREF(held);}

    KernelWrapper& operator=(const KernelWrapper& rhs) {
        KernelWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    KernelWrapper& operator=(KernelWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    double operator()(std::pair<double,double> p, std::pair<double, double> q) {
        if (held) return call_ker(held,p,q); else return 0;
    }

private:
    PyObject* held;
};

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
