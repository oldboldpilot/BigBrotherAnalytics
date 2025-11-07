#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(bigbrother_py, m) {
    m.doc() = "BigBrotherAnalytics Python Bindings";
    // Bindings to be implemented
}
