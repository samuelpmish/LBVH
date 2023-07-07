#include <nanobind/nanobind.h>

#include <iostream>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(pyLBVH, m) {
  m.def("find_intersections", [](){
    std::cout << "hello world" << std::endl;
  });
}
