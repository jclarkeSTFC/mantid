// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2021 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once
#include <string>
#include <variant>
#include <vector>

#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/variant.hpp>

#include <exception>

namespace Mantid::PythonInterface {

struct PyNativeTypeExtractor {
  using PythonOutputT =
      boost::make_recursive_variant<bool, long, double, std::string, std::vector<boost::recursive_variant_>>::type;

  static PythonOutputT convert(const boost::python::object &obj) {
    using namespace boost::python;
    PyObject *rawptr = obj.ptr();
    PythonOutputT out;

    if (PyList_Check(rawptr)) {
      out = handleList(obj);
    } else if (PyBool_Check(rawptr)) {
      out = extract<bool>(obj);
    } else if (PyFloat_Check(rawptr)) {
      out = extract<double>(obj);
    } else if (PyLong_Check(rawptr)) {
      out = extract<long>(obj);
    } else if (PyUnicode_Check(rawptr)) {
      out = extract<std::string>(obj);
    } else {
      throw std::invalid_argument("Unrecognised Python type");
    }
    return out;
  }

private:
  static PythonOutputT handleList(const boost::python::object &obj) {
    auto rawptr = obj.ptr();
    auto n = PyList_Size(rawptr);
    auto vec = std::vector<PythonOutputT>();
    vec.reserve(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
      vec.emplace_back(convert(obj[i]));
    }
    return vec;
  }
};

} // namespace Mantid::PythonInterface
