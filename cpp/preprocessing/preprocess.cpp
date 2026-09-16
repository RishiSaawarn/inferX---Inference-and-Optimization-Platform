#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

// A simple preprocessing function to demonstrate C++ integration
py::array_t<float> preprocess_image(py::array_t<uint8_t> input_image) {
    py::buffer_info buf = input_image.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 3");
    }
    
    auto result = py::array_t<float>(buf.size);
    py::buffer_info res_buf = result.request();
    
    uint8_t* ptr1 = static_cast<uint8_t*>(buf.ptr);
    float* ptr2 = static_cast<float*>(res_buf.ptr);
    
    // Normalize (x / 255.0)
    for (size_t i = 0; i < buf.size; i++) {
        ptr2[i] = static_cast<float>(ptr1[i]) / 255.0f;
    }
    
    return result;
}

PYBIND11_MODULE(inferx_preprocess, m) {
    m.doc() = "C++ Preprocessing Module for InferX";
    m.def("preprocess_image", &preprocess_image, "A function that normalizes an image");
}
