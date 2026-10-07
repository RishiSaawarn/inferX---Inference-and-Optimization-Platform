#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// P2-17: C++ Preprocessing mock implementation with GIL handling 
py::array_t<float> preprocess_image(py::array_t<uint8_t> input_image) {
    auto buf = input_image.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 3");
    }

    auto height = buf.shape[0];
    auto width = buf.shape[1];
    auto channels = buf.shape[2];

    auto result = py::array_t<float>({channels, height, width});
    auto res_buf = result.request();

    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    // Release GIL for processing
    py::gil_scoped_release release;

    // Simple normalization and HWC -> CHW transposition
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                float val = ptr[h * width * channels + w * channels + c] / 255.0f;
                // ImageNet mean/std
                float mean = (c == 0) ? 0.485f : (c == 1) ? 0.456f : 0.406f;
                float std = (c == 0) ? 0.229f : (c == 1) ? 0.224f : 0.225f;
                
                res_ptr[c * height * width + h * width + w] = (val - mean) / std;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(inferx_preprocess, m) {
    m.doc() = "C++ Preprocessing for InferX";
    m.def("preprocess_image", &preprocess_image, "Preprocess image buffer to CHW normalized tensor");
}
