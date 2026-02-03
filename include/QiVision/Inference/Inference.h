#pragma once

/**
 * @file Inference.h
 * @brief Lightweight ONNX inference wrapper for QiVision
 */

#include <QiVision/Core/Export.h>
#include <QiVision/Core/Exception.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Qi::Vision::Inference {

struct QIVISION_API Tensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct QIVISION_API SessionOptions {
    int numThreads = 4;
    int gpuIndex = -1;        // -1 = CPU
    bool enableFP16 = false;  // reserved
};

class QIVISION_API Model {
public:
    Model();
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;

    bool Load(const std::string& modelPath, const SessionOptions& opts = {});
    bool IsLoaded() const;
    void Reset();

    std::vector<Tensor> Run(const std::vector<Tensor>& inputs);
    std::vector<std::string> InputNames() const;
    std::vector<std::string> OutputNames() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Qi::Vision::Inference
