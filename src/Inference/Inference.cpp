/**
 * @file Inference.cpp
 * @brief ONNX inference wrapper implementation
 */

#include <QiVision/Inference/Inference.h>
#include <QiVision/Core/Exception.h>

#ifdef QIVISION_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <numeric>

namespace Qi::Vision::Inference {

#ifdef QIVISION_HAS_ONNXRUNTIME

namespace {

Ort::Env& GetEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "QiVisionInference");
    return env;
}

size_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

} // namespace

class Model::Impl {
public:
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<const char*> outputNamePtrs;
    bool loaded = false;

    void Clear() {
        session.reset();
        inputNames.clear();
        outputNames.clear();
        inputNamePtrs.clear();
        outputNamePtrs.clear();
        loaded = false;
    }
};

#else

class Model::Impl {
public:
    bool loaded = false;
    void Clear() { loaded = false; }
};

#endif  // QIVISION_HAS_ONNXRUNTIME

Model::Model()
    : impl_(std::make_unique<Impl>()) {}

Model::~Model() = default;

Model::Model(Model&& other) noexcept = default;
Model& Model::operator=(Model&& other) noexcept = default;

bool Model::Load(const std::string& modelPath, const SessionOptions& opts) {
#ifdef QIVISION_HAS_ONNXRUNTIME
    if (modelPath.empty()) {
        throw InvalidArgumentException("Inference::Model::Load: modelPath is empty");
    }

    impl_->Clear();
    try {
        impl_->sessionOptions.SetIntraOpNumThreads(opts.numThreads);
        impl_->sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (opts.gpuIndex >= 0) {
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = opts.gpuIndex;
            impl_->sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }

#ifdef _WIN32
        std::wstring wPath(modelPath.begin(), modelPath.end());
        impl_->session = std::make_unique<Ort::Session>(GetEnv(), wPath.c_str(), impl_->sessionOptions);
#else
        impl_->session = std::make_unique<Ort::Session>(GetEnv(), modelPath.c_str(), impl_->sessionOptions);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputCount = impl_->session->GetInputCount();
        impl_->inputNames.reserve(inputCount);
        impl_->inputNamePtrs.reserve(inputCount);
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->inputNames.emplace_back(name.get());
        }
        for (const auto& n : impl_->inputNames) {
            impl_->inputNamePtrs.push_back(n.c_str());
        }

        size_t outputCount = impl_->session->GetOutputCount();
        impl_->outputNames.reserve(outputCount);
        impl_->outputNamePtrs.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->outputNames.emplace_back(name.get());
        }
        for (const auto& n : impl_->outputNames) {
            impl_->outputNamePtrs.push_back(n.c_str());
        }

        impl_->loaded = true;
        return true;
    } catch (const Ort::Exception&) {
        impl_->Clear();
        return false;
    }
#else
    (void)modelPath;
    (void)opts;
    throw UnsupportedException("Inference::Model::Load: ONNXRuntime not available");
#endif
}

bool Model::IsLoaded() const {
    return impl_->loaded;
}

void Model::Reset() {
    impl_->Clear();
}

std::vector<Tensor> Model::Run(const std::vector<Tensor>& inputs) {
#ifdef QIVISION_HAS_ONNXRUNTIME
    if (!impl_->loaded || !impl_->session) {
        throw InvalidArgumentException("Inference::Model::Run: model not loaded");
    }
    if (inputs.empty()) {
        throw InvalidArgumentException("Inference::Model::Run: inputs is empty");
    }

    std::vector<Ort::Value> ortInputs;
    ortInputs.reserve(inputs.size());

    std::vector<const char*> inputNames;
    inputNames.reserve(inputs.size());

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& t = inputs[i];
        const char* name = nullptr;
        if (!t.name.empty()) {
            name = t.name.c_str();
        } else if (i < impl_->inputNamePtrs.size()) {
            name = impl_->inputNamePtrs[i];
        } else {
            throw InvalidArgumentException("Inference::Model::Run: input name missing");
        }
        inputNames.push_back(name);

        size_t expected = NumElements(t.shape);
        if (expected == 0 || expected != t.data.size()) {
            throw InvalidArgumentException("Inference::Model::Run: input shape/data size mismatch");
        }

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float*>(t.data.data()),
            t.data.size(),
            t.shape.data(),
            t.shape.size());

        ortInputs.push_back(std::move(inputTensor));
    }

    auto ortOutputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                          inputNames.data(),
                                          ortInputs.data(),
                                          ortInputs.size(),
                                          impl_->outputNamePtrs.data(),
                                          impl_->outputNamePtrs.size());

    std::vector<Tensor> outputs;
    outputs.reserve(ortOutputs.size());

    for (size_t i = 0; i < ortOutputs.size(); ++i) {
        auto& out = ortOutputs[i];
        auto info = out.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        auto type = info.GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw UnsupportedException("Inference::Model::Run: only float outputs are supported");
        }

        size_t count = NumElements(shape);
        const float* data = out.GetTensorData<float>();

        Tensor t;
        t.name = (i < impl_->outputNames.size()) ? impl_->outputNames[i] : "";
        t.shape = std::move(shape);
        t.data.assign(data, data + count);
        outputs.push_back(std::move(t));
    }

    return outputs;
#else
    (void)inputs;
    throw UnsupportedException("Inference::Model::Run: ONNXRuntime not available");
#endif
}

std::vector<std::string> Model::InputNames() const {
#ifdef QIVISION_HAS_ONNXRUNTIME
    return impl_->inputNames;
#else
    return {};
#endif
}

std::vector<std::string> Model::OutputNames() const {
#ifdef QIVISION_HAS_ONNXRUNTIME
    return impl_->outputNames;
#else
    return {};
#endif
}

} // namespace Qi::Vision::Inference
