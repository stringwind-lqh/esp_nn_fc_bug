// classifier.cc
#include "classifier.h"

#include <iostream>
#include <cmath>
#include <new>


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelout23.h" 
void Classifier::DebugDumpAllTensors() {
    

    TfLiteTensor* output = interpreter_->output_tensor(0);
    std::cout << "\n--- Output Tensor (New Endpoint) ---" << std::endl;
    std::cout << "Type: " << TfLiteTypeGetName(output->type) << std::endl;
    

    std::cout << "Shape: [";
    for (int d = 0; d < output->dims->size; ++d) {
        std::cout << output->dims->data[d] << (d < output->dims->size - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;


    size_t num_outputs = 1;
    for (int d = 0; d < output->dims->size; ++d) {
        num_outputs *= output->dims->data[d];
    }
    const int max_elements_to_print = 32;

    std::cout << "Data (first " << std::min((size_t)max_elements_to_print, num_outputs) << "): ";
    for (size_t i = 0; i < std::min((size_t)max_elements_to_print, num_outputs); ++i) {

        std::cout << static_cast<int>(output->data.int8[i]) << " ";
    }
    std::cout << std::endl;
}

void aligned_delete(void* p) {
    operator delete[](p, std::align_val_t(16));
}

Classifier::Classifier()
    : initialized_(false),
      model_(nullptr),
      resolver_(nullptr),
      interpreter_(nullptr),
      input_tensor_(nullptr),
      output_tensor_(nullptr),
      tensor_arena_(nullptr, aligned_delete) {}

Classifier::~Classifier() {

}

bool Classifier::Init() {
    if (initialized_) {
        return true;
    }

    
    tensor_arena_.reset(new (std::align_val_t(16)) uint8_t[kTensorArenaSize]);
    if (!tensor_arena_) {
        std::cerr << "Failed to allocate tensor arena." << std::endl;
        return false;
    }


    model_ = tflite::GetModel(classification_model_i8out23_tflite);
    if (!model_ || model_->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model schema version mismatch." << std::endl;
        return false;
    }


    resolver_ = std::make_unique<tflite::MicroMutableOpResolver<10>>();
    resolver_->AddConv2D();
    resolver_->AddMaxPool2D();
    resolver_->AddFullyConnected();
    resolver_->AddSoftmax();
    resolver_->AddReshape();
    resolver_->AddQuantize();
    resolver_->AddDequantize();


    interpreter_ = std::make_unique<tflite::MicroInterpreter>(
        model_, *resolver_, tensor_arena_.get(), kTensorArenaSize);


    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        return false;
    }


    input_tensor_ = interpreter_->input_tensor(0);
    output_tensor_ = interpreter_->output_tensor(0);

    if (!input_tensor_ || !output_tensor_ || input_tensor_->type != kTfLiteInt8 || output_tensor_->type != kTfLiteInt8) {
        std::cerr << "Failed to get valid INT8 input/output tensors." << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "Classifier initialized successfully." << std::endl;
    return true;
}

std::vector<float> Classifier::RunInference(const std::vector<float>& input_features) {
    if (!initialized_) {
        std::cerr << "Classifier is not initialized." << std::endl;
        return {}; 
    }


    float input_scale = input_tensor_->params.scale;
    int32_t input_zero_point = input_tensor_->params.zero_point;
    
    std::vector<int8_t> quantized_input_data;
    quantized_input_data.reserve(input_features.size());

    for (const auto& float_val : input_features) {
        int32_t quantized_val = static_cast<int32_t>(std::round(float_val / input_scale)) + input_zero_point;
        if (quantized_val < -128) quantized_val = -128;
        if (quantized_val > 127) quantized_val = 127;
        quantized_input_data.push_back(static_cast<int8_t>(quantized_val));
    }


    std::copy(quantized_input_data.begin(), quantized_input_data.end(), input_tensor_->data.int8);


    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter." << std::endl;
        return {};
    }


    return {};
}