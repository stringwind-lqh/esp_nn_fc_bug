// classifier.h
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <memory>
#include <cstdint>

namespace tflite {
class Model;
template <unsigned int tOpCount>
class MicroMutableOpResolver;
class MicroInterpreter;
} 

struct TfLiteTensor;

class Classifier {
public:
    Classifier();
    ~Classifier(); 


    bool Init();

    std::vector<float> RunInference(const std::vector<float>& input_features);

    void DebugDumpAllTensors();

private:

    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;

    bool initialized_;

    const tflite::Model* model_;
    std::unique_ptr<tflite::MicroMutableOpResolver<10>> resolver_;
    std::unique_ptr<tflite::MicroInterpreter> interpreter_;

    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_tensor_;


    static constexpr size_t kTensorArenaSize = 300 * 1024;

    std::unique_ptr<uint8_t[], void(*)(void*)> tensor_arena_;
};

#endif // CLASSIFIER_H
