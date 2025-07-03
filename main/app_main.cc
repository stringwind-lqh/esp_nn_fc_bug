#include <iostream>
#include <vector>
#include <chrono>

#include "classifier.h"
#include "one_features.h"

void run_inference_test() {

    Classifier classifier;


    if (!classifier.Init()) {
        std::cerr << "Failed to initialize the classifier." << std::endl;
        return;
    }

    std::vector<float> input_features(
        g_one_feature_data, g_one_feature_data + g_one_feature_data_len
    );

    classifier.RunInference(input_features);

    std::cout << "\n>>>>>> DUMPING TENSOR FROM SLICED MODEL <<<<<<" << std::endl;
    classifier.DebugDumpAllTensors();
    std::cout << ">>>>>> DUMPING COMPLETE <<<<<<\n" << std::endl;
  }

extern "C" void app_main() {
    run_inference_test();
}
