#include <iostream>
#include "NN.hpp"

constexpr char NN_DATA_FILENAME[] = "nn_data";
constexpr char DATA_TO_RECOGNIZE_FILENAME[] = "image_to_recognize";
constexpr char TRAINING_DATA_FILENAME[] = "training_images";
constexpr char TRAINING_LABELS_FILENAME[] = "training_labels";
constexpr unsigned int LAYER_0_NEURON_AMOUNT = 784;
constexpr unsigned int LAYER_1_NEURON_AMOUNT = 128;
constexpr unsigned int LAYER_2_NEURON_AMOUNT = 10;
constexpr double LEARNING_RATE = 0.1;

int main() {
    NN* network_0 = new NN();
    if (network_0->initiate(NN_DATA_FILENAME, {
        LAYER_0_NEURON_AMOUNT,
        LAYER_1_NEURON_AMOUNT,
        LAYER_2_NEURON_AMOUNT
    })) {
        delete network_0;
        return 1;
    }
    std::cout << "To recognize enter 0, to train the network enter 1\n";
    bool selection;
    std::cin >> selection;
     if (selection == 0) {
        unsigned int recognized = 0;
        double confidence = 0.0;
        if (network_0->recognize(DATA_TO_RECOGNIZE_FILENAME, recognized, confidence)) {
            std::cerr << "Recognition failed\n";
            delete network_0; 
            return 1;
        }
        std::cout << "Recognized: " << recognized << std::endl;
        std::cout << "Confidence: " << confidence * 100.0 << "%" << std::endl;
    }  else {
        if (network_0->train(TRAINING_DATA_FILENAME, TRAINING_LABELS_FILENAME,LEARNING_RATE)) {
            delete network_0; 
            return 1;
        }
    }
    delete network_0;
    return 0;
}