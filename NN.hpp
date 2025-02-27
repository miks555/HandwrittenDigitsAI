#pragma once
#include <iostream>
#include <vector>

#define DEBUG_MODE 1 //Enable logging

class NN {
  public:
  NN();
  ~NN();
  bool initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounts);
  bool train(std::string fileNameTrainData, std::string fileNameTrainLabels, double learningRate);
  bool recognize(std::string fileNameRecognize, unsigned int& recognizedIndex, double& confidence);
  bool randomizeData();
  private:
  std::string fileNameData;
  size_t dataSize; //In elements
  std::vector<std::vector<double>> weightLayers; //weightLayers[layerIndex][weightIndex], neuronLayers size -1 (input layer)
  std::vector<std::vector<double>> biasLayers;
  std::vector<std::vector<double>> neuronLayers;
  std::vector<std::vector<double>> weightLayersGradient;
  std::vector<std::vector<double>> biasLayersGradient;
  std::vector<std::vector<double>> neuronLayersSums;
  bool parseData();
  bool saveData();
  bool checkFileExistence(std::string fileName);
  bool propagateForward();
  bool propagateBackward(unsigned int label, double learningRate);
  double MeanSquaredErrorDerivative(double num, double target);
  double sigmoid(double num);
  double sigmoidDerivative(double num);
  bool clearConsole();
};