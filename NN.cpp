#include <fstream>
#include <iomanip>
#include <cmath>
#include "NN.hpp"

NN::NN() {
  #if DEBUG_MODE
  std::cout << "Neural network instance created\n";
  #endif
}

NN::~NN() {
  #if DEBUG_MODE
  std::cout << "Neural network instance deleted\n";
  #endif
}

double NN::sigmoid(double num) {
  return 1.0 / (1.0 + std::exp(-num));
}

double NN::sigmoidDerivative(double num) {
  return sigmoid(num) * (1.0 - sigmoid(num));
}

double NN::MeanSquaredErrorDerivative(double num, double target) {
  return 2.0 * (num - target);
}

bool NN::initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounts) {
  this -> fileNameData = fileNameData;
  this -> dataSize = 0;
  neuronLayers.resize(layerNeuronAmounts.size());
  neuronLayersSums.resize(layerNeuronAmounts.size());
  for (size_t i = 0; i < neuronLayers.size(); i++) {
    neuronLayers[i].resize(layerNeuronAmounts[i]);
    neuronLayersSums[i].resize(layerNeuronAmounts[i]);
  }
  #if DEBUG_MODE
  std::cout << "Neural network instance initiated with layout:\n";
  for (size_t i = 0; i < neuronLayers.size(); i++) {
    std::cout << "Layer " << i << ", " << neuronLayers[i].size() << " neurons" << std::endl;
  }
  #endif
  weightLayers.resize(layerNeuronAmounts.size() - 1); //Number of layers without input layer
  biasLayers.resize(layerNeuronAmounts.size() - 1);
  weightLayersGradient.resize(layerNeuronAmounts.size() - 1);
  biasLayersGradient.resize(layerNeuronAmounts.size() - 1);
  for (size_t i = 0; i < layerNeuronAmounts.size() - 1; i++) {
    weightLayers[i].resize(layerNeuronAmounts[i] * layerNeuronAmounts[i + 1]);
    biasLayers[i].resize(layerNeuronAmounts[i + 1]);
    weightLayersGradient[i].resize(layerNeuronAmounts[i] * layerNeuronAmounts[i + 1]);
    biasLayersGradient[i].resize(layerNeuronAmounts[i + 1]);
  }
  for (size_t i = 0; i < weightLayers.size(); i++) {
    dataSize = dataSize + weightLayers[i].size() + biasLayers[i].size();
  }
  #if DEBUG_MODE
  std::cout << "Data size in elements: " << dataSize << std::endl;
  std::cout << "Data size in bytes: " << dataSize * sizeof(double) << std::endl;
  #endif
  if (!checkFileExistence(fileNameData)) {
    #if DEBUG_MODE
    std::cerr << "No neural network data found, randomizing...\n";
    #endif
    if (randomizeData()) {
      #if DEBUG_MODE
      std::cerr << "Data randomization error\n";
      #endif
      return 1;
    }
  } else {
    #if DEBUG_MODE
    std::cout << "Parsing neural network data...\n";
    #endif
    if (parseData()) {
      #if DEBUG_MODE
      std::cerr << "Data parsing error\n";
      #endif
      return 1;
    }
  }
  #if DEBUG_MODE
  std::cout << "Network initiated\n";
  #endif
  return 0;
}

bool NN::checkFileExistence(std::string fileName) {
  std::ifstream fileCheckStream(fileName);
  return fileCheckStream.good();
}

bool NN::randomizeData() {
  srand(static_cast < unsigned int > (time(NULL)));
  for (size_t i = 0; i < weightLayers.size(); i++) {
    size_t n_in = neuronLayers[i].size();
    size_t n_out = neuronLayers[i + 1].size();
    double limit = sqrt(6.0 / (n_in + n_out));
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      weightLayers[i][j] = (static_cast < double > (std::rand()) / RAND_MAX) * 2 * limit - limit;
    }
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      biasLayers[i][j] = 0.0;
    }
  }
  saveData();
  return 0;
}

bool NN::parseData() {
  std::ifstream parseDataStream(fileNameData, std::ios::binary);
  if (!parseDataStream) {
    return 1;
  }
  std::vector < double > parsedData(dataSize);
  parseDataStream.read(reinterpret_cast < char * > (parsedData.data()), dataSize * sizeof(double));
  parseDataStream.close();
  size_t parsedDataIndex = 0;
  for (size_t i = 0; i < weightLayers.size(); i++) {
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      if (parsedDataIndex >= parsedData.size()) {
        return 1;
      }
      biasLayers[i][j] = parsedData[parsedDataIndex];
      parsedDataIndex++;
    }
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      if (parsedDataIndex >= parsedData.size()) {
        return 1;
      }
      weightLayers[i][j] = parsedData[parsedDataIndex];
      parsedDataIndex++;
    }
  }
  return 0;
}

bool NN::saveData() {
  std::ofstream saveDataStream(fileNameData, std::ios::binary);
  if (!saveDataStream) {
    return 1;
  }
  std::vector < double > saveData;
  for (size_t i = 0; i < weightLayers.size(); i++) {
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      saveData.push_back(biasLayers[i][j]);
    }
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      saveData.push_back(weightLayers[i][j]);
    }
  }
  saveDataStream.write(reinterpret_cast <
    const char * > (saveData.data()), saveData.size() * sizeof(double));
  saveDataStream.close();
  if (!checkFileExistence(fileNameData)) {
    return 1;
  }
  return 0;
}

bool NN::propagateForward() {
  for (size_t i = 1; i < neuronLayers.size(); i++) //For all layers except input layer 0
  {
    for (size_t j = 0; j < neuronLayers[i].size(); j++) //For all neurons
    {
      double activationSum = 0;
      for (size_t k = 0; k < neuronLayers[i - 1].size(); k++) //For all neurons connected to neuron
      {
        activationSum = activationSum + neuronLayers[i - 1][k] * weightLayers[i - 1][k + j * neuronLayers[i - 1].size()];
      }
      neuronLayers[i][j] = sigmoid(biasLayers[i - 1][j] + activationSum);
      neuronLayersSums[i][j] = biasLayers[i - 1][j] + activationSum;
    }
  }
  return 0;
}

bool NN::clearConsole() {
  #ifdef _WIN32
  system("cls"); // Windows
  #else
  system("clear"); // Linux/macOS
  #endif
  return 0;
}

bool NN::recognize(std::string fileNameRecognize, unsigned int& maxIndex, double& confidence) {
  std::ifstream fileRecognizeStream(fileNameRecognize);
  if (!fileRecognizeStream) {
    #if DEBUG_MODE
    std::cerr << fileNameRecognize << " not found\n";
    #endif
    return 1;
  }
  unsigned char dataChar;
  neuronLayers[0].resize(0); //To use push_back
  while (fileRecognizeStream.read(reinterpret_cast < char * > ( & dataChar), sizeof(dataChar))) {
    neuronLayers[0].push_back((static_cast < unsigned int > (dataChar)) / 255.0); //Normalize 0 - 255 to 0.0 - 1.0
  }
  fileRecognizeStream.close();
  propagateForward();
  confidence = neuronLayers[neuronLayers.size() - 1][0]; //State solution
  maxIndex = 0;
  for (int i = 1; i < neuronLayers[neuronLayers.size() - 1].size(); ++i) {
    if (neuronLayers[neuronLayers.size() - 1][i] > confidence) {
      confidence = neuronLayers[neuronLayers.size() - 1][i];
      maxIndex = i;
    }
  }
  return 0;
}

bool NN::train(std::string fileNameTrainData, std::string fileNameTrainLabels, double learningRate) {
  std::ifstream trainLabelsStream(fileNameTrainLabels);
  if (!trainLabelsStream) {
    #if DEBUG_MODE
    std::cerr << fileNameTrainLabels << " not found\n";
    #endif
    return 1;
  }
  std::ifstream trainDataStream(fileNameTrainData);
  if (!trainDataStream) {
    #if DEBUG_MODE
    std::cerr << fileNameTrainLabels << " not found\n";
    #endif
    return 1;
  }
  trainLabelsStream.seekg(0, std::ios::end);
  size_t trainSetSize = trainLabelsStream.tellg(); //Input elements
  trainLabelsStream.seekg(0, std::ios::beg);
  for (size_t i = 0; i < trainSetSize; i++) { //For all training elements
    unsigned int label = static_cast < unsigned int > (static_cast < unsigned char > (trainLabelsStream.get())); //Read current label
    for (size_t j = 0; j < neuronLayers[0].size(); j++) { //Read current data
      neuronLayers[0][j] = (static_cast < unsigned int > (static_cast < unsigned char > (trainDataStream.get()))) / 255.0;
    }
    propagateForward();
    propagateBackward(label, learningRate);
    #if DEBUG_MODE
    if (i % (trainSetSize / 1000) == 0) //Progress info
    {
      clearConsole();
      std::cout << std::fixed << std::setprecision(1) << "Training progress: " << ((i * 100.0) / trainSetSize) << "%\n";
    }
    #endif
  }
  trainLabelsStream.close();
  trainDataStream.close();
  saveData();
  clearConsole();
  #if DEBUG_MODE
  std::cout << "Training completed\n";
  #endif
  return 0;
}

bool NN::propagateBackward(unsigned int label, double learningRate) {
  for (size_t i = neuronLayers.size() - 1; i >= 1; i--) { // For every layer
    if (i == neuronLayers.size() - 1) { // Output layer
      for (size_t j = 0; j < neuronLayers[i].size(); j++) { // For every neuron
        double error;
        if (j == label) {
          error = MeanSquaredErrorDerivative(neuronLayers[i][j], 1.0) * sigmoidDerivative(neuronLayersSums[i][j]);
        } else {
          error = MeanSquaredErrorDerivative(neuronLayers[i][j], 0.0) * sigmoidDerivative(neuronLayersSums[i][j]);
        }
        neuronLayers[i][j] = error; //Propagating error in default structure
        for (size_t k = 0; k < neuronLayers[i - 1].size(); k++) {
          weightLayersGradient[i - 1][k + j * neuronLayers[i - 1].size()] = error * neuronLayers[i - 1][k];
        }
        biasLayersGradient[i - 1][j] = error;
      }
    } else { // Hidden layers
      for (size_t j = 0; j < neuronLayers[i].size(); j++) { // For every neuron
        double backActivation = 0.0; // Calculate the error of a neuron in this layer based on the weights and the error in the next layer (backpropagation)
        for (size_t k = 0; k < neuronLayers[i + 1].size(); k++) {
          backActivation = backActivation + neuronLayers[i + 1][k] * weightLayers[i][k + j + k * neuronLayers[i].size()];
        }
        double error = sigmoidDerivative(neuronLayersSums[i][j]) * backActivation;
        neuronLayers[i][j] = error;
        for (size_t k = 0; k < neuronLayers[i - 1].size(); k++) {
          weightLayersGradient[i - 1][k + j * neuronLayers[i - 1].size()] = error * neuronLayers[i - 1][k]; // The cost-weight gradient is the error * the activation of the previous layer, which is quite inconvenient here due to the weight indexing, which favors forward propagation
        }
        biasLayersGradient[i - 1][j] = error;
      }
    }
  }
  for (size_t i = 0; i < biasLayers.size(); i++) {//Update weights and biases
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      biasLayers[i][j] = biasLayers[i][j] - learningRate * biasLayersGradient[i][j];
    }
  }
  for (size_t i = 0; i < weightLayers.size(); i++) {
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      weightLayers[i][j] = weightLayers[i][j] - learningRate * weightLayersGradient[i][j];
    }
  }
  return 0;
}
