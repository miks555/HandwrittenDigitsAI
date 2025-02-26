#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

#define NN_DATA_FILENAME "nn_data"
#define DATA_TO_RECOGNIZE_FILENAME "image_to_recognize"
#define TRAINING_DATA_FILENAME "training_images"
#define TRAINING_LABELS_FILENAME "training_labels"
#define LAYER_0_NEURON_AMOUNT 784 //Input data size
#define LAYER_1_NEURON_AMOUNT 128
#define LAYER_2_NEURON_AMOUNT 10
#define LEARNING_RATE 0.1

class NN {
  public: 
  NN();
  ~NN();
  bool initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounds);
  bool train(std::string fileNameTrainData, std::string fileNameTrainLabels);
  bool recognize(std::string fileNameRecognize);
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
  bool propagateBackward(unsigned int label);
  double MeanSquaredErrorDerivative(double num, double target);
  double sigmoid(double num);
  double sigmoidDerivative(double num);
  bool clearConsole();
};

NN::NN() {
    std::cout << "Neural network instance created\n";
}

NN::~NN() {
  std::cout << "Neural network instance deleted\n";
}

double NN::sigmoid(double num) {
  return 1.0 / (1.0 + pow(M_E, -1.0*num));
}

double NN::sigmoidDerivative(double num) {
   return sigmoid(num) * (1.0 - sigmoid(num));
}

double NN::MeanSquaredErrorDerivative(double num, double target) {
   return 2.0*(num - target);
}

bool NN::initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounds){
  this -> fileNameData = fileNameData;
  this -> dataSize = 0;
  neuronLayers.resize(layerNeuronAmounds.size());
  neuronLayersSums.resize(layerNeuronAmounds.size());
  for (size_t i = 0; i < neuronLayers.size();i++)
  {
    neuronLayers[i].resize(layerNeuronAmounds[i]);
    neuronLayersSums[i].resize(layerNeuronAmounds[i]);
  }
  std::cout << "Neural network instance initiated with layout:\n";
  for (size_t i = 0; i < neuronLayers.size(); i++) {
    std::cout << "Layer "<<i<<", "<<neuronLayers[i].size() << " neurons"<<std::endl;
  }
  weightLayers.resize(layerNeuronAmounds.size()-1); //Number of layers without input layer
  biasLayers.resize(layerNeuronAmounds.size()-1);
  weightLayersGradient.resize(layerNeuronAmounds.size()-1);
  biasLayersGradient.resize(layerNeuronAmounds.size()-1);
  for (size_t i = 0; i < layerNeuronAmounds.size()-1;i++)
  {
    weightLayers[i].resize(layerNeuronAmounds[i]*layerNeuronAmounds[i+1]);
    biasLayers[i].resize(layerNeuronAmounds[i+1]);
    weightLayersGradient[i].resize(layerNeuronAmounds[i]*layerNeuronAmounds[i+1]);
    biasLayersGradient[i].resize(layerNeuronAmounds[i+1]);
  }
  for (size_t i = 0; i < weightLayers.size(); i++) {
    dataSize = dataSize + weightLayers[i].size() + biasLayers[i].size();
  }
  std::cout << "Data size in elements: "<<dataSize<<std::endl;
  std::cout << "Data size in bytes: "<<dataSize*sizeof(double)<<std::endl;
  if (!checkFileExistence(fileNameData)) {
    std::cerr << "No neural network data found, randomizing...\n";
    if(randomizeData()){
      std::cerr<<"Data randomization error\n";
      return 1;
    }
  }else{
  std::cout << "Parsing neural network data...\n";
  if(parseData()){
    std::cerr<<"Data parsing error\n";
    return 1;
  }
  }
  std::cout << "Network initiated\n";
  return 0;
}

bool NN::checkFileExistence(std::string fileName) {
  std::ifstream fileCheckStream(fileName);
  return fileCheckStream.good();
}

bool NN::randomizeData() {
  srand(static_cast<unsigned int>(time(NULL)));
  for (size_t i = 0; i < weightLayers.size(); i++) {
    size_t n_in = neuronLayers[i].size();
    size_t n_out = neuronLayers[i+1].size();
    double limit = sqrt(6.0 / (n_in + n_out));
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      weightLayers[i][j] = (static_cast<double>(std::rand()) / RAND_MAX) * 2 * limit - limit;
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
  std::vector<double> parsedData(dataSize);
  parseDataStream.read(reinterpret_cast<char*>(parsedData.data()), dataSize * sizeof(double));
  parseDataStream.close();
  size_t parsedDataIndex = 0;
  for(size_t i = 0 ; i < weightLayers.size();i++)
  {
    for(size_t j = 0 ; j < biasLayers[i].size();j++)
    {
      if (parsedDataIndex >= parsedData.size()) {
        return 1;
      }
      biasLayers[i][j] = parsedData[parsedDataIndex];
      parsedDataIndex++;
    }
    for(size_t j = 0 ; j < weightLayers[i].size();j++)
    {
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
  std::vector<double> saveData;
  for (size_t i = 0; i < weightLayers.size(); i++) {
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      saveData.push_back(biasLayers[i][j]);
    }
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      saveData.push_back(weightLayers[i][j]);
    }
  }
  saveDataStream.write(reinterpret_cast<const char*>(saveData.data()), saveData.size() * sizeof(double));
  saveDataStream.close();
  if(!checkFileExistence(fileNameData)){
    return 1;
  }
  return 0;
}

bool NN::propagateForward(){
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
}

bool NN::clearConsole() {
  #ifdef _WIN32
    system("cls"); // Windows
  #else
    system("clear"); // Linux/macOS
  #endif
}

bool NN::recognize(std::string fileNameRecognize) {
  std::ifstream fileRecognizeStream(fileNameRecognize);
  if (!fileRecognizeStream) {
    std::cerr <<fileNameRecognize<<" not found\n";
    return 1;
  } 
  unsigned char dataChar;
  neuronLayers[0].resize(0); //To use push_back
  while (fileRecognizeStream.read(reinterpret_cast < char * > ( & dataChar), sizeof(dataChar))) {
    neuronLayers[0].push_back((static_cast < unsigned int > (dataChar)) / 255.0); //Normalize 0 - 255 to 0.0 - 1.0
  }
  fileRecognizeStream.close();
  propagateForward();
  double maxValue = neuronLayers[neuronLayers.size() - 1][0]; //State solution
  int maxIndex = 0;
  for (int i = 1; i < neuronLayers[neuronLayers.size() - 1].size(); ++i) {
    if (neuronLayers[neuronLayers.size() - 1][i] > maxValue) {
      maxValue = neuronLayers[neuronLayers.size() - 1][i];
      maxIndex = i;
    }
  }
  std::cout << "Recognized: " << maxIndex << ", confidence: " << maxValue * 100.0 << "%\n";
  return 0;
}

bool NN::train(std::string fileNameTrainData, std::string fileNameTrainLabels) {
  std::ifstream trainLabelsStream(fileNameTrainLabels);
  if (!trainLabelsStream) {
    std::cerr << fileNameTrainLabels << " not found\n";
    return 1;
  }
  std::ifstream trainDataStream(fileNameTrainData);
  if (!trainDataStream) {
    std::cerr << fileNameTrainLabels << " not found\n";
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
    propagateBackward(label);
    if (i % (trainSetSize / 1000) == 0) //Progress info
    {
      clearConsole();
      std::cout << std::fixed << std::setprecision(1) << "Training progress: " << ((i * 100.0) / trainSetSize) << "%\n";
    }
  }
  trainLabelsStream.close();
  trainDataStream.close();
  saveData();
  clearConsole();
  std::cout << "Training completed\n";
  return 0;
}

bool NN::propagateBackward(unsigned int label) {
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
        double backActivation = 0.0;
        for (size_t k = 0; k < neuronLayers[i + 1].size(); k++) {
          backActivation = backActivation + neuronLayers[i + 1][k] * weightLayers[i][k + j + k * neuronLayers[i].size()];
        }
        double error = sigmoidDerivative(neuronLayersSums[i][j]) * backActivation;
        neuronLayers[i][j] = error;
        for (size_t k = 0; k < neuronLayers[i - 1].size(); k++) {
          weightLayersGradient[i - 1][k + j * neuronLayers[i - 1].size()] = error * neuronLayers[i - 1][k];
        }
        biasLayersGradient[i - 1][j] = error;
      }
    }
  }
  //Update weights and biases
  for (size_t i = 0; i < biasLayers.size(); i++) {
    for (size_t j = 0; j < biasLayers[i].size(); j++) {
      biasLayers[i][j] = biasLayers[i][j] - LEARNING_RATE * biasLayersGradient[i][j];
    }
  }
  for (size_t i = 0; i < weightLayers.size(); i++) {
    for (size_t j = 0; j < weightLayers[i].size(); j++) {
      weightLayers[i][j] = weightLayers[i][j] - LEARNING_RATE * weightLayersGradient[i][j];
    }
  }
  return 0;
}

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
        if (network_0->recognize(DATA_TO_RECOGNIZE_FILENAME)) {
            delete network_0; 
            return 1;
        }
    } else {
        if (network_0->train(TRAINING_DATA_FILENAME, TRAINING_LABELS_FILENAME)) {
            delete network_0; 
            return 1;
        }
    }
    delete network_0;
    return 0;
}