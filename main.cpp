#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

#define NN_DATA_FILENAME "nn_data"
#define IMAGE_TO_RECOGNIZE_FILENAME "image_to_recognize"
#define TRAINING_IMAGES_FILENAME "training_images"
#define TRAINING_LABELS_FILENAME "training_labels"
#define LAYER_0_NEURON_AMOUNT 784 //image size
#define LAYER_1_NEURON_AMOUNT 128
#define LAYER_2_NEURON_AMOUNT 10

class NN {
  public: 
  NN();
  ~NN();
  bool initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounds);
  bool train(std::string fileNameTrainImages, std::string fileNameTrainLabels);
  bool recognize(std::string fileNameRecognize);
  bool randomizeData();
  private:
  std::string fileNameData;
  size_t dataSize; //In elements
  std::vector<std::vector<double>> weightLayers; //weightLayers[layerIndex][weightIndex], neuronLayers size -1 (input layer)
  std::vector<std::vector<double>> biasLayers;
  std::vector<std::vector<double>> neuronLayers;
  bool parseData();
  bool saveData();
  bool checkFileExistence(std::string fileName);
  double sigmoid(double num);
};

NN::NN() {
    std::cout << "Neural network instance created\n";
}

NN::~NN() {
  std::cout << "Neural network instance deleted\n";
  //Free RAM here
}

double NN::sigmoid(double num) {
  return 1.0 / (1.0 + pow(M_E, -1.0*num));
}

bool NN::initiate(std::string fileNameData, std::vector < unsigned int > layerNeuronAmounds){
  this -> fileNameData = fileNameData;
  this -> dataSize = 0;
  neuronLayers.resize(layerNeuronAmounds.size());
  for (size_t i = 0; i < neuronLayers.size();i++)
  {
    neuronLayers[i].resize(layerNeuronAmounds[i]);
  }
  std::cout << "Neural network instance initiated with layout:\n";
  for (size_t i = 0; i < neuronLayers.size(); i++) {
    std::cout << "Layer "<<i<<", "<<neuronLayers[i].size() << " neurons"<<std::endl;
  }
  weightLayers.resize(layerNeuronAmounds.size()-1); //Number of layers without input layer
  biasLayers.resize(layerNeuronAmounds.size()-1);
  for (size_t i = 0; i < layerNeuronAmounds.size()-1;i++)
  {
    weightLayers[i].resize(layerNeuronAmounds[i]*layerNeuronAmounds[i+1]);
    biasLayers[i].resize(layerNeuronAmounds[i+1]);
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

bool NN::recognize(std::string fileNameRecognize) {
  std::ifstream fileRecognizeStream(fileNameRecognize);
  if (!fileRecognizeStream) {
    std::cerr << "Provide a photo file named " <<
      fileNameRecognize <<
      ", 28x28px (784 bytes with a brightness degree of 0-255) written left-to-right from top to bottom. " <<
      "The said file was not detected.\n";
    return 1;
  }
  unsigned char dataChar;
  neuronLayers[0].resize(0); //To use push_back
  while (fileRecognizeStream.read(reinterpret_cast < char * > ( & dataChar), sizeof(dataChar))) {
    neuronLayers[0].push_back((static_cast < unsigned int > (dataChar)) / 255.0); //Normalize 0 - 255 to 0.0 - 1.0
  }
  fileRecognizeStream.close();
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
    }
  }
  double maxValue = neuronLayers[neuronLayers.size() - 1][0]; //State solution
  int maxIndex = 0;
  for (int i = 1; i < neuronLayers[neuronLayers.size() - 1].size(); ++i) {
    if (neuronLayers[neuronLayers.size() - 1][i] > maxValue) {
      maxValue = neuronLayers[neuronLayers.size() - 1][i];
      maxIndex = i;
    }
  }
  std::cout << "Digit recognized: " << maxIndex << ", confidence: " << maxValue * 100.0 << "%\n";
  return 0;
}

bool NN::train(std::string fileNameTrainImages, std::string fileNameTrainLabels) {
  if(!checkFileExistence(fileNameTrainLabels))
  {
    std::cerr << "Digits labels should be written in a file named " 
    << fileNameTrainLabels
    <<" in the order of the photos, one byte is one digit, the said file was not discovered\n";
    return 1;
  }
  if (!checkFileExistence(fileNameTrainImages)) {
    std::cerr << "The training images should be contained in a single file named "
    << fileNameTrainImages
    << ". Each image should be 784 pixels (784 bytes, with brightness levels from 0 to 255), written left to right, top to bottom. You can have as many 784-byte images in this file as the RAM can handle. This file was not found.\n";
    return 1;
  }
  // std::ifstream obiekt456g546g54634e5;
  // obiekt456g546g54634e5.open(TRAINING_LABELS, std::ios::binary);
  // obiekt456g546g54634e5.seekg(0, std::ios::end);
  // unsigned long long * ilosc = new unsigned long long[1]; ////ile zdjenc
  // * ilosc = obiekt456g546g54634e5.tellg();
  // double * img = new double[ * ilosc * 784]; //////foto
  // double * lab = new double[ * ilosc]; //////oznaczenia
  // obiekt456g546g54634e5.seekg(0, std::ios::beg);
  // char * pomoc43f5345f34 = new char[ * ilosc];
  // obiekt456g546g54634e5.read(pomoc43f5345f34, * ilosc);
  // obiekt456g546g54634e5.close();
  // for (size_t i = 0; i < * ilosc; i++) {
  //   lab[i] = static_cast < double > (pomoc43f5345f34[i]);
  // }
  // std::ifstream obiekt9990;
  // obiekt9990.open(TRAINING_IMAGES, std::ios::binary);
  // char * pomoc6g5464h7 = new char[ * ilosc * 784];
  // obiekt9990.read(pomoc6g5464h7, * ilosc * 784);
  // obiekt9990.close();
  // for (size_t i = 0; i < * ilosc * 784; i++) {
  //   if (static_cast < int > (pomoc6g5464h7[i]) < 0) {
  //     img[i] = (256 + static_cast < double > (pomoc6g5464h7[i])) / 255;
  //   } else {
  //     img[i] = (static_cast < double > (pomoc6g5464h7[i])) / 255;
  //   }
  // }
  // //////////calculus
  // double * war0val = new double[784]; /////wartosci
  // double * war1val = new double[784];
  // double * war2val = new double[784];
  // double * war3val = new double[10];
  // double * gradient = new double[1238730];
  // double * c = new double[1];//zerowane
  // double * c_exp = new double[1];//zerowane
  // double * lab_comp = new double[10];
  // double * small = new double[1];
  // * small = 0.12;
  // //zerowanie gradientu
  // for (size_t ti = 0; ti < 1238730; ti++) {
  //   gradient[ti] = 0;
  // }
  // //////////////////iteracja gurna
  // for (size_t i = 0; i < * zakres_n; i++) {
  //   *c=0;
  // ///////liczymy c
  //   for (size_t tt = 0; tt < 784; tt++) //war 0
  //   {
  //     war0val[tt] = img[tt + 784 * i];
  //   }
  //   for (size_t yy = 0; yy < 784; yy++) ///pierwsza warstwa
  //   {
  //     double pomoc4358345h8i = 0.0;
  //     for (size_t hh = 0; hh < 784; hh++) {
  //       pomoc4358345h8i = pomoc4358345h8i + war0val[hh] * war1weight[hh + 784 * yy];
  //     }
  //     war1val[yy] = sigmoid(war1bias[yy] + pomoc4358345h8i);
  //   }
  //   for (size_t yy = 0; yy < 784; yy++) ///druga warstwa
  //   {
  //     double pomoc4358345h8i = 0.0;
  //     for (size_t hh = 0; hh < 784; hh++) {
  //       pomoc4358345h8i = pomoc4358345h8i + war1val[hh] * war2weight[hh + 784 * yy];
  //     }
  //     war2val[yy] = sigmoid(war2bias[yy] + pomoc4358345h8i);
  //   }
  //   for (size_t nn = 0; nn < 10; nn++) ///trzecia warstwa
  //   {
  //     double pomoc564g6 = 0.0;
  //     for (size_t uu = 0; uu < 784; uu++) {
  //       pomoc564g6 = pomoc564g6 + war2val[uu] * war3weight[uu + 784 * nn];
  //     }
  //     war3val[nn] = sigmoid(war3bias[nn] + pomoc564g6);
  //   }
  //   ////////////koniec wyliczania wartosci
  //   for (size_t ggg = 0; ggg < 10; ggg++) {
  //     if (lab[i] == ggg) {
  //       lab_comp[ggg] = 1;
  //     } else {
  //       lab_comp[ggg] = 0;
  //     }
  //   }
  //   for (size_t vhv = 0; vhv < 10; vhv++) {
  //     * c = * c + (lab_comp[vhv] - war3val[vhv]) * (lab_comp[vhv] - war3val[vhv]); ///////////////////////////////////////////////////////////////////////////////////*0.5 ??
  //   }
  //   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////iteracja zagnierzdzona
  //   for (size_t bbi = 0; bbi < 1238730; bbi++) {
  //   	* c_exp=0;
  //     if(bbi%6194==0)
  //     {
  //   std::cout<<"training "<<(bbi/1238730.0*100)/(*zakres_n)<<"% do not interrupt\n";
  //   }
  //   //podmiana
  //     if (bbi <= 783) {
  //       war1bias[bbi] = war1bias[bbi] + * small;
  //     } //{0 - 783}
  //     if (bbi >= 784 && bbi <= 615439) {
  //       war1weight[bbi - 784] = war1weight[bbi - 784] + * small;
  //     } //{784 - 615439}
  //     if (bbi >= 615440 && bbi <= 616223) {
  //       war2bias[bbi - 615440] = war2bias[bbi - 615440] + * small;
  //     } //{615440 - 616223}
  //     if (bbi >= 616224 && bbi <= 1230879) {
  //       war2weight[bbi - 616224] = war2weight[bbi - 616224] + * small;
  //     } //{616224 - 1230879}
  //     if (bbi >= 1230880 && bbi <= 1230889) {
  //       war3bias[bbi - 1230880] = war3bias[bbi - 1230880] + * small;
  //     } //{1230880 - 1230889}
  //     if (bbi >= 1230890) {
  //       war3weight[bbi - 1230890] = war3weight[bbi - 1230890] + * small;
  //     } //{1230890 - 1238729}
  //     //////////////////////////////////////////////////////////////////////////////////kalkulacja c_exp
  //     for (size_t yy = 0; yy < 784; yy++) ///pierwsza warstwa
  //     {
  //       double pomoc4358345h8i = 0.0;
  //       for (size_t hh = 0; hh < 784; hh++) {
  //         pomoc4358345h8i = pomoc4358345h8i + war0val[hh] * war1weight[hh + 784 * yy];
  //       }
  //       war1val[yy] = sigmoid(war1bias[yy] + pomoc4358345h8i);
  //     }
  //     for (size_t yy = 0; yy < 784; yy++) ///druga warstwa
  //     {
  //       double pomoc4358345h8i = 0.0;
  //       for (size_t hh = 0; hh < 784; hh++) {
  //         pomoc4358345h8i = pomoc4358345h8i + war1val[hh] * war2weight[hh + 784 * yy];
  //       }
  //       war2val[yy] = sigmoid(war2bias[yy] + pomoc4358345h8i);
  //     }
  //     for (size_t nn = 0; nn < 10; nn++) ///trzecia warstwa
  //     {
  //       double pomoc564g6 = 0.0;
  //       for (size_t uu = 0; uu < 784; uu++) {
  //         pomoc564g6 = pomoc564g6 + war2val[uu] * war3weight[uu + 784 * nn];
  //       }
  //       war3val[nn] = sigmoid(war3bias[nn] + pomoc564g6);
  //     }
  //     ////////////////////////////////////////////////////////////////////////////////////////////////
  //     for (size_t asdd = 0; asdd < 10; asdd++) {
  //       * c_exp = * c_exp + (lab_comp[asdd] - war3val[asdd]) * (lab_comp[asdd] - war3val[asdd]); ///////////////////////////////////////////////////////////////////////////////////*0.5 ??
  //     }
  //     //odmiana
  //     if (bbi <= 783) {
  //       war1bias[bbi] = war1bias[bbi] - * small;
  //     } //{0 - 783}
  //     if (bbi >= 784 && bbi <= 615439) {
  //       war1weight[bbi - 784] = war1weight[bbi - 784] - * small;
  //     } //{784 - 615439}
  //     if (bbi >= 615440 && bbi <= 616223) {
  //       war2bias[bbi - 615440] = war2bias[bbi - 615440] - * small;
  //     } //{615440 - 616223}
  //     if (bbi >= 616224 && bbi <= 1230879) {
  //       war2weight[bbi - 616224] = war2weight[bbi - 616224] - * small;
  //     } //{616224 - 1230879}
  //     if (bbi >= 1230880 && bbi <= 1230889) {
  //       war3bias[bbi - 1230880] = war3bias[bbi - 1230880] - * small;
  //     } //{1230880 - 1230889}
  //     if (bbi >= 1230890) {
  //       war3weight[bbi - 1230890] = war3weight[bbi - 1230890] - * small;
  //     } //{1230890 - 1238729}
  //     //append
  //   gradient[bbi] = gradient[bbi] + ( *c_exp - *c) / ( * small);
  //   }
  //   /////////////////end
  // }
  // for (size_t ddd = 0; ddd < 1238730; ddd++) {
  //   gradient[ddd] = gradient[ddd] * 0.01 / ( * zakres_n);
  //   if (ddd <= 783) {
  //     war1bias[ddd] = war1bias[ddd] - gradient[ddd];
  //   } //{0 - 783}
  //   if (ddd >= 784 && ddd <= 615439) {
  //     war1weight[ddd - 784] = war1weight[ddd - 784] - gradient[ddd];
  //   } //{784 - 615439}
  //   if (ddd >= 615440 && ddd <= 616223) {
  //     war2bias[ddd - 615440] = war2bias[ddd - 615440] - gradient[ddd];
  //   } //{615440 - 616223}
  //   if (ddd >= 616224 && ddd <= 1230879) {
  //     war2weight[ddd - 616224] = war2weight[ddd - 616224] - gradient[ddd];
  //   } //{616224 - 1230879}
  //   if (ddd >= 1230880 && ddd <= 1230889) {
  //     war3bias[ddd - 1230880] = war3bias[ddd - 1230880] - gradient[ddd];
  //   } //{1230880 - 1230889}
  //   if (ddd >= 1230890) {
  //     war3weight[ddd - 1230890] = war3weight[ddd - 1230890] - gradient[ddd];
  //   } //{1230890 - 1238729}
  // }
  // sav3();
  std::cout << "Training completed\n";
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
    std::cout << "To recognize the digit enter 0, to train the network enter 1\n";
    bool selection;
    std::cin >> selection;
    if (selection == 0) {
        if (network_0->recognize(IMAGE_TO_RECOGNIZE_FILENAME)) {
            delete network_0; 
            return 1;
        }
    } else {
        if (network_0->train(TRAINING_IMAGES_FILENAME, TRAINING_LABELS_FILENAME)) {
            delete network_0; 
            return 1;
        }
    }
    delete network_0;
    return 0;
}