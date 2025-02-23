#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

#define NN_DATA_FILENAME "NN_data"
#define IMAGE_TO_RECOGNIZE_FILENAME "img"


class NN {
public:
    NN(){

    }
    void train(int photosAmount)
    {

    }
    void recognize()
    {

    }

private:
    int layerNum;
    std::vector<int> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    
};


long double * war1bias = new long double[784];
long double * war2bias = new long double[784];
long double * war3bias = new long double[10];

long double * war1weight = new long double[614656];
long double * war2weight = new long double[614656];
long double * war3weight = new long double[7840];

void readModelData() {
  std::string liniachwilowa = "0";
  std::ifstream obiekt5654sds767687;
  obiekt5654sds767687.open(NN_DATA_FILENAME);
  for (int i = 0, j = 0; i < 1238730;) {
    if (i >= 0 and i <= 783) {
      getline(obiekt5654sds767687, liniachwilowa);
      war1bias[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    if (i == 783) {
      j = -1;
    }
    if (i >= 784 and i <= 615439) {
      getline(obiekt5654sds767687, liniachwilowa);
      war1weight[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    if (i == 615439) {
      j = -1;
    }
    if (i >= 615440 and i <= 616223) {
      getline(obiekt5654sds767687, liniachwilowa);
      war2bias[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    if (i == 616223) {
      j = -1;
    }
    if (i >= 616224 and i <= 1230879) {
      getline(obiekt5654sds767687, liniachwilowa);
      war2weight[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    if (i == 1230879) {
      j = -1;
    }
    if (i >= 1230880 and i <= 1230889) {
      getline(obiekt5654sds767687, liniachwilowa);
      war3bias[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    if (i == 1230889) {
      j = -1;
    }
    if (i >= 1230890 and i <= 1238729) {
      getline(obiekt5654sds767687, liniachwilowa);
      war3weight[j] = strtold(liniachwilowa.c_str(), NULL);
    }
    j++;
    i++;
  }
  obiekt5654sds767687.close();
}

void sav3() {
  std::ofstream obiekt6d7hd567;
  obiekt6d7hd567.open(NN_DATA_FILENAME, std::ios::trunc);
  obiekt6d7hd567 << std::fixed;
  obiekt6d7hd567 << std::setprecision(10);
  for (int i = 0; i < 784; i++) {
    obiekt6d7hd567 << war1bias[i] << std::endl;
  }
  for (int i = 0; i < 614656; i++) {
    obiekt6d7hd567 << war1weight[i] << std::endl;
  }
  for (int i = 0; i < 784; i++) {
    obiekt6d7hd567 << war2bias[i] << std::endl;
  }
  for (int i = 0; i < 614656; i++) {
    obiekt6d7hd567 << war2weight[i] << std::endl;
  }
  for (int i = 0; i < 10; i++) {
    obiekt6d7hd567 << war3bias[i] << std::endl;
  }
  for (int i = 0; i < 7840; i++) {
    obiekt6d7hd567 << war3weight[i] << std::endl;
  }
  obiekt6d7hd567.close();
}

long double sigmoid(long double a) {
  return 1.0 / (1.0 + pow(M_E, -1.0*a));
}

void randomizeNNData() {
    std::ofstream modelData;
    modelData.open(NN_DATA_FILENAME, std::ios::binary);
    srand(time(NULL));
    for (int i = 0; i < 1238730; i++) {
        double randomValue = (double) rand() / RAND_MAX * 0.0001;
        modelData.write(reinterpret_cast<const char*>(&randomValue), sizeof(randomValue));
    }
    modelData.close();
}

void rozpoznawaj() {
  long double * war0val = new long double[784]; /////wartosci
  long double * war1val = new long double[784];
  long double * war2val = new long double[784];
  long double * war3val = new long double[10];
  char * pomoc345345 = new char[784];
  std::ifstream obiekt87986; //////wczytanie zdjencia
  obiekt87986.open(IMAGE_TO_RECOGNIZE_FILENAME, std::ios::binary);
  obiekt87986.read(pomoc345345, 784);
  for (int i = 0; i < 784; i++) {
    if (static_cast < int > (pomoc345345[i]) < 0) {
      war0val[i] = (256.0 + static_cast < long double > (pomoc345345[i])) / 255.00;
    } else {
      war0val[i] = (static_cast < long double > (pomoc345345[i])) / 255.00;
    }
  }
  obiekt87986.close();
  for (int i = 0; i < 784; i++) ///pierwsza warstwa
  {
    long double pomoc4358345h8i = 0.0;
    for (int j = 0; j < 784; j++) {
      pomoc4358345h8i = pomoc4358345h8i + war0val[j] * war1weight[j + 784 * i];
    }
    war1val[i] = sigmoid(war1bias[i] + pomoc4358345h8i);
  }
  for (int i = 0; i < 784; i++) ///druga warstwa
  {
    long double pomoc345f345f = 0.0;
    for (int j = 0; j < 784; j++) {
      pomoc345f345f = pomoc345f345f + war1val[j] * war2weight[j + 784 * i];
    }
    war2val[i] = sigmoid(war2bias[i] + pomoc345f345f);
  }
  for (int i = 0; i < 10; i++) ///trzecia warstwa
  {
    long double pomoc564g6 = 0.0;
    for (int j = 0; j < 784; j++) {
      pomoc564g6 = pomoc564g6 + war2val[j] * war3weight[j + 784 * i];
    }
    war3val[i] = sigmoid(war3bias[i] + pomoc564g6);
  }
  long double max1dd23423 = 0.0; ////wypisanie najprawdopodobniejszych/ej
  for (int i = 0; i < 10; i++) {
    if (max1dd23423 < war3val[i]) {
      max1dd23423 = war3val[i];
    }
  }
  for (int i = 0; i < 10; i++) {
    if (max1dd23423 == war3val[i]) {
      std::cout << std::fixed << i << " sure for " << war3val[i] * 100 << "%" << std::endl;
    }
  }
}

void trenuj(int * zakres_n) {
  std::ifstream obiekt456g546g54634e5;
  obiekt456g546g54634e5.open("lab", std::ios::binary);
  obiekt456g546g54634e5.seekg(0, std::ios::end);
  unsigned long long * ilosc = new unsigned long long[1]; ////ile zdjenc
  * ilosc = obiekt456g546g54634e5.tellg();
  long double * img = new long double[ * ilosc * 784]; //////foto
  long double * lab = new long double[ * ilosc]; //////oznaczenia
  obiekt456g546g54634e5.seekg(0, std::ios::beg);
  char * pomoc43f5345f34 = new char[ * ilosc];
  obiekt456g546g54634e5.read(pomoc43f5345f34, * ilosc);
  obiekt456g546g54634e5.close();
  for (int i = 0; i < * ilosc; i++) {
    lab[i] = static_cast < long double > (pomoc43f5345f34[i]);
  }
  std::ifstream obiekt9990;
  obiekt9990.open("img", std::ios::binary);
  char * pomoc6g5464h7 = new char[ * ilosc * 784];
  obiekt9990.read(pomoc6g5464h7, * ilosc * 784);
  obiekt9990.close();
  for (int i = 0; i < * ilosc * 784; i++) {
    if (static_cast < int > (pomoc6g5464h7[i]) < 0) {
      img[i] = (256 + static_cast < long double > (pomoc6g5464h7[i])) / 255;
    } else {
      img[i] = (static_cast < long double > (pomoc6g5464h7[i])) / 255;
    }
  }
  //////////calculus
  long double * war0val = new long double[784]; /////wartosci
  long double * war1val = new long double[784];
  long double * war2val = new long double[784];
  long double * war3val = new long double[10];
  long double * gradient = new long double[1238730];
  long double * c = new long double[1];//zerowane
  long double * c_exp = new long double[1];//zerowane
  long double * lab_comp = new long double[10];
  long double * small = new long double[1];
  * small = 0.12;
  //zerowanie gradientu
  for (int ti = 0; ti < 1238730; ti++) {
    gradient[ti] = 0;
  }
  //////////////////iteracja gurna
  for (int i = 0; i < * zakres_n; i++) {
    *c=0;
	///////liczymy c
    for (int tt = 0; tt < 784; tt++) //war 0
    {
      war0val[tt] = img[tt + 784 * i];
    }
    for (int yy = 0; yy < 784; yy++) ///pierwsza warstwa
    {
      long double pomoc4358345h8i = 0.0;
      for (int hh = 0; hh < 784; hh++) {
        pomoc4358345h8i = pomoc4358345h8i + war0val[hh] * war1weight[hh + 784 * yy];
      }
      war1val[yy] = sigmoid(war1bias[yy] + pomoc4358345h8i);
    }
    for (int yy = 0; yy < 784; yy++) ///druga warstwa
    {
      long double pomoc4358345h8i = 0.0;
      for (int hh = 0; hh < 784; hh++) {
        pomoc4358345h8i = pomoc4358345h8i + war1val[hh] * war2weight[hh + 784 * yy];
      }
      war2val[yy] = sigmoid(war2bias[yy] + pomoc4358345h8i);
    }
    for (int nn = 0; nn < 10; nn++) ///trzecia warstwa
    {
      long double pomoc564g6 = 0.0;
      for (int uu = 0; uu < 784; uu++) {
        pomoc564g6 = pomoc564g6 + war2val[uu] * war3weight[uu + 784 * nn];
      }
      war3val[nn] = sigmoid(war3bias[nn] + pomoc564g6);
    }
    ////////////koniec wyliczania wartosci
    for (int ggg = 0; ggg < 10; ggg++) {
      if (lab[i] == ggg) {
        lab_comp[ggg] = 1;
      } else {
        lab_comp[ggg] = 0;
      }
    }
    for (int vhv = 0; vhv < 10; vhv++) {
      * c = * c + (lab_comp[vhv] - war3val[vhv]) * (lab_comp[vhv] - war3val[vhv]); ///////////////////////////////////////////////////////////////////////////////////*0.5 ??
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////iteracja zagnierzdzona
    for (int bbi = 0; bbi < 1238730; bbi++) {
    	* c_exp=0;
      if(bbi%6194==0)
      {
	  std::cout<<"training "<<(bbi/1238730.0*100)/(*zakres_n)<<"% do not interrupt"<<std::endl;
	  }
	  //podmiana
      if (bbi <= 783) {
        war1bias[bbi] = war1bias[bbi] + * small;
      } //{0 - 783}
      if (bbi >= 784 && bbi <= 615439) {
        war1weight[bbi - 784] = war1weight[bbi - 784] + * small;
      } //{784 - 615439}
      if (bbi >= 615440 && bbi <= 616223) {
        war2bias[bbi - 615440] = war2bias[bbi - 615440] + * small;
      } //{615440 - 616223}
      if (bbi >= 616224 && bbi <= 1230879) {
        war2weight[bbi - 616224] = war2weight[bbi - 616224] + * small;
      } //{616224 - 1230879}
      if (bbi >= 1230880 && bbi <= 1230889) {
        war3bias[bbi - 1230880] = war3bias[bbi - 1230880] + * small;
      } //{1230880 - 1230889}
      if (bbi >= 1230890) {
        war3weight[bbi - 1230890] = war3weight[bbi - 1230890] + * small;
      } //{1230890 - 1238729}
      //////////////////////////////////////////////////////////////////////////////////kalkulacja c_exp
      for (int yy = 0; yy < 784; yy++) ///pierwsza warstwa
      {
        long double pomoc4358345h8i = 0.0;
        for (int hh = 0; hh < 784; hh++) {
          pomoc4358345h8i = pomoc4358345h8i + war0val[hh] * war1weight[hh + 784 * yy];
        }
        war1val[yy] = sigmoid(war1bias[yy] + pomoc4358345h8i);
      }
      for (int yy = 0; yy < 784; yy++) ///druga warstwa
      {
        long double pomoc4358345h8i = 0.0;
        for (int hh = 0; hh < 784; hh++) {
          pomoc4358345h8i = pomoc4358345h8i + war1val[hh] * war2weight[hh + 784 * yy];
        }
        war2val[yy] = sigmoid(war2bias[yy] + pomoc4358345h8i);
      }
      for (int nn = 0; nn < 10; nn++) ///trzecia warstwa
      {
        long double pomoc564g6 = 0.0;
        for (int uu = 0; uu < 784; uu++) {
          pomoc564g6 = pomoc564g6 + war2val[uu] * war3weight[uu + 784 * nn];
        }
        war3val[nn] = sigmoid(war3bias[nn] + pomoc564g6);
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////
      for (int asdd = 0; asdd < 10; asdd++) {
        * c_exp = * c_exp + (lab_comp[asdd] - war3val[asdd]) * (lab_comp[asdd] - war3val[asdd]); ///////////////////////////////////////////////////////////////////////////////////*0.5 ??
      }
      //odmiana
      if (bbi <= 783) {
        war1bias[bbi] = war1bias[bbi] - * small;
      } //{0 - 783}
      if (bbi >= 784 && bbi <= 615439) {
        war1weight[bbi - 784] = war1weight[bbi - 784] - * small;
      } //{784 - 615439}
      if (bbi >= 615440 && bbi <= 616223) {
        war2bias[bbi - 615440] = war2bias[bbi - 615440] - * small;
      } //{615440 - 616223}
      if (bbi >= 616224 && bbi <= 1230879) {
        war2weight[bbi - 616224] = war2weight[bbi - 616224] - * small;
      } //{616224 - 1230879}
      if (bbi >= 1230880 && bbi <= 1230889) {
        war3bias[bbi - 1230880] = war3bias[bbi - 1230880] - * small;
      } //{1230880 - 1230889}
      if (bbi >= 1230890) {
        war3weight[bbi - 1230890] = war3weight[bbi - 1230890] - * small;
      } //{1230890 - 1238729}
      //append
	  gradient[bbi] = gradient[bbi] + ( *c_exp - *c) / ( * small);
    }
    /////////////////end
  }
  for (int ddd = 0; ddd < 1238730; ddd++) {
    gradient[ddd] = gradient[ddd] * 0.01 / ( * zakres_n);
    if (ddd <= 783) {
      war1bias[ddd] = war1bias[ddd] - gradient[ddd];
    } //{0 - 783}
    if (ddd >= 784 && ddd <= 615439) {
      war1weight[ddd - 784] = war1weight[ddd - 784] - gradient[ddd];
    } //{784 - 615439}
    if (ddd >= 615440 && ddd <= 616223) {
      war2bias[ddd - 615440] = war2bias[ddd - 615440] - gradient[ddd];
    } //{615440 - 616223}
    if (ddd >= 616224 && ddd <= 1230879) {
      war2weight[ddd - 616224] = war2weight[ddd - 616224] - gradient[ddd];
    } //{616224 - 1230879}
    if (ddd >= 1230880 && ddd <= 1230889) {
      war3bias[ddd - 1230880] = war3bias[ddd - 1230880] - gradient[ddd];
    } //{1230880 - 1230889}
    if (ddd >= 1230890) {
      war3weight[ddd - 1230890] = war3weight[ddd - 1230890] - gradient[ddd];
    } //{1230890 - 1238729}
  }
  sav3();
}

bool checkFileExistence(const std::string& filename) {
    std::ifstream fileCheck(filename);  // Open file
    if (fileCheck.good()) {
        fileCheck.close();  // Close the file if it's valid
        return true;  // File exists and is accessible
    }
    return false;  // File doesn't exist or can't be opened
}


int main() {
  std::cout << "loading...\n";
  if (checkFileExistence(NN_DATA_FILENAME) != 1) {
    std::cout << "no model data found, creating new untrained model...\n";
    randomizeNNData();
  }
  std::cout << "reading model data...\n";
  readModelData();
  std::cout << "Welcome to the neural network that recognizes handwritten digits, to recognize the digit enter 0, to train the network enter 1\n";
  bool * selection = new bool[1];// Selecting menu option
  std::cin >> * selection;
  if ( * selection == 0) {
    std::ifstream obiekt546456;
    obiekt546456.open(IMAGE_TO_RECOGNIZE_FILENAME);
    bool * good345345 = new bool[1];
    * good345345 = obiekt546456.good();
    obiekt546456.close();
    if ( * good345345 != 1) {
          std::cout << "Recognized digit must be a photo file named " 
              << IMAGE_TO_RECOGNIZE_FILENAME 
              << ", 784 pixels (784 bytes with a brightness degree of 0-255) written left-to-right from top to bottom. "
              << "The said file was not detected." << std::endl;
      std::cout << std::endl << std::endl;
      return 0;
    }
    rozpoznawaj();
    std::cout << std::endl << std::endl;
    return 0;
  } else {
    std::ifstream obiekt777788878756546754654;
    obiekt777788878756546754654.open("lab");
    bool * good555 = new bool[1];
    * good555 = obiekt777788878756546754654.good();
    obiekt777788878756546754654.close();
    std::ifstream obiekt898;
    obiekt898.open("img");
    bool * good777 = new bool[1];
    * good777 = obiekt898.good();
    obiekt898.close();
    if ( * good555 != 1) {
      std::cout << "photo-digit markings should be written in a file named lab in the order of the photos, one byte is one digit, the said file was not discovered" << std::endl;
      std::cout << std::endl << std::endl;
      return 0;
    }
    if ( * good777 != 1) {
      std::cout << "training photos should be numbers on photos in a single file named img, photo 784 pixels (784 bytes with a brightness level of 0-255) written left to right from top to bottom, photos of 784 bytes can be in this file as much as you want (as much as it enters the ram), this file was not detected" << std::endl;
      std::cout << std::endl << std::endl;
      return 0;
    }
    //////////////zakres uczenia
    std::cout << "state how many photos to study" << std::endl;
    int * zakres_n = new int[1];
    std::cin >> * zakres_n;
    trenuj(zakres_n);
    std::cout << "trained";
    std::cout << std::endl << std::endl;
    return 0;
  }
}

/*
struktura

warstwa 0
-
warstwa 1
b - 784 wartosci    {0 - 783}
w - 614656 wartosci {784 - 615439}
warstwa 2
b - 784 wartosci    {615440 - 616223}
w - 614656 wartosci {616224 - 1230879}
warstwa 3
b - 10 wartosci     {1230880 - 1230889}
w - 7840 wartosci   {1230890 - 1238729}

razem 1238730 wartosci
*/
