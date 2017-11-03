#include <iostream>
#include "euclideandistancematrixgpu.h"
#include <vector>
#include <float.h>

using namespace std;

int main(int argc, char** argv){

    if((argc != 10)&&(argc != 11)){
        std::cout<<"Wrong number of arguments!\n";
        std::cout<<"Proper launching: hello.exe <input_file> <output_file> <number_of_neighbours> <number_of_trees> <number_of_elements_in_leaf> <number_of_propagations> <distance_Algorithm> <id_gpu> <cluster: T/F> <--debug>\n";
        std::cout<<"Example launching: hello.exe input.csv output.csv 2 10 10 10 EUCLIDEAN 1 T\n";
        std::cout<<"Example launching: hello.exe input.csv output.csv 2 10 10 10 EUCLIDEAN 1 T --debug\n";
        std::cout<<"Available distance algorithms: EUCLIDEAN TAXICAB COSINE\n";
        std::cout<<"Attention!\n";
        std::cout<<"Delimeter -> ,\n";
        std::cout<<"If file has cluster (at the end of line) you should run program with cluster set T, otherwise F\n";
        return 1;
    }

    //parsowanie argumentow
    std::string inputFile = std::string(argv[1]);
    std::string outputFile = std::string(argv[2]);
    int numberOfNeighbours = atoi(argv[3]);
    int numberOfTrees = atoi(argv[4]);
    int numberOfElemsInLeaf = atoi(argv[5]);
    int numberOfPropagation = atoi(argv[6]);
    int typeOfDistance;
    std::vector<int> devices;
    bool klasterNaKoncu;
    bool debugMode = false;

    if(std::string(argv[7]).compare("EUCLIDEAN") == 0){
        typeOfDistance = DISTANCE_EUCLIDEAN;
    }else if(std::string(argv[7]).compare("TAXICAB") == 0){
        typeOfDistance = DISTANCE_TAXICAB;
    }else if(std::string(argv[7]).compare("COSINE") == 0){
        typeOfDistance = DISTANCE_COSINE;
    }else{
        std::cout<<"Wrong distance algorithm!\n";
        std::cout<<"Available distance algorithms: EUCLIDEAN TAXICAB COSINE\n";
        return 2;
    }

    char* pch = strtok(argv[8],",");
    while (pch != NULL){
        int dev = atoi(pch);
        devices.push_back(dev);
        pch = strtok(NULL, ",");
    }

    //sprawdzanie czy wybrane karty sa dostepne
    for(int i=0 ; i<devices.size() ; ++i){
        int dev = devices[i];
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, dev) == cudaSuccess){
            if(prop.major == 1){
                std::cout<<"Device with id "<<dev<<" can not be use because Compute Capability. Required at least 2.0 Compute Capability, your GPU has "<<prop.major<<"."<<prop.minor<<"\n";
                return 4;
            }
            if(prop.major == 9999){
                std::cout<<"This computer does not have any GPU with CUDA\n";
                return 5;
            }
        }else{
            std::cout<<"Device with id "<<dev<<" is not located in this computer\n";
            return 3;
        }
    }

    if(std::string(argv[9]).compare("T") == 0){
        klasterNaKoncu = true;
    }else{
        klasterNaKoncu = false;
    }

    if(argc == 11){
        if(std::string(argv[10]).compare("--debug") == 0){
            debugMode = true;
        }
    }

    //wczytanie wymiarowosci i ilosci punktow
    int dimentionOfData = 0;
    int numberOfPoints = 0;

    std::ifstream myfile;
    myfile.open(inputFile.c_str());
    if (myfile.is_open()){
    }else{
        std::cout<<"Error opening the file\n";
        return 7;
    }
    std::string line;

    std::getline(myfile, line);
    numberOfPoints = atoi(line.c_str());
    std::getline(myfile, line);
    dimentionOfData = atoi(line.c_str());
    myfile.close();

    //wyliczanie
    EuclideanDistanceMatrixGPU edm(debugMode);
    edm.setDataFile(inputFile);
    edm.setResultsFile(outputFile);
    bool error = edm.initialize(numberOfPoints, dimentionOfData, numberOfNeighbours, devices[0], typeOfDistance, klasterNaKoncu, numberOfTrees, numberOfPropagation, numberOfElemsInLeaf);
    if(error){
        std::cout<<"Initialization failed.\n";
        return 8;
    }
    std::cout<<"I begin analysis.\n";
    edm.calculate();
    std::cout<<"I am writing the solution to a file.\n";
    edm.saveResultToResultFile();
    error = edm.deinitialize();
    if(error){
        std::cout<<"Deinitialization failed.\n";
        return 9;
    }

    cout<<"The end\n";
    return 0;
}
