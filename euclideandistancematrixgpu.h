#ifndef EUCLIDEANDISTANCEMATRIXGPU_H
#define EUCLIDEANDISTANCEMATRIXGPU_H

#include <iostream>
#include "cucall.h"
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>

#define DISTANCE_EUCLIDEAN 0
#define DISTANCE_TAXICAB 1
#define DISTANCE_COSINE 2

#define TRUE 1
#define FALSE 0

#define OUT_OF_SOLUTION SHRT_MIN
#define IN_SOLUTION SHRT_MAX

#define NON_OF_BUCKET_IN_INTEREST 1000

typedef struct __align__(16){
    int parent;
    int leftChild;
    int rightChild;
    int numberOfEntities;
} TreeNode;

typedef struct __align__(8){
    int parent;
    int entityNumber;
} TreeNodeLeaf;

typedef struct __align__(8){
    int start;
    int end;
} Partition;

struct DataPoint{
    int id;
    float distance;
};

class EuclideanDistanceMatrixGPU{
    public:
        EuclideanDistanceMatrixGPU();
        EuclideanDistanceMatrixGPU(bool debugMode);
        ~EuclideanDistanceMatrixGPU();

        void setDataFile(std::string nameOfFile);
        bool initialize(int numberOfEntities, int dimensionOfEntity, int numberOfNeighbors,
                        int device, int typeOfDistance, bool klaster, int numberOfTrees,
                        int numberOfPropagations, int minSize);
        bool deinitialize();
        bool calculate();
        void setResultsFile(std::string nameOfFile);
        bool saveResultToResultFile();

    private:
        bool loadData();
        void buildUpTheTrees();
        void findInitialKNN();
        bool initilizeGPUStructuresForTrees();
        bool deinitializeGPUStructuresForTrees();
        void stubInitialization();
        void propagate();

        //Ilosc blokow na jeden multiprocesor
        int numberOfBlocksPerMultiprocessors;

        //Ilosc multiprocesorow dostepnych w danym procesorze graficznym
        int numberOfMultiprocessors;

        //Typ wyznaczanej odleglosci
        int typeOfDistance;

        //Informacja czy na koncu pliku znajduje sie informacja, do ktorego klastra nlezy dany punkt
        bool klaster;

        //Informacja czy bedziemy zapisywac informacje debugowe
        bool debugMode;

        //Ilosc propagacji, ktora jest przeprowadzana na danych
        int numberOfPropagations;

        //Ilosc wyznaczonych drzew
        int numberOfTrees;

        //Path do pliku z danymi
        std::string inputFile;

        //Path do pliku z danymi wynikowymi
        std::string outputFile;

        //Ilosc wymiarow danych
        int dimensionOfEntity;

        //Ilosc punktow w zbiorze danych
        int numberOfEntities;

        //Ilosc sasiadow do wyznaczenia
        int numberOfNeighbors;

        //Maksymalna ilosc elementow w lisciu w drzewie
        int minSize;

        //Dane punktow znajdujace sie na CPU
        float* dataTable_host;

        //Klastry punktow znajdujace sie na CPU
        int* dataTableId_host;

        //Dane punktow znajdujace sie na GPU
        float* dataTable_device;

        //Pitch danych punktow znajdujacych sie na GPU
        size_t dataTable_Pitch;

        //Wyznaczone odleglosci sasiadow znajdujace sie na CPU
        float* neighboursDistance_host;

        //Wyznaczone odleglosci sasiadow znajdujace sie na GPU. Pierwsza z tablic. Zasada jak w OpenGL z double buforem
        float* neighboursDistance_device;

        //Wyznaczone odleglosci sasiadow znajdujace sie na GPU. Druga z tablic. Zasada jak w OpenGL z double buforem
        float* neighboursDistance2_device;

        //Wyznaczone identyfikatory sasiadow znajdujace sie na CPU
        int* neighboursId_host;

        //Wyznaczone identyfikatory sasiadow znajdujace sie na GPU. Pierwsza z tablic. Zasada jak w OpenGL z double buforem
        int* neighboursId_device;

        //Wyznaczone identyfikatory sasiadow znajdujace sie na GPU. Druga z tablic. Zasada jak w OpenGL z double buforem
        int* neighboursId2_device;

        //Stream, na ktorym wszystkie obliczenia sa uruchamiane
        cudaStream_t executionStreams;

        //Event startowy
        cudaEvent_t startEvents;

        //Event koncowy
        cudaEvent_t stopEvents;

        //Identyfikator urzadzenia, na ktorym prowadzone sa obliczenia
        int device;

        //Iformacja o rozmiarze danych, ktorze sa uruchamiane na danym urzadzeniu
        Partition partition;

        //Wyznaczone drzewa podzialu znajdujace sie na CPU
        std::map<int, std::vector<TreeNode> > trees_host;

        //Wyznaczone liscie podzialu znajdujace sie na CPU
        std::map<int, std::vector<TreeNodeLeaf> > treesLeafs_host;

        //Wyznaczone drzewa podzialu znajdujace sie na GPU
        TreeNode** trees_device;

        //Wyznaczone drzewa podzialu, wsklaziniki do dostepu z CPU
        TreeNode** trees_device_pointer_for_cpu;

        //Wyznaczone liscie podzialu znajdujace sie na GPU
        TreeNodeLeaf** treesLeafs_device;

        //Wyznaczone liscie podzialu, wsklaziniki do dostepu z CPU
        TreeNodeLeaf** treesLeafs_device_pointer_for_cpu;

        //Informacja o ilosci nodow w poszczegolnych drzewach na GPU
        int* trees_size_device;

        //Informacja o ilosci nodow w poszczegolnych drzewach na CPU
        int* trees_size_host;

        //Informacja o ilosci lisci w poszczegolnych drzewach na GPU
        int* treesLeafs_size_device;

        //Informacja o ilosci lisci w poszczegolnych drzewach na CPU
        int* treesLeafs_size_host;

        //Informacja, od ktorego noda trzeba sprawdzic wszystkie elementy w drzewie w czasie znajdowania poczatkowych sasiadow
        int* graphTraversalStartPoint_device;

        //Tablica sluzaca do wyznaczania ktore punkty beda sprawdzane. Pomocna przy usuwaniu duplikatow. Dziala jak std::set
        char* idxChecking_device;

        //Maksymalna ilosc elementow do sprawdzenia na jeden raz
        int dimensionOfIndexesAndDistances;

        //Identyfikatory punktow sprawdzanych w danym kroku
        int* indexes_device;

        //Odleglosci punktow sprawdzanych w danym czasie
        float* distances_device;

        //Odleglosci punktow sprawdzanych w danym czasie
        float* distances2_device;

        //Odleglosci punktow sprawdzanych w danym czasie
        float* distances3_device;

        //Informacja o przynaleznosci do rozwiazania danych punktow w danym czasie
        short* marker_device;
};

#endif // EUCLIDEANDISTANCEMATRIXGPU_H
