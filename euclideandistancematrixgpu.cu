#include "euclideandistancematrixgpu.h"
#include <cmath>
#include <float.h>
#include <set>
#include <algorithm>

#define EMPTY_DIRECTION -1

#define SIDE_LEFT 0
#define SIDE_END 1
#define SIDE_RIGHT 2

#define LEFT_SON 0
#define RIGHT_SON 1

#define STUB_INIT_DIST 100000.0
#define STUB_INIT_ID -1

__global__
void normalizeDataStep1(float* dataTable_device, size_t dataTable_pitch, int numberOfEntities, int numberOfDimension){
    __shared__ float mean[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    mean[threadIdx.x] = 0.0f;
    __syncthreads();

    if(tid < numberOfDimension){
        int number = 0;
        while(number < numberOfEntities){
            float* pElement = (float*)((char*)dataTable_device + tid * dataTable_pitch) + number;
            mean[threadIdx.x] += *pElement;
            ++number;
        }
        mean[threadIdx.x] /= numberOfEntities;
        number = 0;
        while(number < numberOfEntities){
            float* pElement = (float*)((char*)dataTable_device + tid * dataTable_pitch) + number;
            *pElement -= mean[threadIdx.x];
            ++number;
        }
    }
}

__global__
void normalizeDataStep2(float* dataTable_device, size_t dataTable_pitch, int numberOfEntities, int numberOfDimension){
    __shared__ float mX[256];
    __shared__ float mXGlobal;
    int tid = threadIdx.x;
    int bias = blockDim.x;

    mX[tid] = 0.0f;
    if(tid == 0){
        mXGlobal = 0.0;
    }
    __syncthreads();

    if(tid < numberOfDimension){
        int number = 0;
        while(number < numberOfEntities){
            float* pElement = (float*)((char*)dataTable_device + tid * dataTable_pitch) + number;
            mX[threadIdx.x] = max(abs(*pElement), mX[threadIdx.x]);
            ++number;
        }
        tid += bias;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        for(int i=0 ; i<256 ; ++i){
            mXGlobal = max(mXGlobal, mX[i]);
        }
    }
    tid = threadIdx.x;
    __syncthreads();

    if(tid < numberOfDimension){
        int number = 0;
        while(number < numberOfEntities){
            float* pElement = (float*)((char*)dataTable_device + tid * dataTable_pitch) + number;
            *pElement /= mXGlobal;
            ++number;
        }
        tid += bias;
    }
}

__global__
void findGraphTraversalStartPoint(int* graphTraversalStartPoint_device, int numberOfEntities, int numberOfNeighbors,
                                  TreeNode** trees_device, TreeNodeLeaf** treesLeafs_device, int* trees_size_device,
                                  int* treesLeafs_size_device, int numberOfTrees){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        for(int tree=0 ; tree<numberOfTrees ; ++tree){
            TreeNodeLeaf treeNodeLeaf = treesLeafs_device[tree][tid];
            int parent = treeNodeLeaf.parent;
            int entityNumber = treeNodeLeaf.entityNumber;
            int numberOfElementsInNode = trees_device[tree][parent].numberOfEntities;
            while(numberOfElementsInNode < (numberOfNeighbors+1)){
                parent = trees_device[tree][parent].parent;
                numberOfElementsInNode = trees_device[tree][parent].numberOfEntities;
            }
            graphTraversalStartPoint_device[tree*numberOfEntities+entityNumber] = parent;
        }
    }
}

__global__
void findInitialStateOfAproximatedEuclideanKNN(int* graphTraversalStartPoint_device, int numberOfEntities, int numberOfNeighbors,
                                               TreeNode** trees_device, TreeNodeLeaf** treesLeafs_device, int* trees_size_device,
                                               int* treesLeafs_size_device, int numberOfTrees, float* dataTable_device, size_t dataTable_Pitch,
                                               float* neighboursDistance_device, int* neighboursId_device, char* idxChecking_device,
                                               int dimensionOfEntity, int start, int end, int* indexes_device, float* distances_device,
                                               int dimensionOfIndexesAndDistances, short* marker_device, int minSize){
    __shared__ float entity[256];
    __shared__ int hist[256];
    __shared__ int elementsToCheckInTreesLeafs_device;
    __shared__ int startPointForTreesLeafs_device;

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ int idOfBatch;
    __shared__ int elementsPerBatch;
    __shared__ int startOfTheBatch;
    __shared__ int endOfTheBatch;
    __shared__ int idxCheckingIdGlobal;
    __shared__ int idxCheckingIdGlobal2;

    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ double minimalSizeOfBucket;
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    if(threadIdx.x == 0){
        idOfBatch = blockIdx.x;
        elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
        startOfTheBatch = elementsPerBatch*idOfBatch;
        endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
        startOfTheBatch += start;
        endOfTheBatch += start;
        idxCheckingIdGlobal = idOfBatch*numberOfEntities;
        idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;
        minimalSizeOfBucket = 0.0078125;
    }
    __syncthreads();

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy liczby do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=threadIdx.x ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int treeNo=0 ; treeNo<numberOfTrees ; ++treeNo){
            //Dla danego drzewa wyszukujemy pukty, dla ktorych bedziemy szukac poczatku danych w treesLeafs_device
            if(threadIdx.x == 0){
                int startPoint = graphTraversalStartPoint_device[treeNo*numberOfEntities+i];
                TreeNode treeNode = trees_device[treeNo][startPoint];
                elementsToCheckInTreesLeafs_device = treeNode.numberOfEntities;
                while(treeNode.rightChild != EMPTY_DIRECTION){
                    treeNode = trees_device[treeNo][treeNode.leftChild];
                }
                startPointForTreesLeafs_device = treeNode.leftChild;
            }
            __syncthreads();

            //Ustawiamy te liczby, dla ktorych mamy liczyc (te ktore wyszukalismy w treesLeafs_device)
            for(int kk=threadIdx.x ; kk<elementsToCheckInTreesLeafs_device ; kk+=blockDim.x){
                TreeNodeLeaf tnl = treesLeafs_device[treeNo][kk+startPointForTreesLeafs_device];
                int elem = tnl.entityNumber;
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe dla ktorej robimy poszukiwania
        if(threadIdx.x == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=threadIdx.x ; kk<numberOfEntities ; kk+=blockDim.x){
            char idxPtr = idxChecking_device[idxCheckingIdGlobal+kk];
            if(idxPtr == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=threadIdx.x ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(threadIdx.x == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = threadIdx.x;
            while(localTid < numberOfEntities){
                char idxPtr = idxChecking_device[idxCheckingIdGlobal+localTid];
                if(idxPtr == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((threadIdx.x < 256)&&(d+threadIdx.x < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+threadIdx.x) * dataTable_Pitch) + i;
                    entity[threadIdx.x] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distance = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distance += (entity[k-d]-pElementVal)*(entity[k-d]-pElementVal);
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distance;
                }
                __syncthreads();
            }

            biggestNumber[threadIdx.x] = 0.0f;
            smalestNumber[threadIdx.x] = STUB_INIT_DIST;
            __syncthreads();

            for(int lp = numberOfNeighbors+threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
                distances_device[idxCheckingIdGlobal2+lp] = dist;
            }
            __syncthreads();

            for(int lp = threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                biggestNumber[threadIdx.x] = max(biggestNumber[threadIdx.x], ceil(dist));
                smalestNumber[threadIdx.x] = min(smalestNumber[threadIdx.x], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(threadIdx.x < 32){
                for(int ii=threadIdx.x ; ii<256 ; ii+=32){
                    biggestNumber[threadIdx.x] = max(biggestNumber[threadIdx.x], biggestNumber[ii]);
                    smalestNumber[threadIdx.x] = min(smalestNumber[threadIdx.x], smalestNumber[ii]);
                }
            }
            if(threadIdx.x == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(threadIdx.x == 0){
                bias = smalestNumber[0];
                minValue = 0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[threadIdx.x] = 0;
                if(threadIdx.x == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short marker = marker_device[idxCheckingIdGlobal2+lp];
                    if(marker == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(threadIdx.x == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[threadIdx.x]+alreadyFoundNumbers)>numberOfNeighbors){
                    atomicMin(&interestingBucket, threadIdx.x);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((threadIdx.x == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[threadIdx.x] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((threadIdx.x == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short marker = marker_device[idxCheckingIdGlobal2+lp];
                    if((marker < interestingBucket)&&(marker >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((marker > interestingBucket)&&(marker < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(marker == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(threadIdx.x == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //przepisz rozwiazanie wynikowe przy wyszukiwaniu najblizszych liczb
            for(int lp = threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short marker = marker_device[idxCheckingIdGlobal2+lp];
                if(marker == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                    neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = threadIdx.x ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short marker = marker_device[idxCheckingIdGlobal2+lp];
                    if(marker == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                            neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__
void findInitialStateOfAproximatedTaxicabKNN(int* graphTraversalStartPoint_device, int numberOfEntities, int numberOfNeighbors,
                                      TreeNode** trees_device, TreeNodeLeaf** treesLeafs_device, int* trees_size_device, int* treesLeafs_size_device, int numberOfTrees,
                                      float* dataTable_device, size_t dataTable_Pitch, float* neighboursDistance_device, int* neighboursId_device,
                                      char* idxChecking_device, int dimensionOfEntity, int start, int end,
                                      int* indexes_device, float* distances_device, int dimensionOfIndexesAndDistances, short* marker_device, int minSize){
    __shared__ int startPointForTreesLeafs_device;
    __shared__ int elementsToCheckInTreesLeafs_device;
    __shared__ float entity[256];

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ int hist[256];
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
    startOfTheBatch += start;
    endOfTheBatch += start;
    int idxCheckingIdGlobal = idOfBatch*numberOfEntities;
    int idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;

    double minimalSizeOfBucket = 0.000000059604644775;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy bity do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=tid ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int treeNo=0 ; treeNo<numberOfTrees ; ++treeNo){
            //Dla danego drzewa wyszukujemy pukty, dla ktorych bedziemy szukac poczatku danych w treesLeafs_device
            if(tid == 0){
                TreeNode treeNode = trees_device[treeNo][graphTraversalStartPoint_device[treeNo*numberOfEntities+i]];
                elementsToCheckInTreesLeafs_device = treeNode.numberOfEntities;
                while(treeNode.rightChild != EMPTY_DIRECTION){
                    treeNode = trees_device[treeNo][treeNode.leftChild];
                }
                startPointForTreesLeafs_device = treeNode.leftChild;
            }
            __syncthreads();

            //Ustawiamy te bity, dla ktorych mamy liczyc (te ktore wyszukalismy w treesLeafs_device)
            for(int kk=tid ; kk<elementsToCheckInTreesLeafs_device ; kk+=blockDim.x){
                int elem = treesLeafs_device[treeNo][kk+startPointForTreesLeafs_device].entityNumber;
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe dla ktorej robimy poszukiwania
        if(tid == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=tid ; kk<numberOfEntities ; kk+=blockDim.x){
            char idxPtr2 = idxChecking_device[idxCheckingIdGlobal+kk];
            if(idxPtr2 == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(tid == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = tid;
            while(localTid < numberOfEntities){
                char idxPtr2 = idxChecking_device[idxCheckingIdGlobal+localTid];
                if(idxPtr2 == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((tid < 256)&&(d+tid < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                    entity[tid] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distance = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distance += abs(entity[k-d]-pElementVal);
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distance;
                }
                __syncthreads();
            }
            biggestNumber[tid] = 0.0f;
            smalestNumber[tid] = STUB_INIT_DIST;

            __syncthreads();

            for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
                distances_device[idxCheckingIdGlobal2+lp] = dist;
            }
            __syncthreads();

            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                biggestNumber[tid] = max(biggestNumber[tid], ceil(dist));
                smalestNumber[tid] = min(smalestNumber[tid], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(tid < 32){
                for(int ii=tid ; ii<256 ; ii+=32){
                    biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[ii]);
                    smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[ii]);
                }
            }
            if(tid == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(tid == 0){
                bias = smalestNumber[0];
                minValue = 0.0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[tid] = 0;
                if(tid == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(tid == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[tid]+alreadyFoundNumbers) > numberOfNeighbors){
                    atomicMin(&interestingBucket, tid);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((tid == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[tid] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if((mark < interestingBucket)&&(mark >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((mark > interestingBucket)&&(mark < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(mark == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(tid == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //Wpisujemy k najmniejsze liczby jako nowe rozwiazanie do neighbours
            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short mark = marker_device[idxCheckingIdGlobal2+lp];
                if(mark == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                    neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                            neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__
void findInitialStateOfAproximatedCosineKNN(int* graphTraversalStartPoint_device, int numberOfEntities, int numberOfNeighbors,
                                      TreeNode** trees_device, TreeNodeLeaf** treesLeafs_device, int* trees_size_device, int* treesLeafs_size_device, int numberOfTrees,
                                      float* dataTable_device, size_t dataTable_Pitch, float* neighboursDistance_device, int* neighboursId_device,
                                      char* idxChecking_device, int dimensionOfEntity, int start, int end,
                                      int* indexes_device, float* distances_device, float* distances2_device, float* distances3_device, int dimensionOfIndexesAndDistances,
                                      short* marker_device, int minSize){
    __shared__ int startPointForTreesLeafs_device;
    __shared__ int elementsToCheckInTreesLeafs_device;
    __shared__ float entity[256];

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ int hist[256];
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
    startOfTheBatch += start;
    endOfTheBatch += start;
    int idxCheckingIdGlobal = idOfBatch*numberOfEntities;
    int idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;

    double minimalSizeOfBucket = 0.000000059604644775;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy bity do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=tid ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int treeNo=0 ; treeNo<numberOfTrees ; ++treeNo){
            //Dla danego drzewa wyszukujemy pukty, dla ktorych bedziemy szukac poczatku danych w treesLeafs_device
            if(tid == 0){
                TreeNode treeNode = trees_device[treeNo][graphTraversalStartPoint_device[treeNo*numberOfEntities+i]];
                elementsToCheckInTreesLeafs_device = treeNode.numberOfEntities;
                while(treeNode.rightChild != EMPTY_DIRECTION){
                    treeNode = trees_device[treeNo][treeNode.leftChild];
                }
                startPointForTreesLeafs_device = treeNode.leftChild;
            }
            __syncthreads();

            //Ustawiamy te bity, dla ktorych mamy liczyc (te ktore wyszukalismy w treesLeafs_device)
            for(int kk=tid ; kk<elementsToCheckInTreesLeafs_device ; kk+=blockDim.x){
                int elem = treesLeafs_device[treeNo][kk+startPointForTreesLeafs_device].entityNumber;
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe dla ktorej robimy poszukiwania
        if(tid == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=tid ; kk<numberOfEntities ; kk+=blockDim.x){
            char idxPtr2 = idxChecking_device[idxCheckingIdGlobal+kk];
            if(idxPtr2 == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(tid == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = tid;
            while(localTid < numberOfEntities){
                char idxPtr2 = idxChecking_device[idxCheckingIdGlobal+localTid];
                if(idxPtr2 == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        distances2_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        distances3_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((tid < 256)&&(d+tid < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                    entity[tid] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distanceAB = 0.0;
                    float distanceA = 0.0;
                    float distanceB = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distanceAB += entity[k-d]*pElementVal;
                        distanceA += entity[k-d]*entity[k-d];
                        distanceB += pElementVal*pElementVal;
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distanceAB;
                    distances2_device[idxCheckingIdGlobal2+lp] += distanceA;
                    distances3_device[idxCheckingIdGlobal2+lp] += distanceB;
                }
                __syncthreads();
            }

            biggestNumber[tid] = 0.0f;
            smalestNumber[tid] = STUB_INIT_DIST;
            __syncthreads();

            //wyznaczanie odleglosci do liczb najblizszych
            for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float distanceAB = distances_device[idxCheckingIdGlobal2+lp];
                float distanceA = distances2_device[idxCheckingIdGlobal2+lp];
                float distanceB = distances3_device[idxCheckingIdGlobal2+lp];
                float distance = distanceAB/(sqrt(distanceA)*sqrt(distanceB));
                distance = (-1.0*distance)+1.0;
                distances_device[idxCheckingIdGlobal2+lp] = distance;
            }
            __syncthreads();

            //for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                //float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
                //distances_device[idxCheckingIdGlobal2+lp] = dist;
            //}
            //__syncthreads();

            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                //dist = (-1.0*dist)+1.0;
                //distances_device[idxCheckingIdGlobal2+lp] = dist;
                biggestNumber[tid] = max(biggestNumber[tid], ceil(dist));
                smalestNumber[tid] = min(smalestNumber[tid], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(tid < 32){
                for(int ii=tid ; ii<256 ; ii+=32){
                    biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[ii]);
                    smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[ii]);
                }
            }
            if(tid == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(tid == 0){
                bias = smalestNumber[0];
                minValue = 0.0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[tid] = 0;
                if(tid == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(tid == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[tid]+alreadyFoundNumbers) > numberOfNeighbors){
                    atomicMin(&interestingBucket, tid);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((tid == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[tid] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if((mark < interestingBucket)&&(mark >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((mark > interestingBucket)&&(mark < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(mark == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(tid == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //Wpisujemy k najmniejsze liczby jako nowe rozwiazanie do neighbours
            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short mark = marker_device[idxCheckingIdGlobal2+lp];
                if(mark == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance_device[i*numberOfNeighbors+id] = -1.0*(distances_device[idxCheckingIdGlobal2+lp]-1.0);
                    neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance_device[i*numberOfNeighbors+id] = -1.0*(distances_device[idxCheckingIdGlobal2+lp]-1.0);
                            neighboursId_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__
void propagateEuclideanKernel(int numberOfEntities, int numberOfNeighbors, float* dataTable_device, size_t dataTable_Pitch,
                     float* neighboursDistance_device, float* neighboursDistance2_device, int* neighboursId_device, int* neighboursId2_device,
                     char* idxChecking_device, int dimensionOfEntity, int start, int end,
                     int* indexes_device, float* distances_device, int dimensionOfIndexesAndDistances, short* marker_device){
    __shared__ float entity[256];

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ int hist[256];
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
    startOfTheBatch += start;
    endOfTheBatch += start;
    int idxCheckingIdGlobal = idOfBatch*numberOfEntities;
    int idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;

    double minimalSizeOfBucket = 0.000000059604644775;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy bity do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=tid ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int neighHi = 0 ; neighHi < numberOfNeighbors ; neighHi += 1){
            for(int neighLo = tid ; neighLo < numberOfNeighbors ; neighLo+=blockDim.x){
                int interestingNeighbour = neighboursId_device[i*numberOfNeighbors+neighHi];
                int elem = neighboursId_device[interestingNeighbour*numberOfNeighbors+neighLo];
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe sama ze soba
        if(tid == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
        }
        __syncthreads();

        //Zerujemy bity dla wlasnych sasiadow i przepisujemy aktualnie najblizszych
        for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
            int elem = neighboursId_device[i*numberOfNeighbors+kk];
            idxChecking_device[idxCheckingIdGlobal+elem] = 0x00;
            neighboursId2_device[i*numberOfNeighbors+kk] = elem;
            neighboursDistance2_device[i*numberOfNeighbors+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
        }
        __syncthreads();

        //Liczymy najblizszych
        if(tid == 0){
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=tid ; kk<numberOfEntities ; kk+=blockDim.x){
            if(idxChecking_device[idxCheckingIdGlobal+kk] == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId2_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance2_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(tid == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = tid;
            while(localTid < numberOfEntities){
                if(idxChecking_device[idxCheckingIdGlobal+localTid] == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((tid < 256)&&(d+tid < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                    entity[tid] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distance = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distance += (entity[k-d]-pElementVal)*(entity[k-d]-pElementVal);
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distance;
                }
                __syncthreads();
            }

            biggestNumber[tid] = 0.0f;
            smalestNumber[tid] = STUB_INIT_DIST;
            __syncthreads();

            for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
                distances_device[idxCheckingIdGlobal2+lp] = dist;
            }
            __syncthreads();

            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                biggestNumber[tid] = max(biggestNumber[tid], ceil(dist));
                smalestNumber[tid] = min(smalestNumber[tid], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(tid < 32){
                for(int ii=tid ; ii<256 ; ii+=32){
                    biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[ii]);
                    smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[ii]);
                }
            }
            if(tid == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(tid == 0){
                bias = smalestNumber[0];
                minValue = 0.0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[tid] = 0;
                if(tid == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(tid == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[tid]+alreadyFoundNumbers) > numberOfNeighbors){
                    atomicMin(&interestingBucket, tid);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((tid == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[tid] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if((mark < interestingBucket)&&(mark >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((mark > interestingBucket)&&(mark < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(mark == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(tid == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //Wpisujemy k najmniejsze liczby jako nowe rozwiazanie do neighbours
            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short mark = marker_device[idxCheckingIdGlobal2+lp];
                if(mark == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance2_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                    neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance2_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                            neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__
void propagateTaxicabKernel(int numberOfEntities, int numberOfNeighbors, float* dataTable_device, size_t dataTable_Pitch,
                     float* neighboursDistance_device, float* neighboursDistance2_device, int* neighboursId_device, int* neighboursId2_device,
                     char* idxChecking_device, int dimensionOfEntity, int start, int end,
                     int* indexes_device, float* distances_device, int dimensionOfIndexesAndDistances, short* marker_device){
    __shared__ float entity[256];

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ int hist[256];
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
    startOfTheBatch += start;
    endOfTheBatch += start;
    int idxCheckingIdGlobal = idOfBatch*numberOfEntities;
    int idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;

    double minimalSizeOfBucket = 0.000000059604644775;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy bity do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=tid ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int neighHi = 0 ; neighHi < numberOfNeighbors ; neighHi += 1){
            for(int neighLo = tid ; neighLo < numberOfNeighbors ; neighLo+=blockDim.x){
                int interestingNeighbour = neighboursId_device[i*numberOfNeighbors+neighHi];
                int elem = neighboursId_device[interestingNeighbour*numberOfNeighbors+neighLo];
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe sama ze soba
        if(tid == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
        }
        __syncthreads();

        //Zerujemy bity dla wlasnych sasiadow i przepisujemy aktualnie najblizszych
        for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
            int elem = neighboursId_device[i*numberOfNeighbors+kk];
            idxChecking_device[idxCheckingIdGlobal+elem] = 0x00;
            neighboursId2_device[i*numberOfNeighbors+kk] = elem;
            neighboursDistance2_device[i*numberOfNeighbors+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
        }
        __syncthreads();

        //Liczymy najblizszych
        if(tid == 0){
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=tid ; kk<numberOfEntities ; kk+=blockDim.x){
            if(idxChecking_device[idxCheckingIdGlobal+kk] == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId2_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance2_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(tid == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = tid;
            while(localTid < numberOfEntities){
                if(idxChecking_device[idxCheckingIdGlobal+localTid] == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((tid < 256)&&(d+tid < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                    entity[tid] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distance = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distance += abs(entity[k-d]-pElementVal);
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distance;
                }
                __syncthreads();
            }

            biggestNumber[tid] = 0.0f;
            smalestNumber[tid] = STUB_INIT_DIST;
            __syncthreads();

            for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
                distances_device[idxCheckingIdGlobal2+lp] = dist;
            }
            __syncthreads();

            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                biggestNumber[tid] = max(biggestNumber[tid], ceil(dist));
                smalestNumber[tid] = min(smalestNumber[tid], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(tid < 32){
                for(int ii=tid ; ii<256 ; ii+=32){
                    biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[ii]);
                    smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[ii]);
                }
            }
            if(tid == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(tid == 0){
                bias = smalestNumber[0];
                minValue = 0.0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[tid] = 0;
                if(tid == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(tid == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[tid]+alreadyFoundNumbers) > numberOfNeighbors){
                    atomicMin(&interestingBucket, tid);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((tid == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[tid] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if((mark < interestingBucket)&&(mark >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((mark > interestingBucket)&&(mark < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(mark == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(tid == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //Wpisujemy k najmniejsze liczby jako nowe rozwiazanie do neighbours
            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short mark = marker_device[idxCheckingIdGlobal2+lp];
                if(mark == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance2_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                    neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance2_device[i*numberOfNeighbors+id] = distances_device[idxCheckingIdGlobal2+lp];
                            neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__
void propagateCosineKernel(int numberOfEntities, int numberOfNeighbors, float* dataTable_device, size_t dataTable_Pitch,
                     float* neighboursDistance_device, float* neighboursDistance2_device, int* neighboursId_device, int* neighboursId2_device,
                     char* idxChecking_device, int dimensionOfEntity, int start, int end,
                     int* indexes_device, float* distances_device, float* distances2_device, float* distances3_device, int dimensionOfIndexesAndDistances, short* marker_device){
    __shared__ float entity[256];

    __shared__ int numbersToCheck;
    __shared__ int numbersToCheckInThisPart;
    __shared__ int numbersToCheckInThisPartRealPart;

    __shared__ double bias;
    __shared__ double lengthOfBucket;
    __shared__ double maxValue;
    __shared__ double minValue;
    __shared__ int hist[256];
    __shared__ float biggestNumber[256];
    __shared__ float smalestNumber[256];
    __shared__ int foundExactSolution;
    __shared__ int limitOfLengthOfBucketExceeded;
    __shared__ int alreadyFoundNumbers;
    __shared__ int interestingBucket;
    __shared__ int rewrittenNumbers;
    __shared__ int complement;

    int tid = threadIdx.x;
    int idOfBatch = blockIdx.x;
    int elementsPerBatch = ceil(float(end-start)/float(gridDim.x));
    int startOfTheBatch = elementsPerBatch*idOfBatch;
    int endOfTheBatch = elementsPerBatch*(idOfBatch+1) <= (end-start) ? elementsPerBatch*(idOfBatch+1) : (end-start);
    startOfTheBatch += start;
    endOfTheBatch += start;
    int idxCheckingIdGlobal = idOfBatch*numberOfEntities;
    int idxCheckingIdGlobal2 = idOfBatch*dimensionOfIndexesAndDistances;

    double minimalSizeOfBucket = 0.000000059604644775;

    for(int i=startOfTheBatch ; i<endOfTheBatch ; ++i){
        //Zerujemy bity do wyszukiwania dla danego punktu, dla ktorego liczymy
        for(int ii=tid ; ii<numberOfEntities ; ii+=blockDim.x){
            idxChecking_device[idxCheckingIdGlobal+ii] = 0x00;
        }
        __syncthreads();

        //Wyszukujemy liczby do przeszukania
        for(int neighHi = 0 ; neighHi < numberOfNeighbors ; neighHi += 1){
            for(int neighLo = tid ; neighLo < numberOfNeighbors ; neighLo+=blockDim.x){
                int interestingNeighbour = neighboursId_device[i*numberOfNeighbors+neighHi];
                int elem = neighboursId_device[interestingNeighbour*numberOfNeighbors+neighLo];
                idxChecking_device[idxCheckingIdGlobal+elem] = 0x01;
            }
            __syncthreads();
        }
        __syncthreads();

        //Zerujemy bit odpowiedzialny za liczbe sama ze soba
        if(tid == 0){
            idxChecking_device[idxCheckingIdGlobal+i] = 0x00;
        }
        __syncthreads();

        //Zerujemy bity dla wlasnych sasiadow i przepisujemy aktualnie najblizszych
        for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
            int elem = neighboursId_device[i*numberOfNeighbors+kk];
            idxChecking_device[idxCheckingIdGlobal+elem] = 0x00;
            neighboursId2_device[i*numberOfNeighbors+kk] = elem;
            neighboursDistance2_device[i*numberOfNeighbors+kk] = neighboursDistance_device[i*numberOfNeighbors+kk];
        }
        __syncthreads();

        //Liczymy najblizszych
        if(tid == 0){
            numbersToCheck = 0;
        }
        __syncthreads();

        for(int kk=tid ; kk<numberOfEntities ; kk+=blockDim.x){
            if(idxChecking_device[idxCheckingIdGlobal+kk] == 0x01){
                atomicAdd(&numbersToCheck, 1);
            }
        }
        __syncthreads();

        //Przepisujemy te liczby do tablicy z wyszukiwaniem najblizszych sasiadow
        while(numbersToCheck > 0){
            __syncthreads();

            //Przepisujemy aktualne najblizsze liczby
            for(int kk=tid ; kk<numberOfNeighbors ; kk+=blockDim.x){
                indexes_device[idxCheckingIdGlobal2+kk] = neighboursId2_device[i*numberOfNeighbors+kk];
                distances_device[idxCheckingIdGlobal2+kk] = neighboursDistance2_device[i*numberOfNeighbors+kk];
                marker_device[idxCheckingIdGlobal2+kk] = 0;
            }

            //Dopisujemy te co aktualnie sprawdzamy
            if(tid == 0){
                numbersToCheck = 0;
                numbersToCheckInThisPart = numberOfNeighbors;
                numbersToCheckInThisPartRealPart = numberOfNeighbors;
            }
            __syncthreads();

            int localTid = tid;
            while(localTid < numberOfEntities){
                if(idxChecking_device[idxCheckingIdGlobal+localTid] == 0x01){
                    int pos = atomicAdd(&numbersToCheckInThisPart, 1);
                    if(pos < dimensionOfIndexesAndDistances){
                        indexes_device[idxCheckingIdGlobal2+pos] = localTid;
                        distances_device[idxCheckingIdGlobal2+pos] = 0.0f;
                        marker_device[idxCheckingIdGlobal2+pos] = 0;
                        idxChecking_device[idxCheckingIdGlobal+localTid] = 0x00;
                        atomicAdd(&numbersToCheckInThisPartRealPart, 1);
                    }else{
                        atomicAdd(&numbersToCheck, 1);
                    }
                }
                localTid += blockDim.x;
            }
            __syncthreads();

            //Wyznaczamy odleglosc do tych nowych liczb
            for(int d=0 ; d<dimensionOfEntity ; d+=256){
                //wczytaj liczbe dla ktorej bedziemy liczyc odleglosci do innych liczb
                if((tid < 256)&&(d+tid < dimensionOfEntity)){
                    float* pElement = (float*)((char*)dataTable_device + (d+tid) * dataTable_Pitch) + i;
                    entity[tid] = *pElement;
                }
                __syncthreads();

                //wyznaczanie odleglosci do liczb
                for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    float distanceAB = 0.0;
                    float distanceA = 0.0;
                    float distanceB = 0.0;
                    for(int k=d ; (k<dimensionOfEntity)&&(k<(d+256)) ; ++k){
                        int lpp = indexes_device[idxCheckingIdGlobal2+lp];
                        float* pElement = (float*)((char*)dataTable_device + k * dataTable_Pitch) + lpp;
                        float pElementVal = *pElement;
                        distanceAB += entity[k-d]*pElementVal;
                        distanceA += entity[k-d]*entity[k-d];
                        distanceB += pElementVal*pElementVal;
                    }
                    //zapisanie odleglosci do tablicy na bazie ktorej beda wyszukiwani najblizsi sasiedzi
                    distances_device[idxCheckingIdGlobal2+lp] += distanceAB;
                    distances2_device[idxCheckingIdGlobal2+lp] += distanceA;
                    distances3_device[idxCheckingIdGlobal2+lp] += distanceB;
                }
                __syncthreads();
            }

            biggestNumber[tid] = 0.0f;
            smalestNumber[tid] = STUB_INIT_DIST;
            __syncthreads();

            //wyznaczanie odleglosci do liczb najblizszych
            for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float distanceAB = distances_device[idxCheckingIdGlobal2+lp];
                float distanceA = distances2_device[idxCheckingIdGlobal2+lp];
                float distanceB = distances3_device[idxCheckingIdGlobal2+lp];
                float distance = distanceAB/(sqrt(distanceA)*sqrt(distanceB));
                distance = (-1.0*distance)+1.0;
                distances_device[idxCheckingIdGlobal2+lp] = distance;
            }
            __syncthreads();

            //for(int lp = numberOfNeighbors+tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
            //    float dist = sqrt(distances_device[idxCheckingIdGlobal2+lp]);
            //    distances_device[idxCheckingIdGlobal2+lp] = dist;
            //}
            //__syncthreads();

            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                float dist = distances_device[idxCheckingIdGlobal2+lp];
                //dist = (-1.0*dist)+1.0;
                //distances_device[idxCheckingIdGlobal2+lp] = dist;
                biggestNumber[tid] = max(biggestNumber[tid], ceil(dist));
                smalestNumber[tid] = min(smalestNumber[tid], floor(dist));
            }
            __syncthreads();

            //wyszukiwanie najwiekszej liczby w rezultacie
            if(tid < 32){
                for(int ii=tid ; ii<256 ; ii+=32){
                    biggestNumber[tid] = max(biggestNumber[tid], biggestNumber[ii]);
                    smalestNumber[tid] = min(smalestNumber[tid], smalestNumber[ii]);
                }
            }
            if(tid == 0){
                #pragma unroll
                for(int c=0 ; c<32 ; ++c){
                    biggestNumber[0] = max(biggestNumber[0], biggestNumber[c]);
                    smalestNumber[0] = min(smalestNumber[0], smalestNumber[c]);
                }
            }
            __syncthreads();

            //Wyszukujemy k najmniejszych liczb
            if(tid == 0){
                bias = smalestNumber[0];
                minValue = 0.0;
                maxValue = biggestNumber[0] - smalestNumber[0];
                maxValue = pow(2.0, ceil(log(maxValue+1.0)/log(2.0)));
                lengthOfBucket = (maxValue-minValue)/256.0;
                foundExactSolution = FALSE;
                limitOfLengthOfBucketExceeded = FALSE;
                alreadyFoundNumbers = 0;
                rewrittenNumbers = 0;
                complement = 0;
            }
            __syncthreads();

            while((foundExactSolution == FALSE) && (limitOfLengthOfBucketExceeded == FALSE)){
                hist[tid] = 0;
                if(tid == 0){
                    interestingBucket = NON_OF_BUCKET_IN_INTEREST;
                }
                __syncthreads();

                //wyznacz histogram dla aktualnego opisu minValue-maxValue
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int idOfBucketInHist = (distances_device[idxCheckingIdGlobal2+lp]-bias-minValue)/lengthOfBucket;
                        atomicAdd(&hist[idOfBucketInHist], 1);
                        marker_device[idxCheckingIdGlobal2+lp] = idOfBucketInHist;
                    }
                }
                __syncthreads();

                //zsumuj histogram tak, ze hist(i) to suma od hist(0) do hist(i)
                if(tid == 0){
                    for(int k=1 ; k<256 ; ++k){
                        hist[k] += hist[k-1];
                    }
                }
                __syncthreads();

                if((hist[tid]+alreadyFoundNumbers) > numberOfNeighbors){
                    atomicMin(&interestingBucket, tid);
                }

                //jezeli znalezlismy dokladna liczbe to koncz
                if((tid == 0) && (alreadyFoundNumbers == numberOfNeighbors)){
                    foundExactSolution = TRUE;
                }

                //Sprawdzamy czy nie znalezlismy juz rozwiazania przyblizonego
                int tmpSum = hist[tid] + alreadyFoundNumbers;
                if(tmpSum == numberOfNeighbors){
                    foundExactSolution = TRUE;
                }

                //sprawdzamy czy czasami nie osigniemy juz zbyt malej szerokosci kubelka
                if((tid == 0) && (lengthOfBucket < minimalSizeOfBucket)){
                    limitOfLengthOfBucketExceeded = TRUE;
                }
                __syncthreads();

                //dla tych kubelkow z id>interestingBucket zaznaczamy, że nie sa interesujace, a dla id<interestingBucket ze sa w rozwiazaniu, dla id==interestingBucket, do rozpatrzenia w nastepnej iteracji
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if((mark < interestingBucket)&&(mark >= 0)){
                        marker_device[idxCheckingIdGlobal2+lp] = IN_SOLUTION;
                        atomicAdd(&alreadyFoundNumbers, 1);
                    }else if((mark > interestingBucket)&&(mark < 256)){
                        marker_device[idxCheckingIdGlobal2+lp] = OUT_OF_SOLUTION;
                    }else if(mark == interestingBucket){
                        marker_device[idxCheckingIdGlobal2+lp] = 0;
                    }
                }
                __syncthreads();

                //przeliczenie zakresow
                if(tid == 0){
                    bias = bias+interestingBucket*lengthOfBucket;
                    minValue = 0.0;
                    maxValue = lengthOfBucket;
                    lengthOfBucket = (maxValue-minValue)/256.0;
                }
                __syncthreads();
            }
            __syncthreads();

            //Wpisujemy k najmniejsze liczby jako nowe rozwiazanie do neighbours
            for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                short mark = marker_device[idxCheckingIdGlobal2+lp];
                if(mark == IN_SOLUTION){
                    int id = atomicAdd(&rewrittenNumbers, 1);
                    neighboursDistance2_device[i*numberOfNeighbors+id] = -1.0*(distances_device[idxCheckingIdGlobal2+lp]-1.0);
                    neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                }
            }
            __syncthreads();

            //jezeli zostal przekroczony limit kubelka to znajdz odpowiednie liczby dla dopelnienia rezultatu dla najblizszych liczb
            if((limitOfLengthOfBucketExceeded == TRUE)&&(foundExactSolution == FALSE)){
                for(int lp = tid ; lp<numbersToCheckInThisPartRealPart ; lp+=blockDim.x){
                    short mark = marker_device[idxCheckingIdGlobal2+lp];
                    if(mark == 0){
                        int id2 = atomicAdd(&complement, 1);
                        if((id2+alreadyFoundNumbers) < numberOfNeighbors){
                            int id = atomicAdd(&rewrittenNumbers, 1);
                            neighboursDistance2_device[i*numberOfNeighbors+id] = -1.0*(distances_device[idxCheckingIdGlobal2+lp]-1.0);
                            neighboursId2_device[i*numberOfNeighbors+id] = indexes_device[idxCheckingIdGlobal2+lp];
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

void EuclideanDistanceMatrixGPU::propagate(){
    for(int prop=0 ; prop<numberOfPropagations ; ++prop){
        cuCall(cudaSetDevice(device));

        std::cout<<"Urzadzenie "<<device<<" uruchamia zadanie propagacji dla punktow: "<<partition.start<<" - "<<partition.end-1<<" dla iteracji: "<<prop+1<<"\n";
        dim3 grid2(this->numberOfMultiprocessors*this->numberOfBlocksPerMultiprocessors, 1);
        dim3 block2(256, 1);

        if(typeOfDistance == DISTANCE_EUCLIDEAN){
            propagateEuclideanKernel<<<grid2, block2, 0, executionStreams>>>(numberOfEntities, numberOfNeighbors, dataTable_device, dataTable_Pitch,
                                                                    neighboursDistance_device, neighboursDistance2_device, neighboursId_device, neighboursId2_device,
                                                                    idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                    indexes_device, distances_device, dimensionOfIndexesAndDistances, marker_device);
        }else if(typeOfDistance == DISTANCE_TAXICAB){
            propagateTaxicabKernel<<<grid2, block2, 0, executionStreams>>>(numberOfEntities, numberOfNeighbors, dataTable_device, dataTable_Pitch,
                                                                    neighboursDistance_device, neighboursDistance2_device, neighboursId_device, neighboursId2_device,
                                                                    idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                    indexes_device, distances_device, dimensionOfIndexesAndDistances, marker_device);
        }else if(typeOfDistance == DISTANCE_COSINE){
            propagateCosineKernel<<<grid2, block2, 0, executionStreams>>>(numberOfEntities, numberOfNeighbors, dataTable_device, dataTable_Pitch,
                                                                    neighboursDistance_device, neighboursDistance2_device, neighboursId_device, neighboursId2_device,
                                                                    idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                    indexes_device, distances_device, distances2_device, distances3_device, dimensionOfIndexesAndDistances, marker_device);
        }else{
            std::cout<<"We do not have such type of distance\n";
        }
        cuCall(cudaStreamSynchronize(executionStreams));
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err){
            std::cout<<"propagateKernel: "<<cudaGetErrorString(err)<<"\n";
        }

        float* tmp1 = neighboursDistance_device;
        neighboursDistance_device = neighboursDistance2_device;
        neighboursDistance2_device = tmp1;

        int* tmp2 = neighboursId_device;
        neighboursId_device = neighboursId2_device;
        neighboursId2_device = tmp2;
    }
}

bool compareByLengthMin(const DataPoint &a, const DataPoint &b){
    return a.distance < b.distance;
}

bool compareByLengthMinCosine(const DataPoint &a, const DataPoint &b){
    return a.distance > b.distance;
}

std::string trim(std::string const& str){
    if(str.empty())
        return str;

    std::size_t firstScan = str.find_first_not_of(' ');
    std::size_t first     = firstScan == std::string::npos ? str.length() : firstScan;
    std::size_t last      = str.find_last_not_of(' ');
    return str.substr(first, last-first+1);
}

bool EuclideanDistanceMatrixGPU::initilizeGPUStructuresForTrees(){
    bool error = false;

    error |= cuCall(cudaMalloc((void**)&trees_device, numberOfTrees*sizeof(TreeNode*)));
    error |= cuCall(cudaHostAlloc((void**)&trees_device_pointer_for_cpu, numberOfTrees*sizeof(TreeNode*), cudaHostAllocPortable));
    for(int i=0 ; i<numberOfTrees ; ++i){
        int elems = trees_host[i].size();
        error |= cuCall(cudaMalloc((void**)&trees_device_pointer_for_cpu[i], elems*sizeof(TreeNode)));
    }
    error |= cuCall(cudaMemcpy((void*)trees_device, (void*)trees_device_pointer_for_cpu, numberOfTrees*sizeof(TreeNode*), cudaMemcpyHostToDevice));

    error |= cuCall(cudaMalloc((void**)&treesLeafs_device, numberOfTrees*sizeof(TreeNodeLeaf*)));
    error |= cuCall(cudaHostAlloc((void**)&treesLeafs_device_pointer_for_cpu, numberOfTrees*sizeof(TreeNodeLeaf*), cudaHostAllocPortable));
    for(int i=0 ; i<numberOfTrees ; ++i){
        int elems = treesLeafs_host[i].size();
        error |= cuCall(cudaMalloc((void**)&treesLeafs_device_pointer_for_cpu[i], elems*sizeof(TreeNodeLeaf)));
    }
    error |= cuCall(cudaMemcpy((void*)treesLeafs_device, (void*)treesLeafs_device_pointer_for_cpu, numberOfTrees*sizeof(TreeNodeLeaf*), cudaMemcpyHostToDevice));

    error |= cuCall(cudaMalloc((void**)&trees_size_device, numberOfTrees*sizeof(int)));
    error |= cuCall(cudaHostAlloc((void**)&trees_size_host, numberOfTrees*sizeof(int), cudaHostAllocPortable));

    error |= cuCall(cudaMalloc((void**)&treesLeafs_size_device, numberOfTrees*sizeof(int)));
    error |= cuCall(cudaHostAlloc((void**)&treesLeafs_size_host, numberOfTrees*sizeof(int), cudaHostAllocPortable));

    //Przekopiowanie vektorow
    for(int i=0 ; i<numberOfTrees ; ++i){
        error |= cuCall(cudaMemcpyAsync((void*)trees_device_pointer_for_cpu[i], (void*)trees_host[i].data(),
                                        trees_host[i].size()*sizeof(TreeNode), cudaMemcpyHostToDevice, executionStreams));
        error |= cuCall(cudaMemcpyAsync((void*)treesLeafs_device_pointer_for_cpu[i], (void*)treesLeafs_host[i].data(),
                                        treesLeafs_host[i].size()*sizeof(TreeNodeLeaf), cudaMemcpyHostToDevice, executionStreams));
        trees_size_host[i] = trees_host[i].size();
        treesLeafs_size_host[i] = treesLeafs_host[i].size();
    }
    error |= cuCall(cudaMemcpyAsync((void*)trees_size_device, (void*)trees_size_host,
                                    numberOfTrees*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
    error |= cuCall(cudaMemcpyAsync((void*)treesLeafs_size_device, (void*)treesLeafs_size_host,
                                    numberOfTrees*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
    error |= cuCall(cudaStreamSynchronize(executionStreams));

    return error;
}

bool EuclideanDistanceMatrixGPU::deinitializeGPUStructuresForTrees(){
    bool error = false;

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaDeviceSynchronize());

    for(int i=0 ; i<numberOfTrees ; ++i){
        error |= cuCall(cudaFree((void*)trees_device_pointer_for_cpu[i]));
    }
    error |= cuCall(cudaFree((void*)trees_device));
    error |= cuCall(cudaFreeHost((void*)trees_device_pointer_for_cpu));

    for(int i=0 ; i<numberOfTrees ; ++i){
        error |= cuCall(cudaFree((void*)treesLeafs_device_pointer_for_cpu[i]));
    }
    error |= cuCall(cudaFree((void*)treesLeafs_device));
    error |= cuCall(cudaFreeHost((void*)treesLeafs_device_pointer_for_cpu));

    error |= cuCall(cudaFree((void*)trees_size_device));
    error |= cuCall(cudaFreeHost((void*)trees_size_host));

    error |= cuCall(cudaFree((void*)treesLeafs_size_device));
    error |= cuCall(cudaFreeHost((void*)treesLeafs_size_host));

    return error;
}

void EuclideanDistanceMatrixGPU::findInitialKNN(){
    dim3 grid1(ceil(float(numberOfEntities)/256.0), 1);
    dim3 block1(256, 1);
    findGraphTraversalStartPoint<<<grid1, block1, 0, executionStreams>>>(graphTraversalStartPoint_device, numberOfEntities, numberOfNeighbors,
                                                                         trees_device, treesLeafs_device, trees_size_device, treesLeafs_size_device, numberOfTrees);
    cuCall(cudaStreamSynchronize(executionStreams));
    cudaError_t err1 = cudaGetLastError();
    if (cudaSuccess != err1){
        std::cout<<"findGraphTraversalStartPoint: "<<cudaGetErrorString(err1)<<"\n";
    }

    std::cout<<"Urzadzenie "<<device<<" uruchamia zadanie inicjalizacji kNN dla punktow: "<<partition.start<<" - "<<partition.end-1<<"\n";
    dim3 grid2(this->numberOfMultiprocessors*this->numberOfBlocksPerMultiprocessors, 1);
    dim3 block2(256, 1);
    if(typeOfDistance == DISTANCE_EUCLIDEAN){
        findInitialStateOfAproximatedEuclideanKNN<<<grid2, block2, 0, executionStreams>>>(graphTraversalStartPoint_device, numberOfEntities, numberOfNeighbors,
                                                                                 trees_device, treesLeafs_device, trees_size_device, treesLeafs_size_device, numberOfTrees,
                                                                                 dataTable_device, dataTable_Pitch, neighboursDistance_device, neighboursId_device,
                                                                                 idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                                 indexes_device, distances_device, dimensionOfIndexesAndDistances, marker_device, minSize);
    }else if(typeOfDistance == DISTANCE_TAXICAB){
        findInitialStateOfAproximatedTaxicabKNN<<<grid2, block2, 0, executionStreams>>>(graphTraversalStartPoint_device, numberOfEntities, numberOfNeighbors,
                                                                                 trees_device, treesLeafs_device, trees_size_device, treesLeafs_size_device, numberOfTrees,
                                                                                 dataTable_device, dataTable_Pitch, neighboursDistance_device, neighboursId_device,
                                                                                 idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                                 indexes_device, distances_device, dimensionOfIndexesAndDistances, marker_device, minSize);
    }else if(typeOfDistance == DISTANCE_COSINE){
        findInitialStateOfAproximatedCosineKNN<<<grid2, block2, 0, executionStreams>>>(graphTraversalStartPoint_device, numberOfEntities, numberOfNeighbors,
                                                                                 trees_device, treesLeafs_device, trees_size_device, treesLeafs_size_device, numberOfTrees,
                                                                                 dataTable_device, dataTable_Pitch, neighboursDistance_device, neighboursId_device,
                                                                                 idxChecking_device, dimensionOfEntity, partition.start, partition.end,
                                                                                 indexes_device, distances_device, distances2_device, distances3_device,
                                                                                 dimensionOfIndexesAndDistances, marker_device, minSize);
    }else{
        std::cout<<"We do not have such type of distance\n";
    }
    cuCall(cudaStreamSynchronize(executionStreams));
    cudaError_t err2 = cudaGetLastError();
    if (cudaSuccess != err2){
        std::cout<<"findInitialStateOfAproximatedKNN: "<<cudaGetErrorString(err2)<<"\n";
    }
}

__global__
void stubInitializationKernel(int numberOfEntities, int numberOfNeighbors, float* neighboursDistance_device, int* neighboursId_device){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < (numberOfEntities*numberOfNeighbors)){
        neighboursDistance_device[tid] = STUB_INIT_DIST;
        neighboursId_device[tid] = STUB_INIT_ID;
    }
}

void EuclideanDistanceMatrixGPU::stubInitialization(){
    dim3 grid1(ceil(float(numberOfEntities*numberOfNeighbors)/256.0), 1);
    dim3 block1(256, 1);
    stubInitializationKernel<<<grid1, block1, 0, executionStreams>>>(numberOfEntities, numberOfNeighbors, neighboursDistance_device, neighboursId_device);
    cuCall(cudaStreamSynchronize(executionStreams));
    cudaError_t err2 = cudaGetLastError();
    if (cudaSuccess != err2){
        std::cout<<"stubInitializationKernel: "<<cudaGetErrorString(err2)<<"\n";
    }
}

__global__
void makePartitionOfLeaf0(int* treeNodeSizeDevice){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid == 0){
        *treeNodeSizeDevice = 1;
    }
}

__global__
void makePartitionOfLeaf1(float* dataTable_device, size_t dataTable_pitch, int numberOfDimension, int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel,
                          int* points1, int* points2, char* side_device, int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int checkedPoint = elemsPerLeafInCurrentLevel[tid];
        int point1 = elemsPerLeafInCurrentLevel[biasOfElemsPerLeafInCurrentLevel[tid] + points1[tid]];
        int point2 = elemsPerLeafInCurrentLevel[biasOfElemsPerLeafInCurrentLevel[tid] + points2[tid]];
        int size = numberOfElemsPerLeafInCurrentLevel[tid];
        if(size <= minSize){
            return;
        }
        float sideSign = 0.0f;
        for(int dim = 0 ; dim < numberOfDimension ; ++dim){
            float* pElementCheckedPoint = (float*)((char*)dataTable_device + dim * dataTable_pitch) + checkedPoint;
            float* pElementPoint1 = (float*)((char*)dataTable_device + dim * dataTable_pitch) + point1;
            float* pElementPoint2 = (float*)((char*)dataTable_device + dim * dataTable_pitch) + point2;
            sideSign += (*pElementCheckedPoint)*((*pElementPoint2)-(*pElementPoint1));
        }
        if(sideSign < 0){
            side_device[tid] = SIDE_LEFT;
        }else{
            side_device[tid] = SIDE_RIGHT;
        }
    }
}

__global__
void makePartitionOfLeaf2(float* dataTable_device, size_t dataTable_pitch, int numberOfDimension, int numberOfEntities,
                         int* elemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel,
                         int* points1, int* points2, char* side_device, int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if((tid < numberOfEntities) && (tid == biasOfElemsPerLeafInCurrentLevel[tid])){
        int point1 = biasOfElemsPerLeafInCurrentLevel[tid] + points1[tid];
        int point2 = biasOfElemsPerLeafInCurrentLevel[tid] + points2[tid];
        int size = numberOfElemsPerLeafInCurrentLevel[tid];
        if(size <= minSize){
            return;
        }
        side_device[point1] = SIDE_LEFT;
        side_device[point2] = SIDE_RIGHT;
    }
}

__global__
void makePartitionOfLeaf3(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int biasOfElement = biasOfElemsPerLeafInCurrentLevel[tid];
        if(tid == biasOfElement){
            int parent = idOfLeafParent[tid];
            treeNodeDevice[parent].leftChild = 0;
            treeNodeDevice[parent].rightChild = 0;
        }
    }
}

__global__
void makePartitionOfLeaf4(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int element = elemsPerLeafInCurrentLevel[tid];
        int biasOfElement = biasOfElemsPerLeafInCurrentLevel[tid];
        int parent = idOfLeafParent[tid];
        char side = side_device[tid];
        unsigned int* numberOfElemsLeft = (unsigned int*)&treeNodeDevice[parent].leftChild;
        if(side == SIDE_LEFT){
            int newPos = (int)atomicInc(numberOfElemsLeft, INT_MAX);
            elemsPerLeafInCurrentLevel2[biasOfElement + newPos] = element;
            side2_device[biasOfElement + newPos] = SIDE_LEFT;
            biasOfElemsPerLeafInCurrentLevel2[biasOfElement + newPos] = biasOfElement;
            idOfLeafParent2[biasOfElement + newPos] = parent;
        }
    }
}

__global__
void makePartitionOfLeaf5(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int element = elemsPerLeafInCurrentLevel[tid];
        int biasOfElement = biasOfElemsPerLeafInCurrentLevel[tid];
        int parent = idOfLeafParent[tid];
        char side = side_device[tid];
        int numberElemsLeft = treeNodeDevice[parent].leftChild;
        unsigned int* numberOfElemsRight = (unsigned int*)&treeNodeDevice[parent].rightChild;
        if(side == SIDE_RIGHT){
            int newPos = (int)atomicInc(numberOfElemsRight, INT_MAX);
            elemsPerLeafInCurrentLevel2[biasOfElement + numberElemsLeft + newPos] = element;
            side2_device[biasOfElement + numberElemsLeft + newPos] = SIDE_RIGHT;
            biasOfElemsPerLeafInCurrentLevel2[biasOfElement + numberElemsLeft + newPos] = biasOfElement;
            idOfLeafParent2[biasOfElement + numberElemsLeft + newPos] = parent;
        }
    }
}

__global__
void makePartitionOfLeaf6(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int biasOfElement = biasOfElemsPerLeafInCurrentLevel2[tid];
        int parent = idOfLeafParent2[tid];
        char side = side2_device[tid];
        int numberElemsLeft = treeNodeDevice[parent].leftChild;
        int numberElemsRight = treeNodeDevice[parent].rightChild;
        if(side == SIDE_LEFT){
            numberOfElemsPerLeafInCurrentLevel2[tid] = numberElemsLeft;
            biasOfElemsPerLeafInCurrentLevel2[tid] = biasOfElement;
        }
        if(side == SIDE_RIGHT){
            numberOfElemsPerLeafInCurrentLevel2[tid] = numberElemsRight;
            biasOfElemsPerLeafInCurrentLevel2[tid] = biasOfElement + numberElemsLeft;
        }
    }
}

__global__
void makePartitionOfLeaf7(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize, int rand1, int rand2, int* thereWasDividing){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int biasOfElementLeft = biasOfElemsPerLeafInCurrentLevel[tid];
        if(tid == biasOfElementLeft){
            int parent = idOfLeafParent2[tid];
            int numberElemsLeft = treeNodeDevice[parent].leftChild;
            int numberElemsRight = treeNodeDevice[parent].rightChild;
            if((numberElemsLeft > 0 ) && (numberElemsRight > 0)){
                //bylo dzielenie
                atomicCAS(thereWasDividing, 0, 1);

                TreeNode treeNodeLeft = {parent, EMPTY_DIRECTION, EMPTY_DIRECTION, numberElemsLeft};
                TreeNode treeNodeRight = {parent, EMPTY_DIRECTION, EMPTY_DIRECTION, numberElemsRight};

                int idLeft = (int)atomicInc((unsigned int*)treeNodeSizeDevice, INT_MAX);
                int idRight = (int)atomicInc((unsigned int*)treeNodeSizeDevice, INT_MAX);
                treeNodeDevice[idLeft] = treeNodeLeft;
                treeNodeDevice[idRight] = treeNodeRight;

                treeNodeDevice[parent].leftChild = idLeft;
                treeNodeDevice[parent].rightChild = idRight;

                idOfLeafParent2[biasOfElementLeft] = idLeft;
                if(numberElemsLeft > 1){
                    int pointIdx1 = 0;
                    int pointIdx2 = 0;
                    int count = numberElemsLeft;
                    pointIdx1 = int(((double)rand1/(RAND_MAX))*INT_MAX) % count;
                    pointIdx2 = int(((double)rand2/(RAND_MAX))*INT_MAX) % count;
                    if(pointIdx1 == pointIdx2){
                        pointIdx2 = (pointIdx1+1)%count;
                    }
                    points12[biasOfElementLeft] = pointIdx1;
                    points22[biasOfElementLeft] = pointIdx2;
                }else{
                    points12[biasOfElementLeft] = 0;
                    points22[biasOfElementLeft] = 0;
                }

                int biasOfElementRight = biasOfElementLeft + numberElemsLeft;
                idOfLeafParent2[biasOfElementRight] = idRight;
                if(numberElemsRight > 1){
                    int pointIdx1 = 0;
                    int pointIdx2 = 0;
                    int count = numberElemsRight;
                    pointIdx1 = int(((double)rand1/(RAND_MAX))*INT_MAX) % count;
                    pointIdx2 = int(((double)rand2/(RAND_MAX))*INT_MAX) % count;
                    if(pointIdx1 == pointIdx2){
                        pointIdx2 = (pointIdx1+1)%count;
                    }
                    points12[biasOfElementRight] = pointIdx1;
                    points22[biasOfElementRight] = pointIdx2;
                }else{
                    points12[biasOfElementRight] = 0;
                    points22[biasOfElementRight] = 0;
                }
            }else{
                //nie bylo dzielenia
                points12[tid] = 0;
                points22[tid] = 0;
            }
        }
    }
}

__global__
void makePartitionOfLeaf8(int numberOfEntities,
                          int* elemsPerLeafInCurrentLevel, int* elemsPerLeafInCurrentLevel2,
                          int* numberOfElemsPerLeafInCurrentLevel, int* numberOfElemsPerLeafInCurrentLevel2,
                          int* biasOfElemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel2,
                          int* points1, int* points12,
                          int* points2, int* points22,
                          int* idOfLeafParent, int* idOfLeafParent2,
                          char* side_device, char* side2_device,
                          TreeNode* treeNodeDevice, int* treeNodeSizeDevice,
                          int minSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        int biasOfElement = biasOfElemsPerLeafInCurrentLevel2[tid];
        int parent = idOfLeafParent2[biasOfElement];
        int p1 = points12[biasOfElement];
        int p2 = points22[biasOfElement];

        idOfLeafParent2[tid] = parent;
        points12[tid] = p1;
        points22[tid] = p2;
    }
}

__global__
void makePartitionOfLeaf9(int numberOfEntities, int* elemsPerLeafInCurrentLevel, int* biasOfElemsPerLeafInCurrentLevel, int* idOfLeafParent_device,
                          TreeNodeLeaf* treeNodesLeafs, TreeNode* treeNodeDevice){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numberOfEntities){
        TreeNodeLeaf treeNodeLeaf;
        treeNodeLeaf.parent = idOfLeafParent_device[tid];
        treeNodeLeaf.entityNumber = elemsPerLeafInCurrentLevel[tid];
        treeNodesLeafs[tid] = treeNodeLeaf;
        if(tid == biasOfElemsPerLeafInCurrentLevel[tid]){
            treeNodeDevice[treeNodeLeaf.parent].leftChild = tid;
			treeNodeDevice[treeNodeLeaf.parent].rightChild = EMPTY_DIRECTION;
        }
    }
}

void EuclideanDistanceMatrixGPU::buildUpTheTrees(){
    trees_host.clear();
    treesLeafs_host.clear();

    TreeNode* treeNodeDevice;
    TreeNodeLeaf* treeNodesLeafsDevice;
    int* treeNodeSizeDevice;
    int* thereWasDividing;

    cuCall(cudaMalloc((void**)&treeNodeDevice, 2*numberOfEntities*sizeof(TreeNode)));
    cuCall(cudaMalloc((void**)&treeNodesLeafsDevice, numberOfEntities*sizeof(TreeNodeLeaf)));
    cuCall(cudaMalloc((void**)&treeNodeSizeDevice, sizeof(int)));
    cuCall(cudaMalloc((void**)&thereWasDividing, sizeof(int)));

    int* elemsPerLeafInCurrentLevel_host;
    int* elemsPerLeafInCurrentLevel_device;
    int* elemsPerLeafInCurrentLevel2_device;
    int* numberOfElemsPerLeafInCurrentLevel_host;
    int* numberOfElemsPerLeafInCurrentLevel_device;
    int* numberOfElemsPerLeafInCurrentLevel2_device;
    int* biasOfElemsPerLeafInCurrentLevel_host;
    int* biasOfElemsPerLeafInCurrentLevel_device;
    int* biasOfElemsPerLeafInCurrentLevel2_device;
    int* points1_host;
    int* points1_device;
    int* points12_device;
    int* points2_host;
    int* points2_device;
    int* points22_device;
    int* idOfLeafParent_host;
    int* idOfLeafParent_device;
    int* idOfLeafParent2_device;
    char* side_host;
    char* side_device;
    char* side2_device;

    cuCall(cudaHostAlloc((void**)&elemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&elemsPerLeafInCurrentLevel_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&elemsPerLeafInCurrentLevel2_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&numberOfElemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&numberOfElemsPerLeafInCurrentLevel_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&numberOfElemsPerLeafInCurrentLevel2_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&biasOfElemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&biasOfElemsPerLeafInCurrentLevel_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&biasOfElemsPerLeafInCurrentLevel2_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&points1_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&points1_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&points12_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&points2_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&points2_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&points22_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&idOfLeafParent_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&idOfLeafParent_device, numberOfEntities*sizeof(int)));
    cuCall(cudaMalloc((void**)&idOfLeafParent2_device, numberOfEntities*sizeof(int)));
    cuCall(cudaHostAlloc((void**)&side_host, numberOfEntities*sizeof(char), cudaHostAllocPortable));
    cuCall(cudaMalloc((void**)&side_device, numberOfEntities*sizeof(char)));
    cuCall(cudaMalloc((void**)&side2_device, numberOfEntities*sizeof(char)));

    for(int i=0 ; i<numberOfTrees ; ++i){
        std::cout<<"The tree with number: "<<i+1<<" is building\n";
        //Inicjalizacja
        std::vector<TreeNode> treeNodes;
        trees_host[i] = treeNodes;
        std::vector<TreeNodeLeaf> treeNodesLeafs;
        treesLeafs_host[i] = treeNodesLeafs;

        TreeNode treeNode = {EMPTY_DIRECTION, EMPTY_DIRECTION, EMPTY_DIRECTION, numberOfEntities};
        trees_host[i].push_back(treeNode);

        cuCall(cudaMemcpyAsync((void*)treeNodeDevice, (void*)trees_host[i].data(), sizeof(TreeNode), cudaMemcpyHostToDevice, executionStreams));
        makePartitionOfLeaf0<<<1, 1, 0, executionStreams>>>(treeNodeSizeDevice);

        //Inicjalizacja tablic
        int pointIdx1 = 0;
        int pointIdx2 = 0;
        int count = numberOfEntities;
        pointIdx1 = int(((double)rand()/(RAND_MAX))*INT_MAX) % count;
        pointIdx2 = int(((double)rand()/(RAND_MAX))*INT_MAX) % count;
        if(pointIdx1 == pointIdx2){
            pointIdx2 = (pointIdx1+1)%count;
        }

        for(int k=0 ; k<numberOfEntities ; ++k){
            elemsPerLeafInCurrentLevel_host[k] = k;
            numberOfElemsPerLeafInCurrentLevel_host[k] = numberOfEntities;
            biasOfElemsPerLeafInCurrentLevel_host[k] = 0;
            points1_host[k] = pointIdx1;
            points2_host[k] = pointIdx2;
            idOfLeafParent_host[k] = 0;
            side_host[k] = SIDE_LEFT;
        }

        //Przeslanie na GPU odpowiednich tablic
        cuCall(cudaMemcpyAsync((void*)elemsPerLeafInCurrentLevel_device, (void*)elemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)numberOfElemsPerLeafInCurrentLevel_device, (void*)numberOfElemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)biasOfElemsPerLeafInCurrentLevel_device, (void*)biasOfElemsPerLeafInCurrentLevel_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)points1_device, (void*)points1_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)points2_device, (void*)points2_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)idOfLeafParent_device, (void*)idOfLeafParent_host, numberOfEntities*sizeof(int), cudaMemcpyHostToDevice, executionStreams));
        cuCall(cudaMemcpyAsync((void*)side_device, (void*)side_host, numberOfEntities*sizeof(char), cudaMemcpyHostToDevice, executionStreams));

        //Dzielenie galezi
        bool treeeIsGoingToBeEdit = true;
        while(treeeIsGoingToBeEdit == true){
            treeeIsGoingToBeEdit = false;

            //Przeliczenie
            dim3 grid(ceil(float(numberOfEntities)/256.0), 1);
            dim3 block(256, 1);
            makePartitionOfLeaf1<<<grid, block, 0, executionStreams>>>(dataTable_device, dataTable_Pitch, dimensionOfEntity, numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device,
                                                                       points1_device, points2_device, side_device, minSize);

            makePartitionOfLeaf2<<<grid, block, 0, executionStreams>>>(dataTable_device, dataTable_Pitch, dimensionOfEntity, numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device,
                                                                       points1_device, points2_device, side_device, minSize);

            //czasowo wszystko czyscimy
            cuCall(cudaMemsetAsync((void*)elemsPerLeafInCurrentLevel2_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)numberOfElemsPerLeafInCurrentLevel2_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)biasOfElemsPerLeafInCurrentLevel2_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)points12_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)points22_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)idOfLeafParent2_device, 0, numberOfEntities*sizeof(int), executionStreams));
            cuCall(cudaMemsetAsync((void*)side2_device, 0, numberOfEntities*sizeof(char), executionStreams));
            cuCall(cudaMemsetAsync((void*)thereWasDividing, 0, sizeof(int), executionStreams));

            makePartitionOfLeaf3<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize);

            makePartitionOfLeaf4<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize);

            makePartitionOfLeaf5<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize);

            makePartitionOfLeaf6<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize);

            makePartitionOfLeaf7<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize, rand(), rand(), thereWasDividing);

            makePartitionOfLeaf8<<<grid, block, 0, executionStreams>>>(numberOfEntities,
                                                                       elemsPerLeafInCurrentLevel_device, elemsPerLeafInCurrentLevel2_device,
                                                                       numberOfElemsPerLeafInCurrentLevel_device, numberOfElemsPerLeafInCurrentLevel2_device,
                                                                       biasOfElemsPerLeafInCurrentLevel_device, biasOfElemsPerLeafInCurrentLevel2_device,
                                                                       points1_device, points12_device,
                                                                       points2_device, points22_device,
                                                                       idOfLeafParent_device, idOfLeafParent2_device,
                                                                       side_device, side2_device,
                                                                       treeNodeDevice, treeNodeSizeDevice,
                                                                       minSize);

            int thereWasDividingHost;
            cuCall(cudaMemcpyAsync((void*)&thereWasDividingHost, (void*)thereWasDividing, sizeof(int), cudaMemcpyDeviceToHost, executionStreams));
            cuCall(cudaStreamSynchronize(executionStreams));

            int* tmp1 = elemsPerLeafInCurrentLevel_device;
            elemsPerLeafInCurrentLevel_device = elemsPerLeafInCurrentLevel2_device;
            elemsPerLeafInCurrentLevel2_device = tmp1;

            int* tmp2 = numberOfElemsPerLeafInCurrentLevel_device;
            numberOfElemsPerLeafInCurrentLevel_device = numberOfElemsPerLeafInCurrentLevel2_device;
            numberOfElemsPerLeafInCurrentLevel2_device = tmp2;

            int* tmp3 = biasOfElemsPerLeafInCurrentLevel_device;
            biasOfElemsPerLeafInCurrentLevel_device = biasOfElemsPerLeafInCurrentLevel2_device;
            biasOfElemsPerLeafInCurrentLevel2_device = tmp3;

            int* tmp4 = points1_device;
            points1_device = points12_device;
            points12_device = tmp4;

            int* tmp5 = points2_device;
            points2_device = points22_device;
            points22_device = tmp5;

            int* tmp6 = idOfLeafParent_device;
            idOfLeafParent_device = idOfLeafParent2_device;
            idOfLeafParent2_device = tmp6;

            char* tmp7 = side_device;
            side_device = side2_device;
            side2_device = tmp7;

            if(thereWasDividingHost != 0){
                treeeIsGoingToBeEdit = true;
            }
        }

        //Utworzenie koncowych lisci z wlasciwymi elementami
        dim3 grid(ceil(float(numberOfEntities)/256.0), 1);
        dim3 block(256, 1);
        makePartitionOfLeaf9<<<grid, block, 0, executionStreams>>>(numberOfEntities, elemsPerLeafInCurrentLevel_device,
                                                                   biasOfElemsPerLeafInCurrentLevel_device, idOfLeafParent_device,
                                                                   treeNodesLeafsDevice, treeNodeDevice);

        int treeNodeSizeHost;
        cuCall(cudaMemcpyAsync((void*)&treeNodeSizeHost, (void*)treeNodeSizeDevice, sizeof(int), cudaMemcpyDeviceToHost, executionStreams));
        cuCall(cudaStreamSynchronize(executionStreams));

        trees_host[i].resize(treeNodeSizeHost);
        treesLeafs_host[i].resize(numberOfEntities);

        cuCall(cudaMemcpyAsync((void*)trees_host[i].data(), (void*)treeNodeDevice, treeNodeSizeHost*sizeof(TreeNode), cudaMemcpyDeviceToHost, executionStreams));
        cuCall(cudaMemcpyAsync((void*)treesLeafs_host[i].data(), (void*)treeNodesLeafsDevice, numberOfEntities*sizeof(TreeNodeLeaf), cudaMemcpyDeviceToHost, executionStreams));
        cuCall(cudaStreamSynchronize(executionStreams));

        std::cout<<"The tree with number: "<<i+1<<" has been built\n";
    }

    cuCall(cudaFree((void*)treeNodeDevice));
    cuCall(cudaFree((void*)treeNodesLeafsDevice));
    cuCall(cudaFree((void*)treeNodeSizeDevice));
    cuCall(cudaFree((void*)thereWasDividing));

    cuCall(cudaFreeHost((void*)elemsPerLeafInCurrentLevel_host));
    cuCall(cudaFree((void*)elemsPerLeafInCurrentLevel_device));
    cuCall(cudaFree((void*)elemsPerLeafInCurrentLevel2_device));
    cuCall(cudaFreeHost((void*)numberOfElemsPerLeafInCurrentLevel_host));
    cuCall(cudaFree((void*)numberOfElemsPerLeafInCurrentLevel_device));
    cuCall(cudaFree((void*)numberOfElemsPerLeafInCurrentLevel2_device));
    cuCall(cudaFreeHost((void*)biasOfElemsPerLeafInCurrentLevel_host));
    cuCall(cudaFree((void*)biasOfElemsPerLeafInCurrentLevel_device));
    cuCall(cudaFree((void*)biasOfElemsPerLeafInCurrentLevel2_device));
    cuCall(cudaFreeHost((void*)points1_host));
    cuCall(cudaFree((void*)points1_device));
    cuCall(cudaFree((void*)points12_device));
    cuCall(cudaFreeHost((void*)points2_host));
    cuCall(cudaFree((void*)points2_device));
    cuCall(cudaFree((void*)points22_device));
    cuCall(cudaFreeHost((void*)idOfLeafParent_host));
    cuCall(cudaFree((void*)idOfLeafParent_device));
    cuCall(cudaFree((void*)idOfLeafParent2_device));
    cuCall(cudaFreeHost((void*)side_host));
    cuCall(cudaFree((void*)side_device));
    cuCall(cudaFree((void*)side2_device));
}

EuclideanDistanceMatrixGPU::EuclideanDistanceMatrixGPU(){
    typeOfDistance = DISTANCE_EUCLIDEAN;
    this->numberOfBlocksPerMultiprocessors = 10;
    this->numberOfMultiprocessors = 1;
    this->debugMode = false;
    this->minSize = 1;
}

EuclideanDistanceMatrixGPU::EuclideanDistanceMatrixGPU(bool debugMode){
    typeOfDistance = DISTANCE_EUCLIDEAN;
    this->numberOfBlocksPerMultiprocessors = 10;
    this->numberOfMultiprocessors = 1;
    this->debugMode = debugMode;
    this->minSize = 1;
}

EuclideanDistanceMatrixGPU::~EuclideanDistanceMatrixGPU(){

}

void EuclideanDistanceMatrixGPU::setDataFile(std::string nameOfFile){
    this->inputFile = nameOfFile;
}

bool EuclideanDistanceMatrixGPU::loadData(){
    std::ifstream myfile;
    myfile.open(this->inputFile.c_str());
    if (myfile.is_open()){
        std::cout<<"The datafile has been opened\n";
    }else{
        std::cout<<"Error opening the file\n";
        return true;
    }
    std::string line;

    std::getline(myfile, line);
    std::getline(myfile, line);

    int idOfEntity = 0;
    char* lineChar;
    while ((std::getline(myfile, line))&&(idOfEntity<numberOfEntities)){
        std::vector<std::string> cuttedString;
        lineChar = new char[line.length() + 1];
        std::strcpy(lineChar, line.c_str());

        std::string str;
        char* pch = strtok(lineChar,",");
        while (pch != NULL){
            str = std::string(pch);
            str = trim(str);
            cuttedString.push_back(str);
            pch = strtok (NULL, ",");
        }

        delete [] lineChar;

        if(klaster){
            for(int i=0 ; i<cuttedString.size()-1 ; ++i){
                this->dataTable_host[idOfEntity+numberOfEntities*i] = atof(cuttedString[i].c_str());
            }
            this->dataTableId_host[idOfEntity] = atoi(cuttedString[cuttedString.size()-1].c_str());
        }else{
            for(int i=0 ; i<cuttedString.size() ; ++i){
                this->dataTable_host[idOfEntity+numberOfEntities*i] = atof(cuttedString[i].c_str());
            }
        }
        idOfEntity++;
    }
    return false;
}

bool EuclideanDistanceMatrixGPU::initialize(int numberOfEntities, int dimensionOfEntity, int numberOfNeighbors,
                                            int device, int typeOfDistance, bool klaster, int numberOfTrees, int numberOfPropagations, int minSize){
    this->typeOfDistance = typeOfDistance;
    this->klaster = klaster;
    this->numberOfEntities = numberOfEntities;
    this->numberOfNeighbors = numberOfNeighbors;
    this->dimensionOfEntity = dimensionOfEntity;
    this->numberOfTrees = numberOfTrees;
    this->numberOfPropagations = numberOfPropagations;
    this->dimensionOfIndexesAndDistances = min(numberOfNeighbors*numberOfNeighbors+numberOfNeighbors, numberOfEntities);
    this->minSize = minSize;
    this->device = device;

    bool error = false;

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaDeviceReset());
    cudaDeviceProp devProp;
    error |= cuCall(cudaGetDeviceProperties(&devProp, device));
    this->numberOfMultiprocessors = devProp.multiProcessorCount;

    error |= cuCall(cudaHostAlloc((void**)&dataTable_host, numberOfEntities*dimensionOfEntity*sizeof(float), cudaHostAllocPortable));
    error |= cuCall(cudaHostAlloc((void**)&dataTableId_host, numberOfEntities*sizeof(int), cudaHostAllocPortable));

    error |= cuCall(cudaMallocPitch((void**)&dataTable_device, &dataTable_Pitch, numberOfEntities*sizeof(float), dimensionOfEntity));

    error |= cuCall(cudaMallocHost((void**)&neighboursDistance_host, numberOfNeighbors*numberOfEntities*sizeof(float)));
    error |= cuCall(cudaMalloc((void**)&neighboursDistance_device, numberOfNeighbors*numberOfEntities*sizeof(float)));
    error |= cuCall(cudaMalloc((void**)&neighboursDistance2_device, numberOfNeighbors*numberOfEntities*sizeof(float)));

    error |= cuCall(cudaMallocHost((void**)&neighboursId_host, numberOfNeighbors*numberOfEntities*sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&neighboursId_device, numberOfNeighbors*numberOfEntities*sizeof(int)));
    error |= cuCall(cudaMalloc((void**)&neighboursId2_device, numberOfNeighbors*numberOfEntities*sizeof(int)));

    error |= cuCall(cudaStreamCreate(&executionStreams));
    error |= cuCall(cudaEventCreate(&startEvents));
    error |= cuCall(cudaEventCreate(&stopEvents));

    error |= cuCall(cudaMalloc((void**)&idxChecking_device, numberOfMultiprocessors*numberOfBlocksPerMultiprocessors*numberOfEntities*sizeof(char)));

    error |= cuCall(cudaMalloc((void**)&indexes_device, numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*dimensionOfIndexesAndDistances*sizeof(int)));

    error |= cuCall(cudaMalloc((void**)&distances_device, numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*dimensionOfIndexesAndDistances*sizeof(float)));
    error |= cuCall(cudaMalloc((void**)&distances2_device, numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*dimensionOfIndexesAndDistances*sizeof(float)));
    error |= cuCall(cudaMalloc((void**)&distances3_device, numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*dimensionOfIndexesAndDistances*sizeof(float)));

    error |= cuCall(cudaMalloc((void**)&marker_device, numberOfBlocksPerMultiprocessors*numberOfMultiprocessors*dimensionOfIndexesAndDistances*sizeof(short)));

    error |= loadData();

    //send data to GPU
    error |= cuCall(cudaMemcpy2D((void*)dataTable_device, dataTable_Pitch, (void*)dataTable_host, numberOfEntities*sizeof(float),
                                 numberOfEntities*sizeof(float), dimensionOfEntity, cudaMemcpyHostToDevice));

    error |= cuCall(cudaMalloc((void**)&graphTraversalStartPoint_device, numberOfTrees*numberOfEntities*sizeof(int)));

    Partition p = {0, numberOfEntities};
    partition = p;

    return error;
}

bool EuclideanDistanceMatrixGPU::deinitialize(){
    bool error = false;

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaDeviceSynchronize());

    error |= cuCall(cudaFreeHost((void*)dataTable_host));
    error |= cuCall(cudaFreeHost((void*)dataTableId_host));

    error |= cuCall(cudaFree((void*)dataTable_device));

    error |= cuCall(cudaFreeHost((void*)neighboursDistance_host));
    error |= cuCall(cudaFree((void*)neighboursDistance_device));
    error |= cuCall(cudaFree((void*)neighboursDistance2_device));

    error |= cuCall(cudaFreeHost((void*)neighboursId_host));
    error |= cuCall(cudaFree((void*)neighboursId_device));
    error |= cuCall(cudaFree((void*)neighboursId2_device));

    error |= cuCall(cudaStreamDestroy(executionStreams));
    error |= cuCall(cudaEventDestroy(startEvents));
    error |= cuCall(cudaEventDestroy(stopEvents));

    error |= cuCall(cudaFree((void*)idxChecking_device));

    error |= cuCall(cudaFree((void*)indexes_device));

    error |= cuCall(cudaFree((void*)distances_device));
    error |= cuCall(cudaFree((void*)distances2_device));
    error |= cuCall(cudaFree((void*)distances3_device));

    error |= cuCall(cudaFree((void*)marker_device));

    error |= cuCall(cudaFree((void*)graphTraversalStartPoint_device));

    error |= cuCall(cudaDeviceReset());

    return error;
}

bool EuclideanDistanceMatrixGPU::calculate(){
    bool error = false;

    std::cout<<"The device "<<device<<" is calculating the neighbours for: "<<partition.start<<" - "<<partition.end-1<<"\n";

    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaEventRecord(startEvents, executionStreams));

    dim3 grid1(ceil(float(dimensionOfEntity)/256.0), 1);
    dim3 block1(256, 1);
    normalizeDataStep1<<<grid1, block1, 0, executionStreams>>>(dataTable_device, dataTable_Pitch, numberOfEntities, dimensionOfEntity);

    dim3 grid2(1, 1);
    dim3 block2(256, 1);
    normalizeDataStep2<<<grid2, block2, 0, executionStreams>>>(dataTable_device, dataTable_Pitch, numberOfEntities, dimensionOfEntity);

    buildUpTheTrees();

    initilizeGPUStructuresForTrees();

    stubInitialization();

    findInitialKNN();

    propagate();

    deinitializeGPUStructuresForTrees();

    error |= cuCall(cudaEventRecord(stopEvents, executionStreams));
    error |= cuCall(cudaEventSynchronize(stopEvents));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvents, stopEvents);
    std::cout<<"The device "<<device<<": has done task in: "<<milliseconds<<" ms\n";

    return error;
}

void EuclideanDistanceMatrixGPU::setResultsFile(std::string nameOfFile){
    this->outputFile = nameOfFile;
}

template <typename T> std::string tostr(const T& t) {
   std::ostringstream os;
   os<<t;
   return os.str();
}

bool EuclideanDistanceMatrixGPU::saveResultToResultFile(){
    bool error = false;
    error |= cuCall(cudaSetDevice(device));
    error |= cuCall(cudaMemcpyAsync((void*)neighboursDistance_host, (void*)neighboursDistance_device,
                                    numberOfNeighbors*numberOfEntities*sizeof(float), cudaMemcpyDeviceToHost, executionStreams));
    error |= cuCall(cudaMemcpyAsync((void*)neighboursId_host, (void*)neighboursId_device,
                                    numberOfNeighbors*numberOfEntities*sizeof(int), cudaMemcpyDeviceToHost, executionStreams));
    error |= cuCall(cudaStreamSynchronize(executionStreams));

    //Zapisanie rezultatu do pliku
    std::ofstream ofs;
    ofs.open(outputFile.c_str(), std::ofstream::trunc | std::ofstream::binary);
    std::ofstream ofsDebug;
    if(debugMode){
        ofsDebug.open((outputFile+"DEBUG").c_str(), std::ofstream::trunc | std::ofstream::binary);
    }
    bool validationSuccess = true;
    std::ofstream ofsValidation;
    ofsValidation.open((outputFile+"VALIDATION").c_str(), std::ofstream::trunc | std::ofstream::binary);
    if(ofs.is_open()){
        ofs<<numberOfEntities<<";"<< numberOfNeighbors<<";"<<sizeof(long)<<"\n";
        long l = 0x01020304;
        ofs.write((char*)&l, sizeof(long));
        //zapisywanie punktow
        for(int lp=partition.start ; lp<partition.end ; ++lp){
            std::vector<DataPoint> liczbyNear;
            for(int c=0 ; c<numberOfNeighbors ; ++c){
                DataPoint dp = {neighboursId_host[lp*numberOfNeighbors+c], neighboursDistance_host[lp*numberOfNeighbors+c]};
                liczbyNear.push_back(dp);
            }
            if(typeOfDistance == DISTANCE_COSINE){
                std::sort(liczbyNear.begin(), liczbyNear.end(), compareByLengthMinCosine);
            }else{
                std::sort(liczbyNear.begin(), liczbyNear.end(), compareByLengthMin);
            }
            for(std::vector<DataPoint>::iterator it = liczbyNear.begin() ; it != liczbyNear.end() ; ++it){
                DataPoint f = *it;
                ofs.write((char*)&f.id, sizeof(long));
                if((debugMode)&&(ofsDebug.is_open())){
                    ofsDebug<<"NEAR: <"<<lp<<", "<<f.id<<">("<<f.distance<<") ";
                }
            }
            /*
            for(std::vector<DataPoint>::iterator it = liczbyNear.begin() ; it != liczbyNear.end() ; ++it){
                long tmp = -1;
                ofs.write((char*)&tmp, sizeof(long));
                if((debugMode)&&(ofsDebug.is_open())){
                    ofsDebug<<"FAR: <"<<lp<<", "<<tmp<<">("<<FLT_MAX<<") ";
                }
            }
            */
            if((debugMode)&&(ofsDebug.is_open())){
                ofsDebug<<";\n";
            }
        }
        ofs.close();
        if((debugMode)&&(ofsDebug.is_open())){
            ofsDebug.close();
        }
        if(ofsValidation.is_open()){
            if(validationSuccess){
                ofsValidation<<"Everything is OK.";
            }
            ofsValidation.close();
        }
    }else{
        std::cout <<"Can not open the file for saving result.\n";
        error |= true;
    }
    return error;
}
