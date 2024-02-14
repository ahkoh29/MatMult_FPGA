//extern "C" {
#include "matmult.h"
void matmult(vector<float> &A, vector<float> &B, vector<float> &C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if((row < numCRows) && (col < numCColumns)) {
    float pValue = 0;
    for(int k=0; k<numAColumns; k++){
      pValue += A[row*numAColumns+k] * B[k*numBColumns+col];
    }
    C[row*numCColumns+col] = pValue;
  }

}
//}