#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
  int stride = blockDim.x * gridDim.x ; 
  for(int i = idx; i < N; i+=stride )
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  size_t number_of_threads = 256 ;
  size_t number_of_blocks  = (N + number_of_threads -1 ) / number_of_threads;

  cudaError_t syncErr , asyncErr ;


  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a,size);
  cudaMallocManaged(&b,size);
  cudaMallocManaged(&c,size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto<<<number_of_blocks,number_of_threads>>>(c, a, b, N);
  syncErr = cudaDeviceSynchronize();
  asyncErr = cudaGetLastError();
  checkElementsAre(7, c, N);
  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
