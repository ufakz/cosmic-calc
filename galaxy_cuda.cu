#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>


int NO_OF_REAL;
int NO_OF_RAND;

float *real_ra, *real_dl;
float *rand_ra, *rand_dl;

unsigned long long int *DR, *DD, *RR;

int TOTAL_DEGREES = 360;
int BINS_PER_DEGREE = 4;

__global__ void fillHistogram(unsigned long long int *histogram, float* rasc1, float* decl1, float* rasc2, float* decl2, int N, bool skip){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;


    /* We could compute the value of theta once and assign based on condition.
    However, to avoid computing unused values, we only compute when the condition is met. */

    if(i < N && j < N){
        if (skip && j > i) {
            float pif = acosf(-1.0f);
            float theta = sin(decl1[i]) * sin(decl2[j]) + cos(decl1[i]) * cos(decl2[j]) * cos((rasc1[i] - rasc2[j]));

            if (theta > 1.0) theta = 1.0;
            if (theta < -1.0) theta = -1.0;

            theta = acosf(theta) * 180 / pif;
            int ind = (int)(theta * 4.0);

            atomicAdd(&histogram[ind], 2);
        } else if ((skip && j == i) || !skip) {
            float pif = acosf(-1.0f);
            float theta = sin(decl1[i]) * sin(decl2[j]) + cos(decl1[i]) * cos(decl2[j]) * cos((rasc1[i] - rasc2[j]));

            if (theta > 1.0) theta = 1.0;
            if (theta < -1.0) theta = -1.0;

            theta = acosf(theta) * 180 / pif;
            int ind = (int)(theta * 4.0);

            atomicAdd(&histogram[ind], 1);
        }
    }
}

int main(int argc, char *argv[]){
    int readData(char *argv1, char *argv2);
    int getDevice(int deviceno);

    unsigned long long int sumDR, sumDD, sumRR;

    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;

    FILE *outFile;

    if (argc != 4 ){
        printf("Usage: galaxy.out real_data random_data output_data\n");
        return -1;
    }

    if ( getDevice(0) != 0 ) return(-1);

    if (readData(argv[1], argv[2]) != 0){
        return -1;
    }

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    size_t arraybytes = TOTAL_DEGREES * sizeof(unsigned long long int);
    DR = (unsigned long long int *)malloc(arraybytes);
    DD = (unsigned long long int *)malloc(arraybytes);
    RR = (unsigned long long int *)malloc(arraybytes);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("%v", err);
    }

    for (int i=0; i < TOTAL_DEGREES; i++){
         DR[i] = 0LLU;
         DD[i] = 0LLU;
         RR[i] = 0LLU;
    }

    unsigned long long int *gpu_DR, *gpu_DD, *gpu_RR;
    cudaMalloc(&gpu_DR, arraybytes);
    cudaMalloc(&gpu_DD, arraybytes);
    cudaMalloc(&gpu_RR, arraybytes);

    size_t arraybytes_real = (NO_OF_REAL) * sizeof(float);
    size_t arraybytes_rand = (NO_OF_RAND) * sizeof(float);

    float *gpu_real_ra, *gpu_real_dl, *gpu_rand_ra, *gpu_rand_dl;
    cudaMalloc(&gpu_real_ra, arraybytes_real);
    cudaMalloc(&gpu_real_dl, arraybytes_real);
    cudaMalloc(&gpu_rand_ra, arraybytes_rand);
    cudaMalloc(&gpu_rand_dl, arraybytes_rand);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("%v", err);
    }

    /* Move data to GPU */
    cudaMemcpy(gpu_DR, DR, arraybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_DD, DD, arraybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_RR, RR, arraybytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_real_ra, real_ra, arraybytes_real, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_real_dl, real_dl, arraybytes_real, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rand_ra, rand_ra, arraybytes_rand, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rand_dl, rand_dl, arraybytes_rand, cudaMemcpyHostToDevice);

    dim3 blocksInGrid(6250, 6250);
    dim3 threadsInBlock(16, 16);

    fillHistogram<<<blocksInGrid, threadsInBlock>>>(gpu_DR, gpu_real_ra, gpu_real_dl, gpu_rand_ra, gpu_rand_dl, NO_OF_REAL, 0);
    fillHistogram<<<blocksInGrid, threadsInBlock>>>(gpu_DD, gpu_real_ra, gpu_real_dl, gpu_real_ra, gpu_real_dl, NO_OF_REAL, 1);
    fillHistogram<<<blocksInGrid, threadsInBlock>>>(gpu_RR, gpu_rand_ra, gpu_rand_dl, gpu_rand_ra, gpu_rand_dl, NO_OF_RAND, 1);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("%v", err);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(DR, gpu_DR, arraybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(DD, gpu_DD, arraybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(RR, gpu_RR, arraybytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_DR);
    cudaFree(gpu_DD);
    cudaFree(gpu_RR);

    cudaFree(gpu_real_ra);
    cudaFree(gpu_real_dl);
    cudaFree(gpu_rand_ra);
    cudaFree(gpu_rand_dl);


    sumDR = 0LLU;
    for (int i = 0; i < TOTAL_DEGREES; ++i){
        sumDR += DR[i];
    }
    printf("Histogram DR sum = %lld\n", sumDR);
    if ( sumDR != 10000000000LLU){
        printf("Incorrect histogram sum");
    }

    sumDD = 0LLU;
    for (int i = 0; i < TOTAL_DEGREES; ++i){
        sumDD += DD[i];
    }
    printf("Histogram DD sum = %lld\n", sumDD);
    if ( sumDD != 10000000000LLU){
        printf("Incorrect histogram sum");
    }

    sumRR = 0LLU;
    for (int i = 0; i < TOTAL_DEGREES; ++i){
        sumRR += RR[i];
    }
    printf("Histogram DR sum = %lld\n", sumRR);
    if ( sumRR != 10000000000LLU){
        printf("Incorrect histogram sum");
    }



    outFile = fopen(argv[3],"w");
    if (outFile == NULL){
        printf("Cannot open output file %s\n", argv[3]);
        return -1;
    }

    fprintf(outFile,"bin start\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");

    for (int i = 0; i < TOTAL_DEGREES; ++i){
        if (RR[i] > 0){
            double omega = (DD[i] - 2*DR[i] + RR[i]) / ((double)(RR[i]));

            fprintf(outFile,"%6.4f\t%15lf\t%15lld\t%15lld\t%15lld\n",((float)i)/BINS_PER_DEGREE, omega,
            DD[i], DR[i], RR[i]);

            if (i < 5) printf("  %6.4lf", omega);
        }else{
            if (i < 5) printf("  ");
        }
    }

    printf("\n");

    fclose(outFile);

    printf("Results written to file %s\n", argv[3]);

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;
    printf("Time taken = %.2lf s\n", ((float) kerneltime));

    return 0;
}

int readData(char *argv1, char *argv2){
    int i, lineCount;
    char inbuf[180];
    double ra, dl, dpi;
    FILE *file;

    printf("Data in arc minutes!\n");


    dpi = acos(-1.0);

    /* Read real galaxies file */

    file = fopen(argv1, "r");
    if (file == NULL) {
        printf("Cannot open file %s", argv1);
        return -1;
    }

    lineCount = 0;
    while(fgets(inbuf, 180, file) != NULL) ++lineCount;
    rewind(file);

    NO_OF_REAL = lineCount - 1;

    printf("%d", NO_OF_REAL);

    if (NO_OF_REAL != 100000){
        printf("Incorrect number of galaxies\n");
        return 1;
    }

    real_ra = (float *)calloc(NO_OF_REAL, sizeof(float));
    real_dl = (float *)calloc(NO_OF_REAL, sizeof(float));

    fgets(inbuf, 180, file);
    sscanf(inbuf, "%d", &lineCount);

    if (lineCount != 100000){
        printf("Incorrect number of galaxies specified in file\n");
    }

    i = 0;
    while(fgets(inbuf, 80, file) != NULL){
        if (sscanf(inbuf, "%lf %lf", &ra, &dl) != 2){
            printf("Incorrect values in line %d in file %s \n", i+1, argv1);
            fclose(file);
            return -1;
        }

        real_ra[i] = (float)(ra/60.0 * dpi/180.0);
        real_dl[i] = (float)(dl/60.0 * dpi/180.0);
        ++i;
    }

    fclose(file);

    if (i != NO_OF_REAL){
        printf("Cannot read %s correctly\n", argv1);
        return -1;
    }

    /* Read random galaxies file */

    file = fopen(argv2, "r");
    if (file == NULL) {
        printf("Cannot open file %s", argv1);
        return -1;
    }

    lineCount = 0;
    while(fgets(inbuf, 180, file) != NULL) ++lineCount;
    rewind(file);

    NO_OF_RAND = lineCount - 1;

    if (NO_OF_RAND != 100000){
        printf("Incorrect number of galaxies\n");
        return 1;
    }

    rand_ra = (float *)calloc(NO_OF_RAND, sizeof(float));
    rand_dl = (float *)calloc(NO_OF_RAND, sizeof(float));

    fgets(inbuf, 180, file);
    sscanf(inbuf, "%d", &lineCount);

    if (lineCount != 100000){
        printf("Incorrect number of galaxies specified in file\n");
    }

    i = 0;
    while(fgets(inbuf, 80, file) != NULL){
        if (sscanf(inbuf, "%lf %lf", &ra, &dl) != 2){
            printf("Incorrect values in line %d in file %s \n", i+1, argv2);
            fclose(file);
            return -1;
        }

        rand_ra[i] = (float)(ra/60.0 * dpi/180.0);
        rand_dl[i] = (float)(dl/60.0 * dpi/180.0);
        ++i;
    }

    fclose(file);

    if (i != NO_OF_RAND){
        printf("Cannot read %s correctly\n", argv2);
        return -1;
    }

    return 0;
}

int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}


