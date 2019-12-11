#ifndef INCLUDE_OPERATOR_WITHCUDA_CUH
#define INCLUDE_OPERATOR_WITHCUDA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define __dual__ __host__ __device__
#define SUCCESS true

using namespace std;

typedef unsigned int uint32_t;
typedef unsigned short int uint16_t;
typedef float float32_t;
typedef double float64_t;

bool status;

// =============================================================================================
// Functions to check CUDA errors and capability
// =============================================================================================

bool cudaCheck(cudaError_t cudaStatus){
    return cudaStatus == cudaSuccess;
    /*if(cudaStatus != cudaSuccess)
        fprintf(stderr, "\t* [ ERROR ]: %s\n\n", cudaGetErrorString(cudaStatus));
    else
        printf("\t* [ OK ]\n");//*/
}

void cudaCheckKernel(){
    cudaError_t cudaStatus;
    
    cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess)
        fprintf(stderr, "\t* kernel launch... [ ERROR ]: %s\n\n", cudaGetErrorString(cudaStatus));
    else
        printf("\t* kernel launch... [ OK ]\n");

    cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess)
        fprintf(stderr, "\t* cudaDeviceSynchronize() after launching kernel... [ ERROR ]: %d\n", cudaStatus);
    else
        printf("\t* cudaDeviceSynchronize() after launching kernel... [ OK ]\n");
}

extern "C" {
    int check_cuda(){
        int num_gpus;
        cudaError_t cudaStatus;
        
        //printf("-> Checking availability of CUDA:\n");
        cudaStatus = cudaGetDeviceCount(&num_gpus);

        if(cudaStatus == cudaSuccess && num_gpus > 0){
            cudaDeviceProp gpu_properties;
            cudaGetDeviceProperties(&gpu_properties, 0);

            printf("\t* checking availability of CUDA ... [ OK ]\n");
            printf("\t* number of CUDA GPUs detected: %d\n", num_gpus);
            printf("\t* using CUDA GPU: %s\n", gpu_properties.name);

            if(gpu_properties.major >= 5){
                printf("\t* compute capability: %d.%d [ OK ]\n", gpu_properties.major, gpu_properties.minor);
            }
            else{
                printf("\t* compute capability: %d.%d [ ERROR ]. GPU compute capability must be at least 5.0\n", gpu_properties.major, gpu_properties.minor);
                return -1;
            }

            return 0;
        }
        else{
            printf("\t* checking availability of CUDA ... [ ERROR ]: CUDA is not available or GPU is not CUDA compatible\n");
            return -1;
        }
    }
}

// =============================================================================================
// @brief:
//      Represent a single fiber segment (a.k.a IC compartment)
//
// @members:
//      voxel : voxel index of the segment
//      fiber : fiber index of the segment
//      orien : orientation index of the segment
//      contr : contribution of the segment
// =============================================================================================

// =============================================================================================
// Global Variables
// =============================================================================================

// globals in CPU
static int NUM_SEGMENTS;
static int NUM_VOXELS;
static int NUM_FIBERS;
static int NUM_EXCOMPS;
static int NUM_RESFUNCISO;
static int NUM_RESFUNCEC;
static int NUM_RESFUNCIC;
static int NUM_SAMPLES;
static int NUM_ORIENTATIONS;
static int NUM_COLS;
static int NUM_ROWS;
static int SIZE_LUTIC;
static int SIZE_LUTEC;
static int SIZE_LUTISO;

// globals in GPU
__constant__ int num_voxels;
__constant__ int num_fibers;
__constant__ int num_excomps;
__constant__ int num_resfunciso;
__constant__ int num_resfuncec;
__constant__ int num_resfuncic;
__constant__ int num_samples;
__constant__ int num_orientations;
__constant__ int num_cols;
__constant__ int num_rows;
__constant__ int size_lutic;
__constant__ int size_lutec;
__constant__ int size_lutiso;

// =============================================================================================
// Pointers to GPU Memory
// =============================================================================================

// data IC
static uint32_t*  gpu_voxelIC;
static uint32_t*  gpu_fiberIC;
static uint16_t*  gpu_orientIC;
static float32_t* gpu_lengthIC;
//static compartmentIC_t* gpu_compartmentsIC;
static uint32_t*        gpu_segmentsPerBlockIC;
static uint32_t*        gpu_offsetPerBlockIC;

static uint32_t*  gpu_voxelICt;
static uint32_t*  gpu_fiberICt;
static uint16_t*  gpu_orientICt;
static float32_t* gpu_lengthICt;
//static compartmentIC_t* gpu_compartmentsIC_trans;
static uint32_t*        gpu_segmentsPerBlockICt;
static uint32_t*        gpu_offsetPerBlockICt;

// data EC
static uint32_t*  gpu_voxelEC;
static uint16_t*  gpu_orientEC;
//static compartmentEC_t* gpu_compartmentsEC;
static uint32_t*        gpu_segmentsPerBlockEC;
static uint32_t*        gpu_offsetPerBlockEC;

// look-up tables
static float32_t* gpu_lutIC;
static float32_t* gpu_lutEC;
static float32_t* gpu_lutISO;

// vectors x and y
static float64_t* gpu_x;
static float64_t* gpu_y;

// =============================================================================================
// Textures in the GPU
// =============================================================================================

texture<float32_t, 1, cudaReadModeElementType> tex_lutIC;
texture<float32_t, 1, cudaReadModeElementType> tex_lutEC;
texture<float32_t, 1, cudaReadModeElementType> tex_lutISO;

// =============================================================================================
// Preprocessing Functions
// =============================================================================================

void preprocessDataForGPU(uint32_t* data, int NUM_COMPARTMENTS, uint32_t* compartmentsPerBlock, uint32_t* offsetPerBlock, int NUM_BLOCKS){

    // fill arrays with zeros
    memset(compartmentsPerBlock, 0, NUM_BLOCKS * sizeof(uint32_t));
    memset(offsetPerBlock,       0, NUM_BLOCKS * sizeof(uint32_t));

    // count compartments per block
    for(int i = 0; i < NUM_COMPARTMENTS; i++)
        compartmentsPerBlock[data[i]]++;

    // calculate offset per block
    offsetPerBlock[0] = 0;
    for(int i = 1; i < NUM_BLOCKS; i++)
        offsetPerBlock[i] = offsetPerBlock[i-1] + compartmentsPerBlock[i-1];
}

// =============================================================================================
// Kernels Operation Ax
// =============================================================================================

// operation Ax
/*__global__ void multiply_AX_ICpart(segment_t* segments, uint32_t* segmentsPerBlock, uint32_t* offsetPerBlock, float32_t* lut, float64_t* x, float64_t* y);
__global__ void multiply_AX_ECpart(excomp_t*  excomps,  uint32_t* segmentsPerBlock, uint32_t* offsetPerBlock, float32_t* lut, float64_t* x, float64_t* y);
__global__ void multiply_AX_ISOpart(float32_t* lut, float64_t* x, float64_t* y);

// =============================================================================================
// Kernels Operation A'y
// =============================================================================================

// operation A'y
__global__ void multiply_ATY_ICpart(segment_t* segments, uint32_t* segmentsPerBlock, uint32_t* offsetPerBlock, float32_t* lut, float64_t* x, float64_t* y);
__global__ void multiply_ATY_ECpart(excomp_t*  excomps,  uint32_t* segmentsPerBlock, uint32_t* offsetPerBlock, float32_t* lut, float64_t* x, float64_t* y);
__global__ void multiply_ATY_ISOpart(float32_t* lut, float64_t* x, float64_t* y);//*/

// =============================================================================================
// Callable Functions from Python
// =============================================================================================
extern "C" {
    void set_globals(int nsegments,
                     int nvoxels,
                     int nfibers,
                     int nexcomps,
                     int norientations,
                     int nsamples,
                     int nresfunic,
                     int nresfunec,
                     int nresfuniso){

        // copy constant values to CPU
        NUM_SEGMENTS = nsegments;
        NUM_VOXELS   = nvoxels;
        NUM_FIBERS   = nfibers;
        NUM_EXCOMPS  = nexcomps;
        NUM_ORIENTATIONS = norientations;
        NUM_SAMPLES = nsamples;
        NUM_RESFUNCIC = nresfunic;
        NUM_RESFUNCEC = nresfunec;
        NUM_RESFUNCISO = nresfuniso;
        NUM_ROWS = NUM_VOXELS * NUM_SAMPLES;
        NUM_COLS = NUM_FIBERS * NUM_RESFUNCIC + NUM_EXCOMPS * NUM_RESFUNCEC + NUM_VOXELS * NUM_RESFUNCISO;
        SIZE_LUTIC  = NUM_RESFUNCIC  * NUM_ORIENTATIONS * NUM_SAMPLES;
        SIZE_LUTEC  = NUM_RESFUNCEC  * NUM_ORIENTATIONS * NUM_SAMPLES;
        SIZE_LUTISO = NUM_RESFUNCISO * NUM_SAMPLES;

        printf("\t* constant global values ... ");
        status = true;
        status = status && cudaCheck( cudaMemcpyToSymbol(num_voxels,       &NUM_VOXELS,       sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_fibers,       &NUM_FIBERS,       sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_excomps,      &NUM_EXCOMPS,      sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_orientations, &NUM_ORIENTATIONS, sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_samples,      &NUM_SAMPLES,      sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_resfuncic,    &NUM_RESFUNCIC,    sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_resfuncec,    &NUM_RESFUNCEC,    sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_resfunciso,   &NUM_RESFUNCISO,   sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_rows,         &NUM_ROWS,         sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(num_cols,         &NUM_COLS,         sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(size_lutic,       &SIZE_LUTIC,       sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(size_lutec,       &SIZE_LUTEC,       sizeof(int)) );
        status = status && cudaCheck( cudaMemcpyToSymbol(size_lutiso,      &SIZE_LUTISO,      sizeof(int)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* memory for vectors x and y ... ");
        status = true;
        status = status && cudaCheck( cudaMalloc((void**)&gpu_x, NUM_COLS * sizeof(float64_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_y, NUM_ROWS * sizeof(float64_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");
    }
}

extern "C" {
    void set_ic_lut(float32_t* lut) {

        printf("\t* memory for LUT (IC part) ... ");
        status = true;
        status = status && cudaCheck( cudaMalloc((void**)&gpu_lutIC, SIZE_LUTIC * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* Copying LUT in GPU (IC part) ... ");
        status = true;
        status = status && cudaCheck( cudaMemcpy(gpu_lutIC, lut, SIZE_LUTIC * sizeof(float32_t), cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // config texture for look-up table IC
        tex_lutIC.addressMode[0] = cudaAddressModeBorder;
        tex_lutIC.addressMode[1] = cudaAddressModeBorder;
        tex_lutIC.filterMode = cudaFilterModePoint;
        tex_lutIC.normalized = false;

        printf("\t* Linking LUT to a texture (IC  part) ... ");
        status = cudaCheck( cudaBindTexture(NULL, tex_lutIC,  gpu_lutIC,  SIZE_LUTIC * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");
    }
}

extern "C" {
    void set_ec_lut(float32_t* lut) {

        printf("\t* Allocating memory for LUT in GPU (EC part) ... ");
        status = cudaCheck( cudaMalloc((void**)&gpu_lutEC, SIZE_LUTEC * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* Copying LUT in GPU (EC part) ... ");
        status = cudaCheck( cudaMemcpy(gpu_lutEC, lut, SIZE_LUTEC * sizeof(float32_t), cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // config texture for look-up table EC
        tex_lutEC.addressMode[0] = cudaAddressModeBorder;
        tex_lutEC.addressMode[1] = cudaAddressModeBorder;
        tex_lutEC.filterMode = cudaFilterModePoint;
        tex_lutEC.normalized = false;

        printf("\t* Linking LUT to a texture (EC  part) ... ");
        status = cudaCheck( cudaBindTexture(NULL, tex_lutEC,  gpu_lutEC,  SIZE_LUTEC * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");
    }
}

extern "C" {
    void set_iso_lut(float32_t* lut) {

        printf("\t* Allocating memory for LUT in GPU (ISO part) ... ");
        status = cudaCheck( cudaMalloc((void**)&gpu_lutISO, SIZE_LUTISO * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* Copying LUT in GPU (ISO part) ... ");
        status = cudaCheck( cudaMemcpy(gpu_lutISO, lut, SIZE_LUTISO * sizeof(float32_t), cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // config texture for look-up table ISO
        tex_lutISO.addressMode[0] = cudaAddressModeBorder;
        tex_lutISO.addressMode[1] = cudaAddressModeBorder;
        tex_lutISO.filterMode = cudaFilterModePoint;
        tex_lutISO.normalized = false;

        printf("\t* Linking LUT to a texture (ISO  part) ... ");
        status = cudaCheck( cudaBindTexture(NULL, tex_lutISO, gpu_lutISO, SIZE_LUTISO * sizeof(float32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");
    }
}

extern "C" {
    void set_ic_data(uint32_t* voxel, uint32_t* fiber, uint16_t* orient, float32_t* length) {

        /*printf("\t* Allocating auxiliar memory in CPU ... \n");
        status = true;//*/
        uint32_t*  segmentsPerBlock = (uint32_t*) malloc(NUM_VOXELS * sizeof(uint32_t));
        uint32_t*  offsetPerBlock   = (uint32_t*) malloc(NUM_VOXELS * sizeof(uint32_t));
        /*status = status && (segmentsPerBlock == NULL) && (offsetPerBlock == NULL);
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");//*/

        preprocessDataForGPU(voxel, NUM_SEGMENTS, segmentsPerBlock, offsetPerBlock, NUM_VOXELS);

        printf("\t* fiber segments memory allocation ... ");
        status = true;
        status = status && cudaCheck( cudaMalloc((void**)&gpu_voxelIC,  NUM_SEGMENTS * sizeof(uint32_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_fiberIC,  NUM_SEGMENTS * sizeof(uint32_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_orientIC, NUM_SEGMENTS * sizeof(uint16_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_lengthIC, NUM_SEGMENTS * sizeof(float32_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_segmentsPerBlockIC, NUM_VOXELS * sizeof(uint32_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_offsetPerBlockIC,   NUM_VOXELS * sizeof(uint32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* transfering fiber segments ... ");
        status = true;
        status = status && cudaCheck( cudaMemcpy(gpu_voxelIC,  voxel,  NUM_SEGMENTS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_fiberIC,  fiber,  NUM_SEGMENTS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_orientIC, orient, NUM_SEGMENTS * sizeof(uint16_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_lengthIC, length, NUM_SEGMENTS * sizeof(float32_t), cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_segmentsPerBlockIC, segmentsPerBlock, NUM_VOXELS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_offsetPerBlockIC,   offsetPerBlock,   NUM_VOXELS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // delete auxiliar CPU memory
        free(segmentsPerBlock);
        free(offsetPerBlock);
    }
}

extern "C" {
    void set_ic_data_transpose(uint32_t* voxel, uint32_t* fiber, uint16_t* orient, float32_t* length) {
        
        // alloc auxiliar memory in the CPU
        uint32_t*  segmentsPerBlock = (uint32_t*)  malloc(NUM_FIBERS * sizeof(uint32_t));
        uint32_t*  offsetPerBlock   = (uint32_t*)  malloc(NUM_FIBERS * sizeof(uint32_t));
        /*status = status && (segmentsPerBlock == NULL) && (offsetPerBlock == NULL);
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");//*/

        preprocessDataForGPU(fiber, NUM_SEGMENTS, segmentsPerBlock, offsetPerBlock, NUM_FIBERS);

        // alloc memory in the GPU
        printf("\t* extra memory for operator A' ... ");
        status = true;
        status = status && cudaCheck( cudaMalloc((void**)&gpu_voxelICt,  NUM_SEGMENTS * sizeof(uint32_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_fiberICt,  NUM_SEGMENTS * sizeof(uint32_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_orientICt, NUM_SEGMENTS * sizeof(uint16_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_lengthICt, NUM_SEGMENTS * sizeof(float32_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_segmentsPerBlockICt, NUM_FIBERS * sizeof(uint32_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_offsetPerBlockICt,   NUM_FIBERS * sizeof(uint32_t)) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* transfering extra memory for operator A' ... ");
        status = true;
        status = status && cudaCheck( cudaMemcpy(gpu_voxelICt,  voxel,  NUM_SEGMENTS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_fiberICt,  fiber,  NUM_SEGMENTS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_orientICt, orient, NUM_SEGMENTS * sizeof(uint16_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_lengthICt, length, NUM_SEGMENTS * sizeof(float32_t), cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_segmentsPerBlockICt, segmentsPerBlock, NUM_FIBERS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_offsetPerBlockICt,   offsetPerBlock,   NUM_FIBERS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // delete auxilar data in CPU
        free(segmentsPerBlock);
        free(offsetPerBlock);
    }
}

extern "C" {
    void set_ec_data(uint32_t* voxel, uint16_t* orient) {

        // alloc temporal memory in the CPU
        uint32_t* segmentsPerBlock = (uint32_t*) malloc(NUM_VOXELS * sizeof(uint32_t));
        uint32_t* offsetPerBlock   = (uint32_t*) malloc(NUM_VOXELS * sizeof(uint32_t));

        // alloc memory in the GPU
        printf("\t* Allocating memory for operator A in GPU (EC part) ... ");
        status = true;
        status = status && cudaCheck( cudaMalloc((void**)&gpu_voxelEC,  NUM_EXCOMPS * sizeof(uint32_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_orientEC, NUM_EXCOMPS * sizeof(uint16_t)) );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_segmentsPerBlockEC, NUM_VOXELS * sizeof(uint32_t))  );
        status = status && cudaCheck( cudaMalloc((void**)&gpu_offsetPerBlockEC,   NUM_VOXELS * sizeof(uint32_t))  );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // preprocess EC data for the GPU
        preprocessDataForGPU(voxel, NUM_EXCOMPS, segmentsPerBlock, offsetPerBlock, NUM_VOXELS);

        // copy EC data to the GPU
        printf("\t* Copying operator A to GPU (EC part) ... ");
        status = true;
        status = status && cudaCheck( cudaMemcpy(gpu_voxelEC,            voxel,                NUM_EXCOMPS * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_orientEC,           orient,               NUM_EXCOMPS * sizeof(uint16_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_segmentsPerBlockEC, segmentsPerBlock,     NUM_VOXELS  * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        status = status && cudaCheck( cudaMemcpy(gpu_offsetPerBlockEC,   offsetPerBlock,       NUM_VOXELS  * sizeof(uint32_t),  cudaMemcpyHostToDevice) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        // delete auxiliar memory in CPU
        free(segmentsPerBlock);
        free(offsetPerBlock);
    }
}

extern "C" {
    void set_iso_data() {
        
    }
}

extern "C" {
    void free_data() {
        printf("-> Deleting memory in GPU:\n");

        printf("\t* A  operator (IC  part) ... ");
        status = true;
        status = status && cudaCheck( cudaFree(gpu_voxelIC) );
        status = status && cudaCheck( cudaFree(gpu_fiberIC) );
        status = status && cudaCheck( cudaFree(gpu_orientIC) );
        status = status && cudaCheck( cudaFree(gpu_lengthIC) );
        status = status && cudaCheck( cudaFree(gpu_segmentsPerBlockIC) );
        status = status && cudaCheck( cudaFree(gpu_offsetPerBlockIC) );
        status = status && cudaCheck( cudaFree(gpu_lutIC) );
        status = status && cudaCheck( cudaUnbindTexture(tex_lutIC) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* A  operator (EC  part) ... ");
        status = true;
        status = status && cudaCheck( cudaFree(gpu_voxelEC) );
        status = status && cudaCheck( cudaFree(gpu_orientEC) );
        status = status && cudaCheck( cudaFree(gpu_segmentsPerBlockEC) );
        status = status && cudaCheck( cudaFree(gpu_offsetPerBlockEC) );
        status = status && cudaCheck( cudaFree(gpu_lutEC) );
        status = status && cudaCheck( cudaUnbindTexture(tex_lutEC) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* A  operator (ISO part) ... ");
        status = true;
        status = status && cudaCheck( cudaFree(gpu_lutISO) );
        status = status && cudaCheck( cudaUnbindTexture(tex_lutISO) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* A' operator (IC  part) ... ");
        status = true;
        status = status && cudaCheck( cudaFree(gpu_voxelICt) );
        status = status && cudaCheck( cudaFree(gpu_fiberICt) );
        status = status && cudaCheck( cudaFree(gpu_orientICt) );
        status = status && cudaCheck( cudaFree(gpu_lengthICt) );
        status = status && cudaCheck( cudaFree(gpu_segmentsPerBlockICt) );
        status = status && cudaCheck( cudaFree(gpu_offsetPerBlockICt) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* x and y vectors ... ");
        status = true;
        status = status && cudaCheck( cudaFree(gpu_x) );
        status = status && cudaCheck( cudaFree(gpu_y) );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\t* reseting GPU ... ");
        status = true;
        status = status && cudaCheck( cudaDeviceReset() );
        if (status) printf("[ OK ]\n");
        else        printf("[ ERROR ]\n");

        printf("\n");
    }
}

#endif //INCLUDE_OPERATOR_WITHCUDA_CUH