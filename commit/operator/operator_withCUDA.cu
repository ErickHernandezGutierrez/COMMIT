#ifndef INCLUDE_OPERATOR_CUDA_CUH
#define INCLUDE_OPERATOR_CUDA_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include "operator_withCUDA.cuh"

__global__ void multiply_Ax_ICpart(uint32_t*  voxelIC,
                                   uint32_t*  fiberIC,
                                   uint16_t*  orientationIC,
                                   float32_t* lengthIC,
                                   uint32_t*  segmentsPerBlock,
                                   uint32_t*  offsetPerBlock,
                                   float32_t* lut,
                                   float64_t* x,
                                   float64_t* y){

    __shared__ float64_t shmem[1024];
    
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x / 512;
    uint32_t sid = threadIdx.x - 512*gid;

    shmem[tid] = 0.0;

    if(sid >= num_samples) return;

    uint32_t offset = offsetPerBlock[bid] + (segmentsPerBlock[bid]/2)*gid;
    uint32_t num_segments = segmentsPerBlock[bid]/2 + (segmentsPerBlock[bid]%2)*gid;

    //segment_t* segment = segments + offset;
    uint32_t*  voxel = voxelIC + offset;
    uint32_t*  fiber = fiberIC + offset;
    uint16_t*  orientation = orientationIC + offset;
    float32_t* length = lengthIC + offset;

    float64_t sum = 0.0;

    for(int i = 0; i < num_segments; i++){
        int offset_lut = (*orientation)*num_samples + sid;

        float64_t aux = 0.0;
        for(int j = 0; j < num_resfuncic; j++){
            //aux += (double)(lut[lut_offset + j*num_orientations*num_samples])*x[(segment->fiber) + j*num_fibers];
            aux += tex1Dfetch(tex_lutIC, offset_lut + j*num_orientations*num_samples) * x[(*fiber) + j*num_fibers];
        }

        sum += aux * (*length);

        fiber++;
        orientation++;
        length++;
    }

    shmem[tid] = sum;
    __syncthreads();

    if(tid < num_samples)
        y[(*voxel)*num_samples + sid] = sum + shmem[tid+512];
}

__global__ void multiply_Ax_ECpart(uint32_t*  voxelEC,
                                   uint16_t*  orientationEC,
                                   uint32_t*  segmentsPerBlock,
                                   uint32_t*  offsetPerBlock,
                                   float32_t* lut,
                                   float64_t* x,
                                   float64_t* y) {
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if(tid >= num_samples) return;

    uint32_t offset  = offsetPerBlock[bid];
    uint32_t num_segments = segmentsPerBlock[bid];

    //compartmentEC_t* excomp = excomps + offset;
    uint32_t* voxel = voxelEC + offset;
    uint16_t* orientation = orientationEC + offset;

    uint32_t target = num_fibers*num_resfuncic + offset;

    float64_t sum = 0.0;
    for(int i = 0; i < num_segments; i++){
        uint32_t offset_lut = (*orientation)*num_samples + tid;

        for(int j = 0; j < num_resfuncec; j++)
            //sum += (double)(lut[lut_offset + j*num_orientations*num_samples])*x[target + j*num_excomps + i];
            sum += tex1Dfetch(tex_lutEC, offset_lut + j*num_orientations*num_samples) * x[target + j*num_excomps + i];

        orientation++;
    }    

    y[(*voxel)*num_samples + tid] += sum;
}

__global__ void multiply_Ax_ISOpart(float* lut, double* x, double* y) {
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if(tid >= num_samples) return;

    uint32_t target = num_fibers*num_resfuncic + num_excomps*num_resfuncec + bid;

    float64_t sum = 0.0;
    for(int j = 0; j < num_resfunciso; j++)
        //sum += (double)(lut[j*num_samples + tid])*x[target + j*num_voxels];
        sum += (double)(tex1Dfetch(tex_lutISO, j*num_samples + tid))*x[target + j*num_voxels];
        

    y[bid*num_samples + tid] += sum;
}

__global__ void multiply_Aty_ICpart(uint32_t*  voxelICt,
                                    uint32_t*  fiberICt,
                                    uint16_t*  orientICt,
                                    float32_t* lengthICt,
                                    uint32_t*  compartmentsPerBlock,
                                    uint32_t*  offsetPerBlock,
                                    float32_t* lut,
                                    float64_t* x,
                                    float64_t* y){
    __shared__ float64_t shmem[512];
    
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    shmem[tid] = 0.0;

    if(tid >= num_samples) return;

    /*if(bid == 0 && tid == 0){
        for(int i = 0; i < 10; i++){
            printf("%d %d %d %f\n", voxelICt[i], fiberICt[i], orientICt[i], lengthICt[i]);
        }
    }
    else if(bid != 0) return;
    //__syncthreads();//*/

    uint32_t offset = offsetPerBlock[bid];
    uint32_t num_segments = offset + compartmentsPerBlock[bid];

    //segment_t* segment = segments + offset;
    uint32_t*  voxel  = voxelICt  + offset;
    uint32_t*  fiber  = fiberICt  + offset;
    uint16_t*  orient = orientICt + offset;
    float32_t* length = lengthICt + offset;
    //uint fiber = segment->fiber;

    for(int j = 0; j < num_resfuncic; j++){
        int offset_lut = j*num_orientations*num_samples + tid;

        float64_t sum = 0.0;
        //segment = segments + offset;
        voxel  = voxelICt  + offset;
        orient = orientICt + offset;
        length = lengthICt + offset;
        for(int i = offset; i < num_segments; i++){
            //sum += (segment->length) * lut[lut_offset + (segment->orientation)*num_samples] * y[(segment->voxel)*num_samples + tid];
            sum += ((float64_t)(*length)) *( (float64_t) tex1Dfetch(tex_lutIC, offset_lut + (*orient)*num_samples) )* y[(*voxel)*num_samples + tid];
            //segment++;
            voxel++;
            //fiber++;
            orient++;
            length++;
        }

        shmem[tid] = sum;
        __syncthreads();
        
        if(tid < 256) shmem[tid] += shmem[tid + 256]; __syncthreads();
        if(tid < 128) shmem[tid] += shmem[tid + 128]; __syncthreads();
        if(tid <  64) shmem[tid] += shmem[tid +  64]; __syncthreads();
        if(tid <  32) shmem[tid] += shmem[tid +  32]; __syncthreads();
        if(tid <  16) shmem[tid] += shmem[tid +  16]; __syncthreads();
        if(tid <   8) shmem[tid] += shmem[tid +   8]; __syncthreads();
        if(tid <   4) shmem[tid] += shmem[tid +   4]; __syncthreads();
        //if(tid <   2) shmem[tid] += shmem[tid +   2]; __syncthreads();

        if(tid == 0) x[j*num_fibers + (*fiber)] = shmem[0] + shmem[1] + shmem[2] + shmem[3];

        __syncthreads();
    }
}

__global__ void multiply_Aty_ECpart(uint32_t* voxelEC,
                                    uint16_t* orientEC,
                                    uint32_t* segmentsPerBlock,
                                    uint32_t* offsetPerBlock,
                                    float32_t* lut,
                                    float64_t* x,
                                    float64_t* y){
    __shared__ float64_t shmem[512];
    
    uint32_t bid = blockIdx.x;
    uint32_t tid = threadIdx.x;

    shmem[tid] = 0.0;

    if(tid >= num_samples) return;

    uint32_t offset  = offsetPerBlock[bid];
    uint32_t num_compartments = segmentsPerBlock[bid] + offset;

    //compartmentEC_t* peak = peaks + offset;
    uint32_t* voxel = voxelEC + offset;
    uint16_t* orient = orientEC + offset;

    for(int j = 0; j < num_resfuncec; j++){        
        uint32_t offset_lut = j*num_orientations*num_samples + tid;

        //peak = peaks + offset;
        voxel = voxelEC + offset;
        orient = orientEC + offset;
        for(int i = offset; i < num_compartments; i++){
            shmem[tid] =( (float64_t)tex1Dfetch(tex_lutEC, (*orient)*num_samples + offset_lut) )* y[(*voxel)*num_samples + tid];
            //shmem[tid] = tex1Dfetch(tex_lutEC, (peak->orientation)*num_samples + lut_offset) * y[(peak->voxel)*num_samples + tid];
            //shmem[tid] = lut[(peak->orientation)*num_samples + lut_offset] * y[(peak->voxel)*num_samples + tid];
            __syncthreads();
            
            //if(bid == 0){
                //printf("%lf\n", lut[(peak->orientation)*num_samples + lut_offset] * y[(peak->voxel)*num_samples + tid]);

                if(tid < 256) shmem[tid] += shmem[tid + 256]; __syncthreads();
                if(tid < 128) shmem[tid] += shmem[tid + 128]; __syncthreads();
                if(tid <  64) shmem[tid] += shmem[tid +  64]; __syncthreads();
                if(tid <  32) shmem[tid] += shmem[tid +  32]; __syncthreads();
                if(tid <  16) shmem[tid] += shmem[tid +  16]; __syncthreads();
                if(tid <   8) shmem[tid] += shmem[tid +   8]; __syncthreads();
                if(tid <   4) shmem[tid] += shmem[tid +   4]; __syncthreads();
                if(tid <   2) shmem[tid] += shmem[tid +   2]; __syncthreads();

                if(tid == 0) x[num_fibers*num_resfuncic + j*num_excomps + i] = shmem[0] + shmem[1];
            //}

            //peak++;
            voxel++;
            orient++;
            __syncthreads();
        }
    }
} //*/

__global__ void multiply_Aty_ISOpart(float* lut, double* x, double* y){
    __shared__ double shmem[512];

    uint bid = blockIdx.x;
    uint tid = threadIdx.x;
    uint offset = num_fibers*num_resfuncic + num_excomps*num_resfuncec + bid;

    shmem[tid] = 0.0;

    if(tid >= num_samples) return;

    for(int j = 0; j < num_resfunciso; j++){
        //shmem[tid] = (double)(lut[j*num_samples + tid]) * y[bid*num_samples + tid];
        shmem[tid] =( (float64_t) tex1Dfetch(tex_lutISO, j*num_samples + tid) )* y[bid*num_samples + tid];
        __syncthreads();

        if(tid < 256) shmem[tid] += shmem[tid + 256]; __syncthreads();
        if(tid < 128) shmem[tid] += shmem[tid + 128]; __syncthreads();
        if(tid <  64) shmem[tid] += shmem[tid +  64]; __syncthreads();
        if(tid <  32) shmem[tid] += shmem[tid +  32]; __syncthreads();
        if(tid <  16) shmem[tid] += shmem[tid +  16]; __syncthreads();
        if(tid <   8) shmem[tid] += shmem[tid +   8]; __syncthreads();
        if(tid <   4) shmem[tid] += shmem[tid +   4]; __syncthreads(); 

        if(tid == 0)
            x[offset + j*num_voxels] = shmem[0] + shmem[1] + shmem[2] + shmem[3];
    }
}//*/

extern "C"{
    void multiply_Ax(double* x, double* y){

        // Copy vector x to the GPU
        cudaMemcpy(gpu_x, x, NUM_COLS*sizeof(double), cudaMemcpyHostToDevice);

        // Multiply IC part in the GPU
        multiply_Ax_ICpart<<<NUM_VOXELS, 1024>>>(gpu_voxelIC, gpu_fiberIC, gpu_orientIC, gpu_lengthIC, gpu_segmentsPerBlockIC, gpu_offsetPerBlockIC, gpu_lutIC, gpu_x, gpu_y);

        //cudaCheckKernel();

        // Multiply EC part in the GPU
        multiply_Ax_ECpart<<<NUM_VOXELS, 512>>>(gpu_voxelEC, gpu_orientEC, gpu_segmentsPerBlockEC, gpu_offsetPerBlockEC, gpu_lutEC, gpu_x, gpu_y);

        //cudaCheckKernel();

        // Multiply ISO part in the GPU
        multiply_Ax_ISOpart<<<NUM_VOXELS, 512>>>(gpu_lutISO, gpu_x, gpu_y);

        //cudaCheckKernel();

        // Copy back result to CPU
        cudaMemcpy(y, gpu_y, NUM_ROWS*sizeof(double), cudaMemcpyDeviceToHost);
    }
}

extern "C"{
    void multiply_Aty(double* y, double* x){
        
        // Copy vector y to the GPU
        //cudaCheck( cudaMemset(gpu_x, 0, NUM_COLS*sizeof(float64_t)) );
        //cudaCheck( cudaMemcpy(gpu_x, x, NUM_COLS*sizeof(double), cudaMemcpyHostToDevice) );
        cudaCheck( cudaMemcpy(gpu_y, y, NUM_ROWS*sizeof(double), cudaMemcpyHostToDevice) );

        // Multiply IC part in the GPU
        multiply_Aty_ICpart<<<NUM_FIBERS, 512>>>(gpu_voxelICt, gpu_fiberICt, gpu_orientICt, gpu_lengthICt, gpu_segmentsPerBlockICt, gpu_offsetPerBlockICt, gpu_lutIC, gpu_x, gpu_y);

        //cudaCheckKernel();//*/

        // Multiply EC part in the GPU
        multiply_Aty_ECpart<<<NUM_VOXELS, 512>>>(gpu_voxelEC, gpu_orientEC, gpu_segmentsPerBlockEC, gpu_offsetPerBlockEC, gpu_lutEC, gpu_x, gpu_y);

        //cudaCheckKernel();

        // Multiply ISO part in the GPU
        multiply_Aty_ISOpart<<<NUM_VOXELS, 512>>>(gpu_lutISO, gpu_x, gpu_y);

        //cudaCheckKernel();//*/

        // Copy back result to CPU
        cudaCheck( cudaMemcpy(x, gpu_x, NUM_COLS*sizeof(double), cudaMemcpyDeviceToHost) ); 
            
        /*printf("\n\n VECTOR X EC PART:\n");
        for(int i = NUM_FIBERS*NUM_RESFUNCIC; i < NUM_FIBERS*NUM_RESFUNCIC+20; i++)
            printf("%lf ", x[i]);
        printf("\n\n");//*/
    }
}

#endif //INCLUDE_OPERATOR_CUDA_CUH