#include "models/kernels/reduce_kernel_utils.cuh"

__global__ __device__ void fused_dpa_bwd_q_thread(
    const float* key_ihk,
    const float* gaij_eh,
    const long long int* edge_j_e, 
    const long long int e_start,
    const long long int e_end,
    const long long int H,
    float* gque_k
){
    long long int e = e_start + threadIdx.x;
    float g_softmax = 0.0f;
    if (e < e_end){
        long long int j = edge_j_e[e];
        g_softmax = gaij_eh[e*H];
        key_ihk += (j*H)*K_HEAD_DIM;
    }

    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++){
        float gq = g_softmax*key_ihk[k];
        __syncthreads();
        gq = blockReduceSum(gq);
        if (threadIdx.x == 0)
            gque_k[k] = gq;
    }
}

extern "C" __global__
void fused_dpa_bwd_q_v2(
    const float* key_ihk,
    const float* gaij_eh,
    const long long int* edge_ij_e, 
    const long long int* e_start_i,
    const long long int N,
    const long long int H,
    const long long int E,
    float* gque_ihk
){
    const long long int tid = (long long int)blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int i = tid / H;
    const long long int h = tid % H;
    const long long int e_start = e_start_i[i];
    const long long int e_end = e_start_i[i+1];

    fused_dpa_bwd_q_thread<<< 1, ((e_end-e_start+31)/32)*32 >>>(
        key_ihk   + h*K_HEAD_DIM,
        gaij_eh   + h,
        edge_ij_e + E,
        e_start,
        e_end,
        H,
        gque_ihk  + tid*K_HEAD_DIM
    );
}