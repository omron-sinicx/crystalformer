#include "models/kernels/reduce_kernel_utils.cuh"

__global__ __device__ void fused_dpa_bwd_thread(
    const float* que_ihk,
    const float* val_k,
    const float* tbij_ehk,
    const long long int* edge_i_e, 
    const long long int e_start,
    const long long int e_end,
    const long long int H,
    const float* tprob_eh,
    const float* out_ihk,
    const float* gout_ihk,
    float* gkey_k,
    float* gval_k,
    float* tgaij_eh,
    float* tgbij_ehk
){
    long long int e = e_start + threadIdx.x;
    bool isValid = e < e_end;
    
    e = min(e, e_end-1);
    long long int i = edge_i_e[e];

    float pij  = isValid ? tprob_eh[e*H] : 0.0f;
    que_ihk   += (i*H)*K_HEAD_DIM;
    gout_ihk  += (i*H)*V_HEAD_DIM;
    out_ihk   += (i*H)*V_HEAD_DIM;
    tgbij_ehk += (e*H)*V_HEAD_DIM;
    tbij_ehk  += (e*H)*V_HEAD_DIM;

    float g_softmax = 0;
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++){
        float t = gout_ihk[k]*pij;
        g_softmax += (val_k[k] + tbij_ehk[k] - out_ihk[k]) * t;

        if (isValid)
            tgbij_ehk[k] = t;

        __syncthreads();
        t = blockReduceSum(t);
        if (threadIdx.x == 0)
            gval_k[k] = t;
    }

    if (isValid)
        tgaij_eh[e*H] = g_softmax;

    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++){
        float gk = g_softmax*que_ihk[k];
        __syncthreads();
        gk = blockReduceSum(gk);
        if (threadIdx.x == 0)
            gkey_k[k] = gk;
    }
}


__global__ __device__ void fused_dpa_bwd_thread_no_bij(
    const float* que_ihk,
    const float* val_k,
    const long long int* edge_i_e, 
    const long long int e_start,
    const long long int e_end,
    const long long int H,
    const float* tprob_eh,
    const float* out_ihk,
    const float* gout_ihk,
    float* gkey_k,
    float* gval_k,
    float* tgaij_eh
){
    long long int e = e_start + threadIdx.x;
    bool isValid = e < e_end;
    
    e = min(e, e_end-1);
    long long int i = edge_i_e[e];

    float pij  = isValid ? tprob_eh[e*H] : 0.0f;
    que_ihk   += (i*H)*K_HEAD_DIM;
    gout_ihk  += (i*H)*V_HEAD_DIM;
    out_ihk   += (i*H)*V_HEAD_DIM;

    float g_softmax = 0;
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++){
        float t = gout_ihk[k]*pij;
        g_softmax += (val_k[k] - out_ihk[k]) * t;

        __syncthreads();
        t = blockReduceSum(t);
        if (threadIdx.x == 0)
            gval_k[k] = t;
    }

    if (isValid)
        tgaij_eh[e*H] = g_softmax;

    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++){
        float gk = g_softmax*que_ihk[k];
        __syncthreads();
        gk = blockReduceSum(gk);
        if (threadIdx.x == 0)
            gkey_k[k] = gk;
    }
}


extern "C" __global__
void fused_dpa_bwd_v2(
    const float* que_ihk,
    const float* val_ihk,
    const float* tbij_ehk,
    const long long int* edge_ij_e, 
    const long long int* e_start_i,
    const long long int N,
    const long long int H,
    const long long int E,
    const float* tprob_eh,
    const float* out_ihk,
    const float* gout_ihk,
    float* gkey_ihk,
    float* gval_ihk,
    float* tgaij_eh,
    float* tgbij_ehk
){
    const long long int tid = (long long int)blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int j = tid / H;
    const long long int h = tid % H;
    const long long int e_start = e_start_i[j];
    const long long int e_end = e_start_i[j+1]; 

    if (tgbij_ehk != NULL && tbij_ehk != NULL ) {
        fused_dpa_bwd_thread<<< 1, ((e_end-e_start+31)/32)*32 >>>(
            que_ihk   + h*K_HEAD_DIM,
            val_ihk   + tid*V_HEAD_DIM,
            tbij_ehk  + h*V_HEAD_DIM,
            edge_ij_e + E, 
            e_start,
            e_end,
            H,
            tprob_eh  + h,
            out_ihk   + h*V_HEAD_DIM,
            gout_ihk  + h*V_HEAD_DIM,
            gkey_ihk  + tid*K_HEAD_DIM,
            gval_ihk  + tid*V_HEAD_DIM,
            tgaij_eh  + h,
            tgbij_ehk + h*V_HEAD_DIM
        );
    } else {
        fused_dpa_bwd_thread_no_bij<<< 1, ((e_end-e_start+31)/32)*32 >>>(
            que_ihk   + h*K_HEAD_DIM,
            val_ihk   + tid*V_HEAD_DIM,
            edge_ij_e + E, 
            e_start,
            e_end,
            H,
            tprob_eh  + h,
            out_ihk   + h*V_HEAD_DIM,
            gout_ihk  + h*V_HEAD_DIM,
            gkey_ihk  + tid*K_HEAD_DIM,
            gval_ihk  + tid*V_HEAD_DIM,
            tgaij_eh  + h
        );
    }
}