#include "models/kernels/reduce_kernel_utils.cuh"

__global__ __device__ void fused_dpa_fwd_thread(
    const float* que_k,
    const float* key_ihk,
    const float* val_ihk,
    const float* aij_eh,
    const float* bij_ehk,
    const long long int* edge_j_e,
    const long long int e_start,
    const long long int e_end,
    const long long int H,
    float* prob_eh,
    float* out_k
) {
    long long int e = e_start + threadIdx.x;
    bool isValid = e < e_end;

    float attn;
    long long int j = 0;
    if (isValid){
        j = edge_j_e[e];
        key_ihk += (j*H)*K_HEAD_DIM;

        attn = 0;
        #pragma unroll
        for (int k = 0; k < K_HEAD_DIM; k++){
            attn += que_k[k]*key_ihk[k];
        }
        if (aij_eh != NULL) attn += aij_eh[e*H];
    } else {
        attn = -1e20;
    }
    __syncthreads();
    
    float max_attn = blockReduceMax(attn);

    attn = exp(attn - max_attn);
    float sum = blockReduceSum(attn);

    attn /= sum;
    if (isValid)
        prob_eh[e*H] = attn;

    if (bij_ehk != NULL) {
        val_ihk += (j*H)*V_HEAD_DIM;
        bij_ehk += (e*H)*V_HEAD_DIM;

        #pragma unroll
        for (int k = 0; k < V_HEAD_DIM; k++){
            float output = isValid ? (val_ihk[k]+bij_ehk[k])*attn : 0.0f;
            __syncthreads();
            output = blockReduceSum(output);
            if (threadIdx.x == 0){
                out_k[k] = output;
            }
        }
    } else {
        val_ihk += (j*H)*V_HEAD_DIM;

        #pragma unroll
        for (int k = 0; k < V_HEAD_DIM; k++){
            float output = isValid ? (val_ihk[k])*attn : 0.0f;
            __syncthreads();
            output = blockReduceSum(output);
            if (threadIdx.x == 0)
                out_k[k] = output;
        }
    }
}

extern "C" __global__
void fused_dpa_fwd_v2(
    const float* que_ihk,
    const float* key_ihk,
    const float* val_ihk,
    const float* aij_eh,
    const float* bij_ehk,
    const long long int* edge_ij_e, 
    const long long int* e_start_i,
    const long long int N,
    const long long int H,
    const long long int E,
    float* prob_eh,
    float* out_ihk
){
    const long long int tid = (long long int)blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= N*H) return;
    
    const long long int i = tid / H;
    const long long int h = tid % H;
    const long long int e_start = e_start_i[i];
    const long long int e_end = e_start_i[i+1];

    fused_dpa_fwd_thread<<< 1, ((e_end-e_start+31)/32)*32 >>>(
        que_ihk + tid*K_HEAD_DIM,
        key_ihk + h*K_HEAD_DIM,
        val_ihk + h*V_HEAD_DIM,
        aij_eh  ? aij_eh + h : aij_eh,
        bij_ehk ? bij_ehk+ h*V_HEAD_DIM : bij_ehk,
        edge_ij_e + E,
        e_start,
        e_end,
        H,
        prob_eh + h,
        out_ihk + tid*V_HEAD_DIM
    );

}