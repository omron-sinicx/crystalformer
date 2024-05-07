#include "models/kernels/pairwise_sum.cuh"

extern "C" __global__
void fused_dpa_fwd_v3(
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
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;
    const long long int i = tid/H;
    const long long int h = tid%H;
    const long long int e_start = e_start_i[i];
    const long long int e_end = e_start_i[i+1];

    // Load q onto shared mem.
    constexpr int DIM_MAX = (V_HEAD_DIM>K_HEAD_DIM) ? V_HEAD_DIM : K_HEAD_DIM;
    __shared__ float _vec_k[THREAD_NUM][DIM_MAX+1];
    que_ihk += tid*K_HEAD_DIM;
    float* que_k = _vec_k[threadIdx.x];
    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++){
        que_k[k] = que_ihk[k];
    }
    
    __shared__ float _attns[THREAD_NUM][MAX_SYSTEM_SIZE+1];
    __shared__ float _output[THREAD_NUM][V_HEAD_DIM+1];
    float *attns = _attns[threadIdx.x];
    float *output = _output[threadIdx.x];

    float max_attn = -1e20;
    int e_count = 0;
    for (long long int e = e_start; e < e_end; e++)
    {
        long long int j = edge_ij_e[E+e];
        const float* key_k = &key_ihk[(j*H+h)*K_HEAD_DIM];

        float attn = 0;
        #pragma unroll
        for (int k = 0; k < K_HEAD_DIM; k++){
            attn += que_k[k]*key_k[k];
        }
        if (aij_eh != NULL)
           attn += aij_eh[e*H+h];
        max_attn = max(max_attn, attn);
        attns[e_count] = attn;
        e_count++;
    }
    
    for (int j = 0; j < e_count; j++)
    {
        float v = exp(attns[j] - max_attn);
        attns[j] = v;
    }

    // Compute pairwise sum for better numerical accuracy.
    constexpr int BS = 64;
    for (int j = e_count; j < (e_count+BS-1)/BS*BS; j++)
        attns[j] = 0.0f;
    float s[MAX_SYSTEM_SIZE_POW2/BS] = {0.0f};
    for (int j = 0; j < (e_count+BS-1)/BS; j++){
        s[j] = fixed_length_pairwise_sum<BS>(&attns[j*BS]);
    }
    float sum = fixed_length_pairwise_sum<MAX_SYSTEM_SIZE_POW2/BS>(s);
    

    for (int j = 0; j < e_count; j++){
        attns[j] /= sum;
        prob_eh[(e_start+j)*H+h] = attns[j];
    }

    float *short_run_sum = _vec_k[threadIdx.x];
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++){
        output[k] = 0;
        short_run_sum[k] = 0;
    }

    e_count = 0;
    if (bij_ehk != NULL) {
        for (long long int e = e_start; e < e_end; e++)
        {
            long long int j = edge_ij_e[E+e];
            const float* val_k = &val_ihk[(j*H+h)*V_HEAD_DIM];
            const float* bij_k = &bij_ehk[(e*H+h)*V_HEAD_DIM];

            float attn = attns[e_count];
            #pragma unroll
            for (int k = 0; k < V_HEAD_DIM; k++){
                short_run_sum[k] += (val_k[k]+bij_k[k])*attn;
            }
            e_count++;

            if ((e_count % RUNNING_SUM_LEN) == 0 || e == e_end-1){
                #pragma unroll
                for (int k = 0; k < V_HEAD_DIM; k++){
                    output[k] += short_run_sum[k];
                    short_run_sum[k] = 0;
                }
            }
        }
    } else {
        for (long long int e = e_start; e < e_end; e++)
        {
            long long int j = edge_ij_e[E+e];
            const float* val_k = &val_ihk[(j*H+h)*V_HEAD_DIM];

            float attn = attns[e_count];
            #pragma unroll
            for (int k = 0; k < V_HEAD_DIM; k++){
                short_run_sum[k] += val_k[k]*attn;
            }
            e_count++;

            if ((e_count % RUNNING_SUM_LEN) == 0 || e == e_end-1){
                #pragma unroll
                for (int k = 0; k < V_HEAD_DIM; k++){
                    output[k] += short_run_sum[k];
                    short_run_sum[k] = 0;
                }
            }
        }
    }

    out_ihk += tid*V_HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++)
        out_ihk[k] = output[k];
}