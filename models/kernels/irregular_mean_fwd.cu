#include "models/kernels/pairwise_sum.cuh"

extern "C" __global__
void irregular_mean_fwd(
    const float* src_n,
    const long long int* start_n,
    const long long int B,
    const long long int D,
    float* dst_n
){
    const unsigned long long int tid = (unsigned long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= B*D) return;
    const unsigned int n = tid/D;
    const unsigned int k = tid%D;
    const unsigned long long int start = start_n[n];
    const unsigned long long int end = start_n[n+1];
    int len = end - start;

    src_n += start*D + k;
    float sum = 0;

    // This code matches torch.sum() when batch_num = 1.
    // constexpr int K = 4;
    // int batch_size = 8192/max(1<<log2_ceil(D), 32);
    // int batch_num = (len+batch_size-1)/batch_size;
    // for (int k = 0; k < batch_num; k++){
    //     int batch_len = (k==batch_num-1) ? (len - k*batch_size) : batch_size;
    //     float t[K] = {0.0f};
    //     for (int i = 0; i < (batch_len+K-1)/K; i++ ){
    //         #pragma unroll
    //         for (int j = 0; j < K; j++ ){  
    //             int index = i*K + j;
    //             if (index < batch_len )
    //                 t[j] += src_n[index*D];
    //         }
    //     }
    //     float s = 0;
    //     #pragma unroll
    //     for (int j = 0; j < K; j++ ){  
    //         s += t[j];
    //     }
    //     sum += s;
    //     src_n += batch_len*D;
    // }

    // dst_n[n*D+k] = sum * (1.0f/len);
    // return ;

    constexpr int INTER_NUM = 4;
    constexpr int MAX_BUFF_LEN = MAX_SYSTEM_SIZE_POW2/INTER_NUM;
    __shared__ float _data[32][MAX_BUFF_LEN+1];
    float *data = _data[threadIdx.x];
    int log2len = log2_ceil(len);

    switch(log2len){
        case 0:// 1
        sum = src_n[0]; break;

        case 1:// 2
        sum = src_n[0] + src_n[D]; break;

        case 2:// 4
        sum = pairwise_sum<2, INTER_NUM>(src_n, D, len, data); break;

        case 3:// 8
        sum = pairwise_sum<3, INTER_NUM>(src_n, D, len, data); break;

        case 4:// 16
        sum = pairwise_sum<4, INTER_NUM>(src_n, D, len, data); break;

        case 5:// 32
        sum = pairwise_sum<5, INTER_NUM>(src_n, D, len, data); break;

        case 6:// 64
        sum = pairwise_sum<6, INTER_NUM>(src_n, D, len, data); break;

        case 7:// 128
        sum = pairwise_sum<7, INTER_NUM>(src_n, D, len, data); break;

        case 8:// 256
        sum = pairwise_sum<8, INTER_NUM>(src_n, D, len, data); break;

        case 9:// 512
        sum = pairwise_sum<9, INTER_NUM>(src_n, D, len, data); break;

        case 10:// 1024
        sum = pairwise_sum<10, INTER_NUM>(src_n, D, len, data); break;
    }

    dst_n[n*D+k] = sum / (float)len;
}