
__forceinline__ __device__ int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}


#if 0
template <int num>
__forceinline__ __device__ float fixed_length_pairwise_sum(float *input)
{
    #pragma unroll
    for (int i = num/2; i > 0; i /= 2 ){
        #pragma unroll
        for (int j = 0; j < i; j++){
            input[j] += input[j+i];
        }
    }
    return input[0];
}
template <> __forceinline__ float fixed_length_pairwise_sum<1>(float *input) {
    return input[0];
}
template <> __forceinline__ float fixed_length_pairwise_sum<2>(float *input) {
    return input[0] + input[1];
}
template <> __forceinline__ float fixed_length_pairwise_sum<4>(float *input) {
    return (input[0] + input[2]) + (input[1] + input[3]);
}
template <> __forceinline__ float fixed_length_pairwise_sum<8>(float *input) {
    return ((input[0] + input[4]) + (input[2] + input[6])) \
         + ((input[1] + input[5]) + (input[3] + input[7]));
}

#else
template <int num>
__forceinline__ __device__ float fixed_length_pairwise_sum(float *input)
{
    // #pragma unroll
    // for (int i = 1; i <= num/2; i *= 2 ){
    //     #pragma unroll
    //     for (int j = 0; j < num; j += 2*i){
    //         input[j] += input[j+i];
    //     }
    // }
    // return input[0];
    return fixed_length_pairwise_sum<num/2>(input) + fixed_length_pairwise_sum<num/2>(input+num/2);
}

template <> __forceinline__ float fixed_length_pairwise_sum<1>(float *input) {
    return input[0];
}
template <> __forceinline__ float fixed_length_pairwise_sum<2>(float *input) {
    return input[0] + input[1];
}
template <> __forceinline__ float fixed_length_pairwise_sum<3>(float *input) {
    return (input[0] + input[1]) + (input[2]);
}
template <> __forceinline__ float fixed_length_pairwise_sum<4>(float *input) {
    return (input[0] + input[1]) + (input[2] + input[3]);
}
template <> __forceinline__ float fixed_length_pairwise_sum<5>(float *input) {
    return fixed_length_pairwise_sum<4>(input) + input[4];
}
template <> __forceinline__ float fixed_length_pairwise_sum<7>(float *input) {
    return fixed_length_pairwise_sum<4>(input) + ((input[4] + input[5]) + input[6]);
}
template <> __forceinline__ float fixed_length_pairwise_sum<8>(float *input) {
    return ((input[0] + input[1]) + (input[2] + input[3])) \
         + ((input[4] + input[5]) + (input[6] + input[7]));
}
template <> __forceinline__ float fixed_length_pairwise_sum<9>(float *input) {
    return fixed_length_pairwise_sum<8>(input) + input[8];
}
#endif

template <int log2_elements, int iter_num>
float pairwise_sum(const float *src, int stride, int num, float *buff)
{
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int batch_num = (next_power_of_two>iter_num) ? next_power_of_two/iter_num : 1;
    float mini_batch[iter_num];
    
    #pragma unroll
    for (int i = 0; i < batch_num; i++){
        #if 1
        #pragma unroll
        for (int j = 0; j < iter_num; j++){
            int index = i*iter_num + j;
            if (index < num){
                mini_batch[j] = src[index*stride];
            } else {
                mini_batch[j] = 0;
            }
        }
        buff[i] = fixed_length_pairwise_sum<iter_num>(mini_batch);
        
        #else 
        #pragma unroll
        buff[i] = 0;
        for (int j = 0; j < iter_num; j++){
            int index = i*iter_num + j;
            if (index < num){
                buff[i] += src[index*stride];
            }
        }
        #endif
    }
    return fixed_length_pairwise_sum<batch_num>(buff);
}

