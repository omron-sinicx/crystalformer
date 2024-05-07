
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

    float sum = 0;
    const float* end_ptr = src_n + end*D + k;
    src_n += start*D + k;
    while(src_n != end_ptr){
        sum += *src_n;
        src_n += D;
    }
    dst_n[n*D+k] = sum / (end - start);
}