extern "C" __global__
void irregular_transpose(
    const float* src_matrices,
    const long long int* upper_e_t,
    const long long int* mat_sec_t,
    const long long int* dims_t,
    const long long int hE,
    const long long int D_,
    float* dst_matrices
){
    const unsigned long long int tid = (unsigned long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= hE*D_) return;
    const unsigned long long int t = tid/D_;
    const unsigned int k = tid%D_;
    const unsigned int D = D_;
    const unsigned long long int e = upper_e_t[t];
    const unsigned long long int e0 = mat_sec_t[t];
    const unsigned int dim = dims_t[t];
    const unsigned int i = (e - e0) / dim;
    const unsigned int j = (e - e0) % dim;

    const unsigned long long int mat_index = e0*D_ + k;
    src_matrices += mat_index;
    dst_matrices += mat_index;
    const unsigned int ij = (i*dim + j)*D;
    const unsigned int ji = (j*dim + i)*D;
    float tmp = src_matrices[ji];
    dst_matrices[ji] = src_matrices[ij];
    dst_matrices[ij] = tmp;
}

