
extern "C" __global__
void irregular_transpose_old(
    const float* src_matrices,
    const long long int* start_i,
    const short* dims_i,
    const long long int B,
    const long long int D_,
    float* dst_matrices
){
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= B*D_) return;
    const unsigned int D = D_;
    const unsigned int i = tid/D;
    const unsigned int k = tid%D;
    const unsigned int dim = dims_i[i];

    src_matrices += start_i[i]*D+k;
    dst_matrices += start_i[i]*D+k;
    
    unsigned int ij_step = D;
    unsigned int ji_step = D*dim;
    for (unsigned int row = 0; row < dim; row++){
        unsigned int ij = (dim+1)*row*D;
        unsigned int ji = ij;
        for (unsigned int col = row; col < dim; col++){
            // unsigned int ij = (row*dim+col)*D;
            // unsigned int ji = (col*dim+row)*D;
            
            float tmp = src_matrices[ji];
            dst_matrices[ji] = src_matrices[ij];
            dst_matrices[ij] = tmp;
            ij += ij_step;
            ji += ji_step;
        }
    }
}

