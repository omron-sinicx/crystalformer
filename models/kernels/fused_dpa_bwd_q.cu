
extern "C" __global__
void fused_dpa_bwd_q(
    const float* key_ihk,
    const float* gaij_eh,
    const long long int* edge_ij_e, 
    const long long int* e_start_i,
    const long long int N,
    const long long int H,
    const long long int E,
    float* gque_ihk
){
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int K = VPE_DIM;
    const long long int i = tid/H;
    const long long int h = tid%H;
    const long long int e_end = e_start_i[i+1];

    __shared__ float _gq[THREAD_NUM][K_HEAD_DIM+1];
    float *gq = _gq[threadIdx.x];

    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++)
        gq[k] = 0;

    for (long long int e = e_start_i[i]; e < e_end; e++)
    {
        long long int j = edge_ij_e[E+e];
        float g_softmax = gaij_eh[e*H+h];
        const float *key = key_ihk + (j*H+h)*K_HEAD_DIM;
        #pragma unroll
        for (int k = 0; k < K_HEAD_DIM; k++){
            gq[k] += g_softmax*key[k];
        }
        // gb = go.reshape(s,1,H,K) * p.reshape(s,s,H,1)
        // gv = gb.sum(dim=0)
        // gval.append(gv)
        // gbij.append(gb.reshape(s*s,H,K))
        // gsm = (v.reshape(1,s,H,K) + b.reshape(s,s,H,K) - o.reshape(s,1,H,K))*gb
        // ga = gsm.sum(dim=3)
        // gq = (ga.reshape(s,s,H,1)*k.reshape(1,s,H,K)).sum(dim=1)
        // gk = (ga.reshape(s,s,H,1)*q.reshape(s,1,H,K)).sum(dim=0)
        // gaij.append(ga.reshape(s*s,H))
        // gque.append(gq)
        // gkey.append(gk)
    }
    
    gque_ihk += (i*H+h)*K_HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++)
        gque_ihk[k] = gq[k];
}