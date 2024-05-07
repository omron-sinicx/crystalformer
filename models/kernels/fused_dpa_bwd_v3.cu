
extern "C" __global__
void fused_dpa_bwd_v3(
    const float* que_ihk,
    const float* key_ihk,
    const float* val_ihk,
    const float* taij_eh,
    const float* tbij_ehk,
    const long long int* batch_i,
    const long long int* edge_ij_e, 
    const long long int* e_start_i,
    const long long int N,
    const long long int H,
    const long long int E,
    const float* tprob_eh,
    const float* out_ihk,
    const float* gout_ihk,
    float* gque_ihk,
    float* gkey_ihk,
    float* gval_ihk,
    float* tgaij_eh,
    float* tgbij_ehk
){
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int K = VPE_DIM;
    const long long int j = tid/H;
    const long long int h = tid%H;
    const long long int n = batch_i[j];
    const long long int e_start = e_start_i[j];
    const long long int e_end = e_start_i[j+1];

    const float* que_k = &que_ihk[tid*K_HEAD_DIM];
    __shared__ float _v[THREAD_NUM][V_HEAD_DIM+1];
    __shared__ float _gv[THREAD_NUM][V_HEAD_DIM+1];
    __shared__ float _gk[THREAD_NUM][K_HEAD_DIM+1];
    __shared__ float _run_gv[THREAD_NUM][V_HEAD_DIM+1];
    __shared__ float _run_gk[THREAD_NUM][K_HEAD_DIM+1];
    float *v = _v[threadIdx.x];
    float *gv = _gv[threadIdx.x];
    float *gk = _gk[threadIdx.x];
    float *rgv = _run_gv[threadIdx.x];
    float *rgk = _run_gk[threadIdx.x];

    const float *v_src = val_ihk + (j*H+h)*V_HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++){
        gv[k] = 0;
        rgv[k] = 0;
        v[k] = v_src[k];
    }

    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++){
        gk[k] = 0;
        rgk[k] = 0;
    }

    if (tgbij_ehk != NULL && tbij_ehk != NULL ) {
        for (long long int e = e_start; e < e_end; e++)
        {
            long long int i = edge_ij_e[E+e];

            float pij = tprob_eh[e*H+h];
            const float *go = gout_ihk + (i*H+h)*V_HEAD_DIM;
            const float *o = out_ihk + (i*H+h)*V_HEAD_DIM;
            float *gb = tgbij_ehk + (e*H+h)*V_HEAD_DIM;
            const float *b = tbij_ehk + (e*H+h)*V_HEAD_DIM;
            float g_softmax = 0;
            #pragma unroll
            for (int k = 0; k < V_HEAD_DIM; k++){
                float t = go[k]*pij;
                rgv[k] += t;
                gb[k] = t;
                g_softmax += (v[k] + b[k] - o[k]) * t;
            }

            tgaij_eh[e*H+h] = g_softmax;

            const float *q = que_ihk + (i*H+h)*K_HEAD_DIM;
            #pragma unroll
            for (int k = 0; k < K_HEAD_DIM; k++){
                rgk[k] += g_softmax*q[k];
            }
            
            if (((e-e_start) % RUNNING_SUM_LEN) == 0 || e == e_end-1){
                #pragma unroll
                for (int k = 0; k < V_HEAD_DIM; k++){
                    gv[k] += rgv[k];
                    rgv[k] = 0;
                }
                #pragma unroll
                for (int k = 0; k < K_HEAD_DIM; k++){
                    gk[k] += rgk[k];
                    rgk[k] = 0;
                }
            }
        }
    } else {
        for (long long int e = e_start; e < e_end; e++)
        {
            long long int i = edge_ij_e[E+e];

            float pij = tprob_eh[e*H+h];
            const float *go = gout_ihk + (i*H+h)*V_HEAD_DIM;
            const float *o = out_ihk + (i*H+h)*V_HEAD_DIM;
            float g_softmax = 0;
            #pragma unroll
            for (int k = 0; k < V_HEAD_DIM; k++){
                float t = go[k]*pij;
                rgv[k] += t;
                g_softmax += (v[k] - o[k]) * t;
            }

            tgaij_eh[e*H+h] = g_softmax;

            const float *q = que_ihk + (i*H+h)*K_HEAD_DIM;
            #pragma unroll
            for (int k = 0; k < K_HEAD_DIM; k++){
                rgk[k] += g_softmax*q[k];
            }

            if (((e-e_start) % RUNNING_SUM_LEN) == 0 || e == e_end-1){
                #pragma unroll
                for (int k = 0; k < V_HEAD_DIM; k++){
                    gv[k] += rgv[k];
                    rgv[k] = 0;
                }
                #pragma unroll
                for (int k = 0; k < K_HEAD_DIM; k++){
                    gk[k] += rgk[k];
                    rgk[k] = 0;
                }
            }
        }
    }
    
    gval_ihk += (j*H+h)*V_HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < V_HEAD_DIM; k++)
        gval_ihk[k] = gv[k];

    gkey_ihk += (j*H+h)*K_HEAD_DIM;
    #pragma unroll
    for (int k = 0; k < K_HEAD_DIM; k++)
        gkey_ihk[k] = gk[k];
}