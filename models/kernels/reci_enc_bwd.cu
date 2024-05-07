extern "C" __global__

void reci_enc_bwd(
    const float* a_ik,
    const float* kr_base_e,
    const float* rvecs_n,
    const float* vcell_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int* e_start_i,
    const float* z_ek,
    const float* gz_ek,
    const float* sumexp_ek,
    const long long int N,
    const long long int H,
    const long long int E,
    float* ga_ik){
    
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int k = tid%H;
    const long long int i = tid/H;
    const long long int n = batch_i[i];
    rvecs_n += n*9;
    const float r1_x = rvecs_n[0];
    const float r1_y = rvecs_n[1];
    const float r1_z = rvecs_n[2];
    const float r2_x = rvecs_n[3];
    const float r2_y = rvecs_n[4];
    const float r2_z = rvecs_n[5];
    const float r3_x = rvecs_n[6];
    const float r3_y = rvecs_n[7];
    const float r3_z = rvecs_n[8];
    const float a = a_ik[i*H + k];
    const int R = LATTICE_RANGE;
    const long long int e_start = e_start_i[i];
    const long long int e_end = e_start_i[i+1];

    float sum = 0;
    for (long long int e = e_start; e < e_end; e++)
    {
        const long long int j = edge_ij_e[E+e];
        const float* kr_base = &kr_base_e[e*3];
        const float kr_base_1 = kr_base[0];
        const float kr_base_2 = kr_base[1];
        const float kr_base_3 = kr_base[2];
        const long long int ek = e*H+k;
        const float z = z_ek[ek];
        const float sum_exp = sumexp_ek[ek];
        const float gz = gz_ek[ek];
        
        float s = 0;
        // Because of symmetry, n1 range can be [0, R] instead of [-R, R]
        // by scaling values at n1 != 0 by a factor of 2.
        //for (float n1 = -R; n1 <= R; n1++) { const float scale = 1.0f;
        for (float n1 = 0; n1 <= R; n1++) { float scale = n1==0.0f ? 1.0f : 2.0f;
            for (float n2 = -R; n2 <= R; n2++)
            for (float n3 = -R; n3 <= R; n3++)
            {
                float k1 = r1_x*n1 + r2_x*n2 + r3_x*n3;
                float k2 = r1_y*n1 + r2_y*n2 + r3_y*n3;
                float k3 = r1_z*n1 + r2_z*n2 + r3_z*n3;
                float kk = (k1*k1 + k2*k2 + k3*k3);
                s += kk * expf(a*kk) * cosf(kr_base_1*n1 + kr_base_2*n2 + kr_base_3*n3) * scale;
            }
        }
        //sum += s*gz/fmaxf(sum_exp, 1e-6);
        sum += s*gz*(sum_exp < 1e-6? 0.0f : 1.0f/sum_exp);
    }
    ga_ik[tid] = sum;
}