extern "C" __global__

void reci_enc_fwd(
    const float* a_ik,
    const float* kr_base_e,
    const float* rvecs_n,
    const float* vcell_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int N,
    const long long int H,
    const long long int E,
    float* z_ek,
    float* sumexp_ek){
    
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E*H) return;

    const long long int k = tid%H;
    const long long int e = tid/H;
    const long long int i = edge_ij_e[e];
    const long long int j = edge_ij_e[E+e];
    const long long int n = batch_i[i];
    kr_base_e += e*3;
    const float kr_base_1 = kr_base_e[0];
    const float kr_base_2 = kr_base_e[1];
    const float kr_base_3 = kr_base_e[2];
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
    const float vcell = vcell_n[n];
    const int R = LATTICE_RANGE;

    // Unlike real space, normalization using max_logit is not needed since always max_logit = 0.
    float sum = 0.0;
    float sum_exp = 0.0;

    // Because of symmetry, n1 range can be [0, R] instead of [-R, R]
    // by scaling by a factor of 2 for values at n1!=0.
    // for (float n1 = -R; n1 <= R; n1++){ const float scale=1;
    for (float n1 = 0; n1 <= R; n1++) { float scale = n1==0.0f ? 1.0f : 2.0f;
        for (float n2 = -R; n2 <= R; n2++)
        for (float n3 = -R; n3 <= R; n3++)
        {
            float k1 = r1_x*n1 + r2_x*n2 + r3_x*n3;
            float k2 = r1_y*n1 + r2_y*n2 + r3_y*n3;
            float k3 = r1_z*n1 + r2_z*n2 + r3_z*n3;
            float exp_ak = expf(a*(k1*k1 + k2*k2 + k3*k3));
            sum += exp_ak * cosf(kr_base_1*n1 + kr_base_2*n2 + kr_base_3*n3) * scale;
        }
    }
    // if (n == 0 && sum < 0) {
    //     printf("%f, %f, (%f %f %f)\n", sum, a, kr_base_1, kr_base_2, kr_base_3);
    // }
    
    const float C_4PI = 12.5663706144;
    float log_ci = 1.5f*(logf(-C_4PI*a)) - logf(vcell);
    //log_ci = 0;
    z_ek[tid] = logf(fmaxf(sum,1e-6)) + log_ci;
    sumexp_ek[tid] = sum;
}