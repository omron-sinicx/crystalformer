#include <math_constants.h>
extern "C" __global__

void real_enc_bwd(
    const float* a_ik,
    const float* rpos_ij_e,
    const float* dist2_min_e,
    const float* tvecs_n,
    const long long int* batch_i,
    const long long int* edge_ij_e,
    const long long int* e_start_i,
    const float* z_ek,
    const float* gz_ek,
    const float* gv_ekd,
    const long long int N,
    const long long int H,
    const long long int E,
    const long long int K_,
    const double dist_max,
    const double wscale,
    const float* rveclens_n,
    const double cutoff_radius,
    float* ga_ik){
    
    const long long int tid = (long long int)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N*H) return;

    const long long int K = VPE_DIM;
    const long long int k = tid%H;
    const long long int i = tid/H;
    const long long int n = batch_i[i];
    tvecs_n += n*9;
    const float t1_x = tvecs_n[0];
    const float t1_y = tvecs_n[1];
    const float t1_z = tvecs_n[2];
    const float t2_x = tvecs_n[3];
    const float t2_y = tvecs_n[4];
    const float t2_z = tvecs_n[5];
    const float t3_x = tvecs_n[6];
    const float t3_y = tvecs_n[7];
    const float t3_z = tvecs_n[8];
    const float a = a_ik[i*H + k];
    const int R = LATTICE_RANGE;
    const long long int e_start = e_start_i[i];
    const long long int e_end = e_start_i[i+1];
    #if VPE_DIM > 0
    __shared__ float shared_gv[THREAD_NUM][VPE_DIM+1];
    #endif

    rveclens_n += n*3;
    const float rvl1 = rveclens_n[0];
    const float rvl2 = rveclens_n[1];
    const float rvl3 = rveclens_n[2];
    
    float cutoff = (float)cutoff_radius;
    int R1 = LATTICE_RANGE, R2 = LATTICE_RANGE, R3 = LATTICE_RANGE;
    if (cutoff != 0.0f)
    {
        if (cutoff < 0) {
            // Better sync the threads in each block?
            // -> disabled due to thread stucking
            // float a_max = a;
            // for (int t = 0; t < THREAD_NUM; t++)
            //     a_max = max(a_max, a_ik[i*H + t]);
            //cutoff = sqrt(-0.5f/a_max)*(-cutoff);
            cutoff = sqrt(-0.5f/a)*(-cutoff);
        }
        R1 = ceil((cutoff + 0.01f)*rvl1/(2.0*CUDART_PI_F));
        R2 = ceil((cutoff + 0.01f)*rvl2/(2.0*CUDART_PI_F));
        R3 = ceil((cutoff + 0.01f)*rvl3/(2.0*CUDART_PI_F));
        float cutoff2 = cutoff*cutoff;

        #if MINIMUM_RANGE > 0
        R1 = max(R1, MINIMUM_RANGE);
        R2 = max(R2, MINIMUM_RANGE);
        R3 = max(R3, MINIMUM_RANGE);
        #endif
    }
    
    float sum = 0;
    float sum_v = 0;
    for (long long int e = e_start; e < e_end; e++)
    {
        const long long int j = edge_ij_e[E+e];
        const float r_ijx = rpos_ij_e[e*3+0];
        const float r_ijy = rpos_ij_e[e*3+1];
        const float r_ijz = rpos_ij_e[e*3+2];
        const long long int ek = e*H+k;
        const float z = z_ek[ek];
        const float gz = gz_ek[ek];

        #if VPE_DIM > 0
        float *sgv = shared_gv[threadIdx.x];
        const float *gv = &gv_ekd[ek*K];
        #pragma unroll
        for (int dim = 0; dim < VPE_DIM; dim++) {
            sgv[dim] = gv[dim];
        }
        #endif

        float px_avr = 0;
        float pbg_avr = 0;
        const float reci_ws_sqrt2 = 1.0f/((float)wscale*sqrt(2.0f));
        const float mu0 = (float)dist_max/VPE_DIM;
        for (float n1 = -R1; n1 <= R1; n1++)
        for (float n2 = -R2; n2 <= R2; n2++)
        for (float n3 = -R3; n3 <= R3; n3++)
        {
            float dx = r_ijx + t1_x*n1 + t2_x*n2 + t3_x*n3;
            float dy = r_ijy + t1_y*n1 + t2_y*n2 + t3_y*n3;
            float dz = r_ijz + t1_z*n1 + t2_z*n2 + t3_z*n3;
            float d2 = dx*dx + dy*dy + dz*dz;
            // float dx = fmaf(t1_x, n1, fmaf(t2_x, n2, fmaf(t3_x, n3, r_ijx)));
            // float dy = fmaf(t1_y, n1, fmaf(t2_y, n2, fmaf(t3_y, n3, r_ijy)));
            // float dz = fmaf(t1_z, n1, fmaf(t2_z, n2, fmaf(t3_z, n3, r_ijz)));
            // float d2 = fmaf(dx,dx, fmaf(dy,dy, dz*dz));
            float p = expf(a*d2 - z);
            float px = d2*p;
            px_avr += px;

            #if VPE_DIM > 0
            float bg = 0;
            float b = -sqrtf(d2)/mu0*reci_ws_sqrt2;
            #pragma unroll
            for (int dim = 0; dim < VPE_DIM; dim++)
            {
                b += reci_ws_sqrt2;
                bg += expf(-b*b)*sgv[dim];
            }
            sum_v += px*bg;
            pbg_avr += p*bg;
            #endif
        }
        /*
        b: (E, 1, R, K)
        x: (E, 1, R, 1)
        y: (N, H, 1, 1)
        z: (E, H, 1, K)
        g: (E, H, 1, K)
        p: (E, H, R, 1)

        (E,H,R,K)   (E,H,R,1)     (E,H,R,K)       (E,H,1,K): (E,H,R,1)*(E,1,R,K)*(E,H,1,K)
        dz/dye    =    p*x    * (    b*g     -    (p*b*g).sum(axis=R))

        (E,H,1,1)
        dz/dyi    = (dz/dye).sum(axis=R,K).sum_for_j()

                     (E,H,R,1)*(E,H,R,1)                (E,H,1,1)        *(E,H,1,1)
        dz/dye    =    (p*x)  *(b*g).sum(axis=K)    -   (p*x).sum(axis=R)*(p*b*g).sum(axis=R,K))
        */

        sum += px_avr*gz;
        sum_v -= px_avr*pbg_avr;

    }
    ga_ik[tid] = sum + sum_v;
}