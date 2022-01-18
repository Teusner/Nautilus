#include <iostream>
#define N 3

struct MMaterial {
    float rho, cp, Qp, cs, Qs;
    float inv_rho;
    void init() { if (rho != 0) inv_rho = 1 / rho; };
};

__constant__ MMaterial M[N];

__global__ void Mat() {
    for (int i = 0; i < N; ++i) {
        printf("Material : {%f, %f, %f, %f, %f}, [%f]\n", M[i].rho, M[i].cp, M[i].Qp, M[i].cs, M[i].Qs, M[i].inv_rho);
    }
}

int main(int argc, char** argv) {
    MMaterial *temp_m = (MMaterial*) malloc( sizeof(MMaterial) * N );

    temp_m[0].rho = 1000;
    temp_m[0].cp = 1500;
    temp_m[0].Qp = 100;

    temp_m[1].rho = 800;
    temp_m[1].cp = 300;
    temp_m[1].Qp = 10;

    for (int i = 0; i < N; ++i) {
        temp_m[i].init();
    }

    cudaMemcpyToSymbol(M, temp_m, sizeof(MMaterial) * N);
    free(temp_m);

    /// Copying one material on constant memory
    MMaterial *h_m = (MMaterial*) malloc( sizeof(MMaterial) * N );
    cudaMemcpyFromSymbol(h_m, M, sizeof(MMaterial)*N);

    unsigned int index = 2;
    h_m[index].rho = 70;
    h_m[index].cp = 70;
    h_m[index].Qp = 70;
    h_m[index].cs = 70;
    h_m[index].Qs = 70;
    h_m[index].init();

    cudaMemcpyToSymbol(M, h_m, sizeof(MMaterial)*N);

    Mat<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}