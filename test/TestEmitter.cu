#include "Module.h"

#include <gtest/gtest.h>
#include <functional>


TEST(Emitter, Instanciation) {
    unsigned int i = 1;
    unsigned int j = 5;
    unsigned int k = 12;
    std::function<float(float)> f = [](float t) { return t; };

    // Emitter Object
    Emitter e(i, j, k, f);

    // Tests
    EXPECT_EQ(e.X(), i);
    EXPECT_EQ(e.Y(), j);
    EXPECT_EQ(e.Z(), k);
    EXPECT_EQ(e(0), 0);
}

TEST(Emitter, DeviceAllocation) {
    unsigned int i = 1;
    unsigned int j = 5;
    unsigned int k = 12;
    std::function<float(float)> f = [](float t) { return t; };

    // Emitter object
    Emitter e(i, j, k, f);

    // Copy Emitter on device
    Emitter *d_e = nullptr;
    cudaMalloc((Emitter **)&d_e, sizeof(Emitter));
    cudaMemcpy(d_e, &e, sizeof(Emitter), cudaMemcpyHostToDevice);
    
    // Copy Emitter on host
    Emitter *h_e = (Emitter *)malloc(sizeof(Emitter));
    cudaMemcpy(h_e, d_e, sizeof(Emitter), cudaMemcpyDeviceToHost);

    // Tests
    EXPECT_EQ(h_e->X(), i);
    EXPECT_EQ(h_e->Y(), j);
    EXPECT_EQ(h_e->Z(), k);
    EXPECT_EQ(h_e->operator()(0), 0);

    // Free memory
    delete h_e;
    cudaFree(d_e);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}