#include "Module.h"

#include <gtest/gtest.h>
#include <functional>


TEST(Reciever, Instanciation) {
    unsigned int i = 1;
    unsigned int j = 5;
    unsigned int k = 12;

    // Reciever Object
    Reciever r(i, j, k);

    // Tests
    EXPECT_EQ(r.X(), i);
    EXPECT_EQ(r.Y(), j);
    EXPECT_EQ(r.Z(), k);
}

TEST(Reciever, DeviceAllocation) {
    unsigned int i = 1;
    unsigned int j = 5;
    unsigned int k = 12;

    // Reciever object
    Reciever r(i, j, k);

    // Copy Reciever on device
    Reciever *d_r = nullptr;
    cudaMalloc((Reciever **)&d_r, sizeof(Reciever));
    cudaMemcpy(d_r, &r, sizeof(Reciever), cudaMemcpyHostToDevice);
    
    // Copy Reciever on host
    Reciever *h_r = (Reciever *)malloc(sizeof(Reciever));
    cudaMemcpy(h_r, d_r, sizeof(Reciever), cudaMemcpyDeviceToHost);

    // Tests
    EXPECT_EQ(h_r->X(), i);
    EXPECT_EQ(h_r->Y(), j);
    EXPECT_EQ(h_r->Z(), k);

    // Free memory
    delete h_r;
    cudaFree(d_r);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}