#include <gtest/gtest.h>
#include "core/Field.cuh"


TEST(PressureField, ComponenSize) {
    dim3 d(10, 15, 20);
    unsigned int size = d.x * d.y * d.z;
    PressureField<thrust::device_vector<float>> P(size);
    EXPECT_EQ(P.x.size(), size);
    EXPECT_EQ(P.y.size(), size);
    EXPECT_EQ(P.z.size(), size);
    EXPECT_EQ(P.xy.size(), size);
    EXPECT_EQ(P.yz.size(), size);
    EXPECT_EQ(P.xz.size(), size);
}

TEST(VelocityField, ComponentSize) {
    dim3 d(10, 15, 20);
    unsigned int size = d.x * d.y * d.z;
    VelocityField<thrust::device_vector<float>> U(size);
    EXPECT_EQ(U.x.size(), size);
    EXPECT_EQ(U.y.size(), size);
    EXPECT_EQ(U.z.size(), size);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}