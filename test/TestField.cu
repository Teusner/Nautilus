#include <gtest/gtest.h>
#include "core/Field.cuh"


TEST(PressureField, Dimension) {
    dim3 d(10, 15, 20);
    PressureField P(d);
    EXPECT_EQ(P.Dim().x, d.x);
    EXPECT_EQ(P.Dim().y, d.y);
    EXPECT_EQ(P.Dim().z, d.z);
}

TEST(PressureField, ComponentDimension) {
    dim3 d(10, 15, 20);
    PressureField P(d);
    unsigned int size = d.x * d.y * d.z;
    EXPECT_EQ(P.x.size(), size);
    EXPECT_EQ(P.y.size(), size);
    EXPECT_EQ(P.z.size(), size);
    EXPECT_EQ(P.xy.size(), size);
    EXPECT_EQ(P.yz.size(), size);
    EXPECT_EQ(P.xz.size(), size);
}

TEST(VelocityField, Dimension) {
    dim3 d(10, 15, 20);
    VelocityField U(d);
    EXPECT_EQ(U.Dim().x, d.x);
    EXPECT_EQ(U.Dim().y, d.y);
    EXPECT_EQ(U.Dim().z, d.z);
}

TEST(VelocityField, ComponentDimension) {
    dim3 d(10, 15, 20);
    VelocityField U(d);
    unsigned int size = d.x * d.y * d.z;
    EXPECT_EQ(U.x.size(), size);
    EXPECT_EQ(U.y.size(), size);
    EXPECT_EQ(U.z.size(), size);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}