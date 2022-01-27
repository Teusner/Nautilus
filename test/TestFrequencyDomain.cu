#include <gtest/gtest.h>
#include "core/FrequencyDomain.cuh"


TEST(FrequencyDomain, Instanciation) {
    float omega_min = 10;
    float omega_max = 100;
    FrequencyDomain FD(omega_min, omega_max);

    EXPECT_EQ(FD.OmegaMin(), omega_min);
    EXPECT_EQ(FD.OmegaMax(), omega_max);
}

TEST(FrequencyDomain, ReverseInstanciation) {
    float omega_min = 10;
    float omega_max = 100;
    FrequencyDomain FD(omega_max, omega_min);

    EXPECT_EQ(FD.OmegaMin(), omega_min);
    EXPECT_EQ(FD.OmegaMax(), omega_max);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}