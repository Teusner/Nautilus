#include <gtest/gtest.h>
#include <vector>
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

TEST(FrequencyDomain, ValuesTest) {
    float omega_min = 2*M_PI*2;
    float omega_max = 2*M_PI*25;
    std::vector<float> tau_sigma = {0.099472, 0.0072343};
    FrequencyDomain FD(omega_max, omega_min, tau_sigma);

    EXPECT_NEAR(FD.tau(20), float(0.10110), float(0.05));
    EXPECT_NEAR(FD.tau(100), float(0.019156), float(0.005));
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}