#include <gtest/gtest.h>
#include "Event.h"


TEST(EventTest, Lower) {
    Event e0(0);
    Event e1(1);
    EXPECT_TRUE(e0 < e1);
}

TEST(EventTest, Greater) {
    Event e0(0);
    Event e1(1);
    EXPECT_TRUE(e1 > e0);
}

TEST(EventTest, LowerEqual) {
    Event e0(0);
    Event e1(0);
    EXPECT_TRUE(e0 <= e1);
}

TEST(EventTest, GreaterEqual) {
    Event e0(0);
    Event e1(0);
    EXPECT_TRUE(e1 >= e0);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc,argv);
        return RUN_ALL_TESTS();
}