#include "Event.h"

bool operator<(const Event &lhs, const Event &rhs) {
    return lhs.i() < rhs.i();
}

bool operator>(const Event &lhs, const Event &rhs) {
    return lhs.i() > rhs.i();
}

bool operator<=(const Event &lhs, const Event &rhs) {
    return lhs.i() <= rhs.i();
}

bool operator>=(const Event &lhs, const Event &rhs) {
    return lhs.i() >= rhs.i();
}