#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Module.h"

#include <iostream>

int main(void) {
    Emitter a = Emitter(3, 1, 2, [](float x){return 2*x;});
    a(0); a(1); a(2);
    std::cout << "Emitter :\n";
    a.print();

    Reciever b = Reciever(3, 1, 2);
    b.Record(0, 1); b.Record(1, 3); b.Record(2, 2);
    std::cout << "Reciever :\n";
    b.print();

    return 0;
}