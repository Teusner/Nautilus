#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


#include "Module.h"
#include "Material.h"
#include "Scene.h"


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

    /// Scene
    dim3 d(10, 10, 10);
    Scene s{d};
    
    /// Material
    Material m(800, 300, 1);
    std::cout << m << std::endl;
    m.update_device();

    /// Scene vector
    thrust::device_vector<Material *> M(d.x * d.y * d.z, m.device_ptr());


    // s.AddMaterial(m);
    // s.PrintMaterials();

    return 0;
}