#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


#include "Module.h"
#include "core/Material.cuh"
#include "Scene.h"
#include "Event.h"
#include "Solver.h"

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

    /// Scene vector
    thrust::host_vector<Material *> M(5, &m);
    thrust::copy(M.begin(), M.end(), std::ostream_iterator<Material *>(std::cout, "\n"));

    // s.AddMaterial(m);
    // s.PrintMaterials();

    /// Events
    Event e1(2);
    Event e2(1);

    std::cout << e1.i() << " " << e2.i() << "\n";
    std::cout << (e2 < e1) << "\n";

    /// Scene
    dim3 Size(10, 10, 10);
    Scene sc(Size);
    sc.PrintMaterials();

    /// Solver
    std::cout << "Adding events to the Solver" << std::endl;
    Solver solve(sc);
    solve.Events.push(e1);
    solve.Events.push(e2);

    while (!solve.Events.empty()){
        Event e = solve.Events.top();
        std::cout << e.i() << std::endl;
        solve.Events.pop();
    }

    return 0;
}