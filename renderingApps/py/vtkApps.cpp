#include <fstream>
#include <string>
#include "../../loader.h"
#include "../vtkApps/vtkFuns.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../ext/pybind11_json/pybind11_json.hpp"
#include <iostream>
#include <chrono>

namespace py = pybind11;

// int run_offline(std::string jsonStr, std::string outname, int header_sel){
    
//     {
//         auto start_time = std::chrono::high_resolution_clock::now();
//         run(jsonStr, outname, header_sel);
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//         std::cout << "Total render time (seconds): " << duration.count() << "\n";
//         std::cout << "Total render time (minutes): " << duration.count()/60000 << " minutes\n";
//     }
//     return 0;
// }

int run_offline(std::string jsonStr, std::string output_dir, int header_sel){
    
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Pass the output directory to the run function
        if (!output_dir.empty()) {
            std::cout << "Using output directory for VTK rendering: " << output_dir << std::endl;
        }
        
        run(jsonStr, output_dir, header_sel);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total render time (milliseconds): " << duration.count() << "\n";
        std::cout << "Total render time (minutes): " << duration.count()/60000 << " minutes\n";
    }
    return 0;
}


PYBIND11_MODULE(vistool_py_vtk, m) {
    // Optional docstring
    m.doc() = "the vtk renderer's py library";
        
    m.def("run_offline_app", &run_offline, "run render app");
}


