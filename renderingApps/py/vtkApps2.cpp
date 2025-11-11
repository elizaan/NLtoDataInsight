#include <fstream>
#include <string>
#include "../../loader.h"
#include "../vtkApps/vtkFuns2.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../ext/pybind11_json/pybind11_json.hpp"
#include <iostream>
#include <chrono>

namespace py = pybind11;

int run_offline_improved(std::string jsonStr, std::string output_dir, int header_sel){
    
    {
        auto start_time = std::chrono::high_resolution_clock::now();
    
        // Pass the output directory to the improved run function
        if (!output_dir.empty()) {
            std::cout << "Using output directory for improved VTK rendering: " << output_dir << std::endl;
        }
        
        std::cout << "\n=== RUNNING IMPROVED OCEANOGRAPHIC VISUALIZATION ===" << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "- Oceanographic colormaps (blue gradient for salinity)" << std::endl;
        std::cout << "- Depth-based opacity" << std::endl;
        std::cout << "- Enhanced streamline visibility" << std::endl;
        std::cout << "- Ocean gradient background" << std::endl;
        std::cout << "- Improved lighting and shading" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        run2(jsonStr, output_dir, header_sel);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\n render time (milliseconds): " << duration.count() << std::endl;
        std::cout << " render time (minutes): " << duration.count()/60000.0 << " minutes" << std::endl;
        std::cout << "=== VISUALIZATION COMPLETE ===" << std::endl;
    }
    return 0;
}


PYBIND11_MODULE(vistool_py_vtk2, m) {
    // Optional docstring
    m.doc() = "Improved VTK renderer's Python library with oceanographic visualization enhancements";

    m.def("run_offline_app_improved", &run_offline_improved, "Run improved oceanographic visualization renderer");
}
