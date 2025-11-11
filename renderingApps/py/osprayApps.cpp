#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../ext/pybind11_json/pybind11_json.hpp"
#include <iostream>
#include "../osprayApps/RenderFuncs.h"


#include "../osprayApps/GLFWOSPWindow.h"
#include <chrono>

namespace py = pybind11;

void echo(int i) {
    std::cout << i <<std::endl;
}

template<class T>
std::vector<T>makeVectorFromPyArray( py::array_t<T>py_array )
{
    return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}

static std::vector<std::string>
init_app(const std::vector<std::string>& args)
{
    int argc = args.size();
    const char **argv = new const char*[argc];
    
    for (int i = 0; i < argc; i++)
        argv[i] = args[i].c_str();
    
    OSPError res = ospInit(&argc, argv);
    
    if (res != OSP_NO_ERROR)
    {
        delete [] argv;
	std::cout <<"ospInit() failed";
    }
    
    std::vector<std::string> newargs;
    
    for (int i = 0; i < argc; i++)
        newargs.push_back(std::string(argv[i]));
    
    delete [] argv;
    
    return newargs;
}

int run_app(py::array_t<float> &input_array, py::list &input_names, int x, int y, int z, int count, int mode, std::string path_to_bgmap, std::string outputName)
{
    
    std::cout << "only outputname:" << outputName << std::endl;
    std::string env_var = "KF_WIDGET_OUTPUT_PATH="+ outputName;
    putenv(const_cast<char*>(env_var.c_str()));
    std::cout << "Setting environment variable: " << env_var << std::endl;
    // use scoped lifetimes of wrappers to release everything before ospShutdown()
    {
    
	// process py inputs
	py::buffer_info buf_info = input_array.request();
        auto fnames = input_names.cast<std::vector<std::string>>();
	auto ptr = static_cast<float *>(buf_info.ptr);
	run_interactive_in_place(ptr, fnames, x, y, z, count, mode, path_to_bgmap);
    }
    ospShutdown();

    return 0;    
    
}


int run_offline(std::string config_str, std::string output_dir, int header_sel) 
{
    // use scoped lifetimes of wrappers to release everything before ospShutdown()
    {
        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Pass the output directory to the run function
        run(config_str, output_dir, header_sel);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total render time (milliseconds): " << duration.count() << "\n";
        std::cout << "Total render time (minutes): " << duration.count()/60000 << " minutes\n";
    }
    ospShutdown();

    return 0;    
}


PYBIND11_MODULE(vistool_py_osp, m) {
    // Optional docstring
    m.doc() = "the ospray renderer's py library";
        
    m.def("init_app", &init_app, "init render app");
    m.def("run_app", &run_app, "run render app");
    m.def("run_offline_app", &run_offline, "run render app");
}
