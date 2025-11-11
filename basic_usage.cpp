#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "loader.h"

using json = nlohmann_loader::json;
using namespace visuser;

const std::string USAGE =
    "./mini_vistool <config.json> [options]\n"
    "Options:\n"
    //"  -prefix <name>       Provide a prefix to prepend to the image file names.\n"
    //"  -fif <N>             Restrict the number of frames being rendered in parallel.\n"
    //"  -no-output           Don't save output images (useful for benchmarking).\n"
    //"  -detailed-stats      Record and print statistics about CPU use, thread pinning, etc.\n"
    "  -h                   Print this help.";

int main(int argc, char **argv)
{
    if (argc < 2) std::cout << USAGE<< std::endl;

    std::vector<std::string> args(argv, argv + argc);
    
    Animator animator;
    for (int i = 1; i < argc; ++i) {
        if (args[i] == "-h") {
            std::cout << USAGE << "\n";
            return 0;
        } else {
            animator.init(args[i].c_str());
        }
    }

    for (int i = 0; i < animator.kfs.size(); ++i) {
	animator.kfs[i].print_info();
	for (int j = 0; j < animator.kfs[i].get_data_list_size(); ++j) {
	    if (animator.get_scene_data(i, j)["type"] == "structured"){
		
	    }else if (animator.get_scene_data(i, j)["type"] == "rectilinear"){
	    }
	}
    }

   
    /*AniObjWidget widget(config);
    widget.load_info();
    widget.load_cameras();
    widget.load_tfs();
    std::vector<Camera> cams = widget.cameras;
    
    {
    	widget.print_info();
	for(auto &c : widget.cameras)
	    c.print();
    }*/
    return 0;
}
