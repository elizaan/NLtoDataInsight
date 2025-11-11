#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#include <conio.h>
#include <malloc.h>
#include <windows.h>
#else
#include <alloca.h>
#endif

#include "GLFWOSPWindow.h"
#include "RenderFuncs.h"

using json = nlohmann_loader::json;
using namespace visuser;

int main(int argc, const char **argv)
{


#ifdef _WIN32
    bool waitForKey = false;
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
	// detect standalone console: cursor at (0,0)?
	waitForKey = csbi.dwCursorPosition.X == 0 && csbi.dwCursorPosition.Y == 0;
    }
#endif

    // initialize OSPRay; OSPRay parses (and removes) its commandline parameters,
    // e.g. "--osp:debug"
    OSPError init_error = ospInit(&argc, argv);
    if (init_error != OSP_NO_ERROR)
	return init_error;
	
    // use scoped lifetimes of wrappers to release everything before ospShutdown()
    {
    
	std::vector<std::string> args(argv, argv + argc);
	std::string config_str;
	std::string outname;
    
	// which kf file to render
	// -2 = render all frames, -1 = render all kfs, >0 = render selected kf
	int header_sel = -2; 
	for (int i = 1; i < argc; ++i) {
	    if (args[i] == "-h") {
		std::cout << "./vtk_vistool <config.json> [options] \n"
			  << "  [options]: \n"
			  << "  -f img_output_name"
			  << "  -header-sel # ([default] -2 = render all frames, -1 = render all kfs, >0 = render selected k)"
			  << "\n";
		return 0;
	    } else {
		if (i == 1){
		    config_str = args[i];
		}else{
		    if (args[i] == "-f"){
			config_str = args[++i];
		    }else if (args[i] == "-header-sel")
			header_sel = std::max(0, std::stoi(args[++i]));
		}
	    }
	}
  
	run_interactive_sel(config_str, outname, header_sel);
	
	
    }
    ospShutdown();

#ifdef _WIN32
    if (waitForKey) {
	printf("\n\tpress any key to exit");
	_getch();
    }
#endif

    

    return 0;
}

