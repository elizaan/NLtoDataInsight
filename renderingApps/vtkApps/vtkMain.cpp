#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFixedPointVolumeRayCastMapper.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkImageData.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>


#include <vtkSmartPointer.h>
#include <vtkImageWriter.h>
#include <vtkPNGWriter.h>
#include <vtkPostScriptWriter.h>
#include <vtkWindowToImageFilter.h>


#include <fstream>
#include "../../loader.h"
#include "vtkFuns.h"

using namespace visuser;
using namespace nlohmann_loader;


int main(int argc, char* argv[])
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
	      }else if (args[i] == "-header-sel"){
		  header_sel = std::stoi(args[++i]);
	      }else if (args[i] == "-header")
		  header_sel = -1;
	  }
      }
  }
  
  run(config_str, outname, header_sel);

  //iren->Start();

  return EXIT_SUCCESS;
}


