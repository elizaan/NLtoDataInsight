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

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkStreamTracer.h>
#include <vtkStructuredGridOutlineFilter.h>
#include <vtkPointData.h>
#include <vtkStreamTracer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>
#include <vtkPolyDataWriter.h>
#include <array>
#include <vtkPointSource.h>
#include <vtkPlaneSource.h>
#include <vtkLookupTable.h>

#include <fstream>
#include "../../loader.h"
#include "vtkFuns.h"

using namespace visuser;
using namespace nlohmann_loader;

std::string getOutName(std::string str, uint32_t idx)
{
    std::string base_filename = str.substr(str.find_last_of("/\\") + 1);
    std::string outname = base_filename.substr(0, base_filename.find_last_of("."));
    return "img_"+outname+"_kf"+std::to_string(idx)+".png";
}


void loadTransferFunction(json &j,
			  vtkVolumeProperty *volumeProperty)
{
    
    // Create transfer mapping scalar value to opacity
    vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
    // Create transfer mapping scalar value to color
    vtkNew<vtkColorTransferFunction> colorTransferFunction;
    std::vector<float> colors = j["transferFunc"]["colors"].get<std::vector<float>>();
    std::vector<float> opacities = j["transferFunc"]["opacities"].get<std::vector<float>>();
    glm::vec2 tfRange = get_vec2f(j["transferFunc"]["range"]);
    

    for (uint32_t i=0; i<colors.size()/3; i++){
	float current_val = tfRange[0] + float(i)/(colors.size()/3) * (tfRange[1] - tfRange[0]);
        opacityTransferFunction->AddPoint(current_val, opacities[i]);
	colorTransferFunction->AddRGBPoint(current_val, colors[3*i], colors[i*3+1], colors[i*3+2]);
    }
    std::cout << "load tf sz= "<< colors.size()/3<<" \n";

    volumeProperty->SetColor(colorTransferFunction);
    volumeProperty->SetScalarOpacity(opacityTransferFunction);
    //volumeProperty->ShadeOn();
    volumeProperty->SetInterpolationTypeToLinear();
}


void loadCamera(AnimatorKF &keyframe, vtkRenderer *ren1)
{
    
    // load camera
    double world_scale_xy = 1;
    for (size_t i=0; i<2; i++)
	world_scale_xy = std::min(world_scale_xy, double(keyframe.get_bbox()[i]/keyframe.get_max_dims()[i]));
    double cam_scale =  1/world_scale_xy;
    
    visuser::Camera c;
    keyframe.get_current_cam(c);
    auto camera = ren1->GetActiveCamera();
    double eyePos[3] = {c.pos[0]*cam_scale, c.pos[1]*cam_scale, c.pos[2]*cam_scale };
    double focalPoint[3] = {eyePos[0] + c.dir[0]*cam_scale*60,
	eyePos[1] + c.dir[1]*cam_scale*60,
	eyePos[2] + c.dir[2]*cam_scale*60};

    std::cout <<"load pos:"<< c.pos[0] <<" "<<c.pos[1] <<" "<< c.pos[2]<<"\n";
    std::cout <<"load dir:"<< c.dir[0] <<" "<<c.dir[1] <<" "<< c.dir[2]<<"\n";
    std::cout <<"load up:"<< c.up[0] <<" "<<c.up[1] <<" "<< c.up[2]<<"\n";
    std::cout << "load scale:"<< cam_scale << "\n";
    camera->SetPosition(eyePos[0], eyePos[1], eyePos[2]);
    camera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
    camera->SetViewUp(c.up[0], c.up[1], c.up[2]);
    ren1->ResetCameraClippingRange();
    //ren1->ResetCamera();
}


void loadKF(Animator &animator,
	    uint32_t idx,
	    vtkImageData *img,
	    vtkRenderer *ren1)
{
    bool loadVol = false;
    bool loadDerived = false;
    ren1->RemoveAllViewProps();
    for (int j = 0; j < animator.kfs[idx].get_data_list_size(); ++j) {
	int dim[3];
	std::string file_name;
	double spc[3] = {1.0, 1.0, 1.0};
	
	// The property describes how the data will look
	vtkNew<vtkVolumeProperty> volumeProperty;
	json data = animator.get_scene_data(idx, j);
	json info = animator.get_scene_info(idx, j);
	if (data["type"] == "streamline"){
	    //
	    // Read data
	    //
	    file_name = data["src"]["name"];
	    vtkNew<vtkStructuredGridReader> reader;
	    reader->SetFileName(file_name.c_str());
	    reader->Update(); // force a read to occur
	    reader->GetOutput()->GetLength();
	    vtkNew<vtkNamedColors> colors;
	    double range[2];

	    reader->GetOutput()->GetPointData()->GetScalars()->GetRange(range);
	    std::cout << range[0] << " "<<range[1] <<std::endl;
            	// Add this after reading the structured grid data in the streamline section
		// Place it right after this line:
		// reader->GetOutput()->GetPointData()->GetScalars()->GetRange(range);
		// std::cout << range[0] << " "<<range[1] <<std::endl;

		// Print active scalar field name
		vtkDataArray* activeScalars = reader->GetOutput()->GetPointData()->GetScalars();
		std::cout << "Active scalar field name: " << 
			(activeScalars ? activeScalars->GetName() : "unnamed") << std::endl;

		// Check for velocity vectors
		vtkDataArray* velocityVectors = reader->GetOutput()->GetPointData()->GetVectors();
		if (velocityVectors) {
			std::cout << "Velocity field name: " << velocityVectors->GetName() << std::endl;
			
			// Get the velocity magnitude range
			double velocityRange[2] = {DBL_MAX, -DBL_MAX};  // Initialize min to high, max to low
			int pointCount = reader->GetOutput()->GetNumberOfPoints();
			
			for (int i = 0; i < pointCount; i++) {
				double velocity[3];
				velocityVectors->GetTuple(i, velocity);
				double magnitude = sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
				
				velocityRange[0] = std::min(velocityRange[0], magnitude);
				velocityRange[1] = std::max(velocityRange[1], magnitude);
			}
			
			std::cout << "Velocity magnitude range: " << velocityRange[0] << " to " << velocityRange[1] << std::endl;
			
			// Print some sample velocity values
			std::cout << "Sample velocity vectors and magnitudes:" << std::endl;
			for (int i = 0; i < std::min(10, pointCount); i++) {
				int pointId = (pointCount / std::max(10, 1)) * i;
				double velocity[3];
				velocityVectors->GetTuple(pointId, velocity);
				double magnitude = sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
				std::cout << "Point " << pointId << ": (" 
						<< velocity[0] << ", " << velocity[1] << ", " << velocity[2] 
						<< ") - Magnitude: " << magnitude << std::endl;
		}
	}

		// Also print more details about scalar field
		if (activeScalars) {
			// Print some scalar field statistics
			int scalarPointCount = reader->GetOutput()->GetNumberOfPoints();
			std::cout << "Scalar field statistics (from " << scalarPointCount << " points):" << std::endl;
			
			// Sample some scalar values
			std::cout << "Sample scalar values:" << std::endl;
			for (int i = 0; i < std::min(10, scalarPointCount); i++) {
				int pointId = (scalarPointCount / std::max(10, 1)) * i;
				double value = activeScalars->GetTuple1(pointId);
				std::cout << "Point " << pointId << ": " << value << std::endl;
			}
			
			// Print more detailed range information
			double percentiles[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
			std::cout << "Suggested scalar ranges to consider:" << std::endl;
			std::cout << "  Full range: [" << range[0] << ", " << range[1] << "]" << std::endl;
			std::cout << "  Middle 50%: [" << range[0] + (range[1]-range[0])*0.25 << ", " 
					<< range[0] + (range[1]-range[0])*0.75 << "]" << std::endl;
		}
	    //
	    // Outline around data.
	    //
	    vtkNew<vtkStructuredGridOutlineFilter> outlineF;
	    outlineF->SetInputConnection(reader->GetOutputPort());
	    vtkNew<vtkPolyDataMapper> outlineMapper;
	    outlineMapper->SetInputConnection(outlineF->GetOutputPort());
	    vtkNew<vtkActor> outline;
	    outline->SetMapper(outlineMapper);
	    outline->GetProperty()->SetColor(colors->GetColor3d("LampBlack").GetData());

	    // ONLY ONE EXAMPLE
	    // HARD-CODED NOW!
	    
	    // Create transfer mapping scalar value to opacity
	    vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
	    opacityTransferFunction->AddPoint(range[0], 0.0);
		// for mediterranean sea
	    // opacityTransferFunction->AddPoint(35, 0.0); //35 before
	    // opacityTransferFunction->AddPoint(38, 0.1); //38 before
		opacityTransferFunction->AddPoint(35, 0.0);  // Fixed syntax for red sea
	    opacityTransferFunction->AddPoint(40, 0.1);  // Fixed syntax	
	    opacityTransferFunction->AddPoint(range[1], 0.1);

	    // Create transfer mapping scalar value to color
	    vtkNew<vtkColorTransferFunction> colorTransferFunction;
	    colorTransferFunction->AddRGBPoint(range[0], 0.0, 0.0, 0.0);
		// for mediterranean sea
	    // colorTransferFunction->AddRGBPoint(35, 0.0, 0.0, 0.0);
	    // colorTransferFunction->AddRGBPoint(38, 1.0, 1.0, 1.0);
		colorTransferFunction->AddRGBPoint(35, 0.0, 0.0, 0.0);            // Dark blue for low salinity
	           // Medium blue for average salinity
		colorTransferFunction->AddRGBPoint(40, 1.0, 1.0, 1.0); 		  // White for high salinity	

	    colorTransferFunction->AddRGBPoint(range[1], 1.0, 1.0, 1.0);

	    // The property describes how the data will look
	    vtkNew<vtkVolumeProperty> volumeProperty;
	    volumeProperty->SetColor(colorTransferFunction);
	    volumeProperty->SetScalarOpacity(opacityTransferFunction);
	    //volumeProperty->ShadeOn();
	    volumeProperty->SetInterpolationTypeToLinear();
    
	    vtkNew<vtkImageData> imageData;
	    imageData->GetPointData()->SetScalars(reader->GetOutput()->GetPointData()->GetArray(0));
		int dims[3];
		reader->GetOutput()->GetDimensions(dims);
		imageData->SetDimensions(dims);
	    // imageData->SetDimensions(reader->GetOutput()->GetDimensions()); 
	    imageData->SetSpacing(1.0, 1.0, 1.0); 
	    imageData->SetOrigin(0.0, 0.0, 0.0);


	    // The mapper / ray cast function know how to render the data
	    vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
	    volumeMapper->SetInputData(imageData);

	    // The volume holds the mapper and the property and
	    // can be used to position/orient the volume
	    vtkNew<vtkVolume> volume;
	    volume->SetMapper(volumeMapper);
	    volume->SetProperty(volumeProperty);

	    //
	    // regular streamlines
	    //
	    // Create a plane
	    auto ext = outlineMapper->GetBounds();
	    double bblow[3] = {ext[0], ext[2], ext[4]};
	    double bblen[3] = {(ext[1] - ext[0]), (ext[3] - ext[2]), (ext[5] - ext[4])};
    
	    vtkNew<vtkPlaneSource> source;
	    source->SetOrigin(bblow[0]+ bblen[0]/4, bblow[1], bblow[2]);
	    source->SetPoint1(bblow[0]+ bblen[0]/4, bblow[1] + bblen[1], bblow[2]);
	    source->SetPoint2(bblow[0]+ bblen[0]/4, bblow[1], bblow[2]+ bblen[2]);
	    //source->SetOrigin(0, bblow[1]+107/194.0*bblen[1], 0);
	    //source->SetPoint1(bblow[0]+ bblen[0], bblow[1]+107/194.0*bblen[1], 0);
	    //source->SetPoint2(0, bblow[1]+145/194.0*bblen[1], bblow[2]+bblen[2]);
	    source->SetXResolution(20);
	    source->SetYResolution(20);
	    source->Update();
    
	    /*vtkNew<vtkPointSource> source;
	      source->SetCenter(7.172012790671904, 82.78572990198285, 28.830825601071112);
	      source->SetRadius(15.0);
	      source->SetNumberOfPoints(100);*/
    
	    vtkPolyData* poly = source->GetOutput();
	    vtkNew<vtkPolyDataMapper> mapper;
	    mapper->SetInputData(poly);

	    vtkNew<vtkActor> actor;
	    actor->SetMapper(mapper);
	    actor->GetProperty()->SetColor(colors->GetColor3d("Banana").GetData());

	    vtkNew<vtkStreamTracer> streamers;
	    //streamers->DebugOn();
	    streamers->SetInputConnection(reader->GetOutputPort());
	    streamers->SetSourceConnection(source->GetOutputPort());
	    streamers->SetMaximumPropagation(150);
	    streamers->SetInitialIntegrationStep(.5);
	    streamers->SetMinimumIntegrationStep(.1);
	    streamers->SetIntegratorType(2);
	    streamers->SetIntegrationDirectionToBoth();
	    //streamers->SetIntegrationDirectionToBackward();
	    //streamers->SetIntegrationDirectionToForward();
	    streamers->Update();
        // for mediterranean sea
	    // double scalar_range[2] = {36, 38};
		double scalar_range[2] = {35, 39}; //for red sea

	    vtkNew<vtkLookupTable> rainbowBlueRedLut;
	    rainbowBlueRedLut->SetNumberOfColors(256);
	    rainbowBlueRedLut->SetHueRange(0.667, 0.0);
	    rainbowBlueRedLut->Build();
	    vtkNew<vtkPolyDataMapper> streamLineMapper;
	    streamLineMapper->SetInputConnection(streamers->GetOutputPort());
	    streamLineMapper->SetLookupTable(rainbowBlueRedLut);
	    streamLineMapper->SetScalarRange(scalar_range);

	    vtkNew<vtkActor> streamLineActor;
	    streamLineActor->SetMapper(streamLineMapper);
	    streamLineActor->VisibilityOn();

	    // Add the actor to the scene
	    ren1->AddVolume(volume);
	    //ren1->AddActor(actor);
	    ren1->AddActor(outline);
	    ren1->AddActor(streamLineActor);

	    break;
	}
	
	if (data["type"] == "structured"){
	    // load only one volume for now
	    if (!loadVol) loadVol = true;
	    else break;
	    // load data info
	    glm::vec3 data_dims = get_vec3f(data["src"]["dims"]);
	    dim[0] = data_dims[0];
	    dim[1] = data_dims[1];
	    dim[2] = data_dims[2];
	    file_name = data["src"]["name"];
	    // load scene info  
	    loadTransferFunction(info, volumeProperty);
	}
	
	//vtkNew<vtkImageData> img;
	img->SetDimensions(dim);
	img->AllocateScalars(VTK_INT, 1);
  
	std::cout <<"Loading volume "<<file_name <<" with dim "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<"\n";
	std::fstream file;
	file.open(file_name, std::fstream::in | std::fstream::binary);
	for (int z = 0; z < dim[2]; ++z)
	    for (int y = 0; y < dim[1]; ++y)
		for (int x = 0; x < dim[0]; ++x)
		    {
			float buff;
			file.read((char*)(&buff), sizeof(buff));
			img->SetScalarComponentFromDouble(x, y, z, 0, buff);
		    }

	file.close();
	std::cout<< "End load\n";

	img->SetSpacing(spc);

	// The mapper / ray cast function know how to render the data
	vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
	//volumeMapper->SetInputConnection(reader->GetOutputPort());
	volumeMapper->SetInputData(img);

	// The volume holds the mapper and the property and
	// can be used to position/orient the volume
	vtkNew<vtkVolume> volume;
	volume->SetMapper(volumeMapper);
	volume->SetProperty(volumeProperty);
	volume->SetPosition(0, 0, 0);

	// Create the standard renderer, render window
	// and interactor
	ren1->AddVolume(volume);
    }
    vtkNew<vtkNamedColors> colors;
    ren1->SetBackground(colors->GetColor3d("Wheat").GetData());
    
    loadCamera(animator.kfs[idx], ren1);

}


void writeImage(std::string const& fileName, vtkRenderWindow* renWin, bool rgba)
{
    if (!fileName.empty())
	{
	    std::string fn = fileName;
	    auto writer = vtkSmartPointer<vtkPNGWriter>::New();
		
	    vtkNew<vtkWindowToImageFilter> window_to_image_filter;
	    window_to_image_filter->SetInput(renWin);
	    window_to_image_filter->SetScale(1); // image quality
	    if (rgba) window_to_image_filter->SetInputBufferTypeToRGBA();	
	    else window_to_image_filter->SetInputBufferTypeToRGB();
		
	    // Read from the front buffer.
	    window_to_image_filter->ReadFrontBufferOff();
	    window_to_image_filter->Update();

	    writer->SetFileName(fn.c_str());
	    writer->SetInputConnection(window_to_image_filter->GetOutputPort());
	    writer->Write();
	}
    else std::cerr << "No filename provided." << std::endl;

    return;
}

// void run(std::string jsonStr, std::string fname, int header_sel){
 
//   // load json
//   std::cout << "\n\nStart json loading ... \n";
      
//   Animator animator;
//   animator.init(jsonStr.c_str());

//   std::cout << "\nEnd json loading ... \n\n";

//   vtkNew<vtkRenderer> ren1;

//   vtkNew<vtkRenderWindow> renWin;
//   renWin->AddRenderer(ren1);

//   vtkNew<vtkRenderWindowInteractor> iren;
//   iren->SetRenderWindow(renWin);

//   // Create the reader for the data
//   vtkNew<vtkImageData> img;
//   loadKF(animator, 0, img, ren1);

//   renWin->SetSize(1200, 900);
//   renWin->SetWindowName("SimpleRayCast");
//   renWin->Render();

//   if (header_sel >= 0){ // render selected keyframe
//       // save file
//       writeImage(getOutName( "", header_sel), renWin, false);
//       // Render and interact
//       //vtkNew<vtkRenderWindowInteractor> iRen;
//       //iRen->SetRenderWindow(renWin);
//       //renWin->Render();
//       //renWin->SetSize(800, 600);
//       //renWin->SetWindowName("StructuredGrid");
//       //iRen->Start();
//   }else{
//       // reload widget for each key frame
//       for (int kf_idx=0; kf_idx<animator.kfs.size(); kf_idx++){
// 	  loadKF(animator, kf_idx, img, ren1);
// 	  if (header_sel == -1){// render key frames
// 	      renWin->Render();
// 	      // save file
// 	      writeImage(getOutName(fname, kf_idx), renWin, false);
// 	  }else if (header_sel == -2){//renderAllFrames
// 	      glm::vec2 frameRange = animator.kfs[kf_idx].get_fRange();
// 	      for (int f = frameRange[0]; f <= frameRange[1]; f++){
// 		  renWin->Render();
// 		  std::string outname = fname+"img_"+std::to_string(kf_idx)+"_f"+std::to_string(f)+".png";
// 		  writeImage(outname, renWin, false);
// 		  if (f < frameRange[1]){
// 		      // advance frame 
// 		      animator.kfs[kf_idx].advanceFrame();
// 		      // load camera
// 		      loadCamera(animator.kfs[kf_idx], ren1);
// 		  }
// 	      }

// 	  }

//       }
//   }
// }


void run(std::string jsonStr, std::string output_dir, int header_sel){
 
	// Create output directory if it doesn't exist
	if (!output_dir.empty()) {
		if (!std::filesystem::exists(output_dir)) {
			std::cout << "Creating output directory: " << output_dir << std::endl;
			std::filesystem::create_directories(output_dir);
		}
	}
  
	// load json
	std::cout << "\n\nStart json loading ... \n";
		
	Animator animator;
	animator.init(jsonStr.c_str());
  
	std::cout << "\nEnd json loading ... \n\n";
  
	vtkNew<vtkRenderer> ren1;
  
	vtkNew<vtkRenderWindow> renWin;
	renWin->AddRenderer(ren1);
  
	vtkNew<vtkRenderWindowInteractor> iren;
	iren->SetRenderWindow(renWin);
  
	// Create the reader for the data
	vtkNew<vtkImageData> img;
	loadKF(animator, 0, img, ren1);
  
	// renWin->SetSize(1200, 900);
	// renWin->SetSize(2048, 1080);
	renWin->SetSize(800, 600);
	renWin->SetWindowName("SimpleRayCast");
	renWin->Render();
  
	if (header_sel >= 0){ // render selected keyframe
		// save file with output directory
		std::string outname;
		if (!output_dir.empty()) {
			outname = output_dir + "/" + getOutName("", header_sel);
		} else {
			outname = getOutName("", header_sel);
		}
		writeImage(outname, renWin, false);
	} else {
		// reload widget for each key frame
		for (int kf_idx=0; kf_idx<animator.kfs.size(); kf_idx++){
			loadKF(animator, kf_idx, img, ren1);
			if (header_sel == -1){ // render key frames
				renWin->Render();
				// save file with output directory
				std::string outname;
				if (!output_dir.empty()) {
					outname = output_dir + "/" + getOutName("", kf_idx);
				} else {
					outname = getOutName("", kf_idx);
				}
				writeImage(outname, renWin, false);
			} else if (header_sel == -2){ // renderAllFrames
				glm::vec2 frameRange = animator.kfs[kf_idx].get_fRange();
				for (int f = frameRange[0]; f <= frameRange[1]; f++){
					renWin->Render();
					std::string outname;
					if (!output_dir.empty()) {
						outname = output_dir + "/img_" + std::to_string(kf_idx) + "_f" + std::to_string(f) + ".png";
					} else {
						outname = "img_" + std::to_string(kf_idx) + "_f" + std::to_string(f) + ".png";
					}
					writeImage(outname, renWin, false);
					if (f < frameRange[1]){
						// advance frame 
						animator.kfs[kf_idx].advanceFrame();
						// load camera
						loadCamera(animator.kfs[kf_idx], ren1);
					}
				}
			}
		}
	}
  }

