// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/* This is a small example tutorial how to use OSPRay in an application.
 *
 * On Linux build it in the build_directory with
 *   g++ ../apps/ospTutorial/ospTutorial.cpp -I ../ospray/include \
 *       -I ../../rkcommon -L . -lospray -Wl,-rpath,. -o ospTutorial
 * On Windows build it in the build_directory\$Configuration with
 *   cl ..\..\apps\ospTutorial\ospTutorial.cpp /EHsc -I ..\..\ospray\include ^
 *      -I ..\.. -I ..\..\..\rkcommon ospray.lib
 * Above commands assume that rkcommon is present in a directory right "next
 * to" the OSPRay directory. If this is not the case, then adjust the include
 * path (alter "-I <path/to/rkcommon>" appropriately).
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb_image_write.h"

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

#include "rkcommon/utility/SaveImage.h"
#include "GLFWOSPWindow.h"
#include "RenderFuncs.h"

using json = nlohmann_loader::json;
using namespace visuser;

enum DATATYPE{TRI_MESH, VOL, TOTAL_DATA_TYPES};
std::string dataTypeString[] = {"triangle_mesh", "volume"};

GLFWOSPWindow *GLFWOSPWindow::activeWindow = nullptr;

float clamp(float x)
{
    if (x < 0.f) {
	return 0.f;
    }
    if (x > 1.f) {
	return 1.f;
    }
    return x;
}



ospray::cpp::TransferFunction makeTransferFunction(const vec2f &valueRange, tfnw::TransferFunctionWidget& widget)
{
    ospray::cpp::TransferFunction transferFunction("piecewiseLinear");
    // std::string tfColorMap{"custom"};
    // std::string tfOpacityMap{"opaque"};
    std::string tfColorMap{"jet"};
    std::string tfOpacityMap{"linear"};

  
    std::vector<vec3f> colors;
    std::vector<float> opacities;

    if (tfColorMap == "jet") {
	colors.emplace_back(0, 0, 0.562493);
	colors.emplace_back(0, 0, 1);
	colors.emplace_back(0, 1, 1);
	colors.emplace_back(0.500008, 1, 0.500008);
	colors.emplace_back(1, 1, 0);
	colors.emplace_back(1, 0, 0);
	colors.emplace_back(0.500008, 0, 0);
    } else if (tfColorMap == "custom") {
        // // Custom colormap optimized for data with many near-zero values
        colors.emplace_back(0.05, 0.0, 0.3);       // Very dark purple/blue for near-zero values
        colors.emplace_back(0.1, 0.0, 0.6);        // Dark purple
        colors.emplace_back(0.0, 0.0, 0.9);        // Deep blue
        colors.emplace_back(0.0, 0.5, 1.0);        // Sky blue
        colors.emplace_back(0.0, 0.8, 0.8);        // Cyan
        colors.emplace_back(0.4, 1.0, 0.4);        // Green
        colors.emplace_back(0.8, 1.0, 0.0);        // Yellow-green
        colors.emplace_back(1.0, 0.8, 0.0);        // Yellow
        colors.emplace_back(1.0, 0.4, 0.0);        // Orange
        colors.emplace_back(1.0, 0.0, 0.0);        // Red
        colors.emplace_back(0.7, 0.0, 0.0);        // Dark red
        // colors.emplace_back(0.05, 0.0, 0.3);       // Very dark purple/blue for near-zero values
        // colors.emplace_back(0.1, 0.0, 0.6);        // Dark purple
        // colors.emplace_back(0.0, 0.0, 0.9);        // Deep blue
        // colors.emplace_back(0.0, 0.5, 1.0);        // Sky blue
        // colors.emplace_back(0.0, 0.8, 0.8);        // Cyan
        // colors.emplace_back(0.4, 1.0, 0.4);        // Green
        // colors.emplace_back(0.8, 0.8, 0.0);        // Yellow (slightly muted)
        // colors.emplace_back(1.0, 0.5, 0.0);        // Orange
        // colors.emplace_back(1.0, 0.3, 0.0);        // Orange-red
        // colors.emplace_back(1.0, 0.0, 0.0);        // Bright red
        // colors.emplace_back(0.8, 0.0, 0.0);        // Deep red
        // colors.emplace_back(0.6, 0.0, 0.0);
    }
    else if (tfColorMap == "rgb") {
	colors.emplace_back(0, 0, 1);
	colors.emplace_back(0, 1, 0);
	colors.emplace_back(1, 0, 0);
    } else if (tfColorMap == "gray") {
	colors.emplace_back(0.f, 0.f, 0.f);
	colors.emplace_back(1.f, 1.f, 1.f);
    }

    if (tfOpacityMap == "linear") {
	opacities.emplace_back(0.f);
	opacities.emplace_back(1.f);
	widget.alpha_control_pts[0].x = 0.f;
	widget.alpha_control_pts[0].y = 0.f;
        widget.alpha_control_pts[1].x = 1.f;
	widget.alpha_control_pts[1].y = 1.f;
    } else if (tfOpacityMap == "linearInv") {
	opacities.emplace_back(1.f);
	opacities.emplace_back(0.f);
        widget.alpha_control_pts[0].x = 1.f;
	widget.alpha_control_pts[0].y = 1.f;
        widget.alpha_control_pts[1].x = 0.f;
	widget.alpha_control_pts[1].y = 0.f;
    } else if (tfOpacityMap == "opaque") {
	opacities.emplace_back(1.f);
	opacities.emplace_back(1.f);
        widget.alpha_control_pts[0].x = 0.f;
	widget.alpha_control_pts[0].y = 1.f;
        widget.alpha_control_pts[1].x = 1.f;
	widget.alpha_control_pts[1].y = 1.f;
    }

    transferFunction.setParam("color", ospray::cpp::CopiedData(colors));
    transferFunction.setParam("opacity", ospray::cpp::CopiedData(opacities));
    transferFunction.setParam("valueRange", valueRange);
    transferFunction.commit();
    //    widget.update_colormap();

    return transferFunction;
}

ospray::cpp::TransferFunction loadTransferFunction(json &j, tfnw::TransferFunctionWidget& tfWidget)
{
    ospray::cpp::TransferFunction transferFunction("piecewiseLinear");
    
    std::vector<vec3f> colors3f;
    std::vector<float> colors = j["transferFunc"]["colors"].get<std::vector<float>>();
    std::vector<float> opacities = j["transferFunc"]["opacities"].get<std::vector<float>>();
    glm::vec2 tfRange = get_vec2f(j["transferFunc"]["range"]);
    const vec2f valueRange = {tfRange[0], tfRange[1]};
    // const vec2f valueRange = {35, 38}; //med

    for (uint32_t i=0; i<colors.size()/3; i++){
    	colors3f.emplace_back(colors[i*3], colors[i*3+1], colors[i*3+2]);
    }
    
    tfWidget.set_osp_colormapf(colors, opacities);
    std::cout << "load tf col sz="<< tfWidget.osp_colors.size()<<" "
	      <<tfWidget.alpha_control_pts.size() <<" \n";
    
    transferFunction.setParam("color", ospray::cpp::CopiedData(colors3f));
    transferFunction.setParam("opacity", ospray::cpp::CopiedData(opacities));
    transferFunction.setParam("valueRange", valueRange);
    transferFunction.commit();

    return transferFunction;
    
}

void loadCamera(AnimatorKF &keyframe, GLFWOSPWindow &glfwOspWindow)
{
    // load camera
    visuser::Camera c;
    keyframe.get_current_cam(c);;
    vec3f pos(c.pos[0], c.pos[1], c.pos[2]);
    vec3f dir(c.dir[0], c.dir[1], c.dir[2]);
    vec3f up(c.up[0], c.up[1], c.up[2]);
    ospray::cpp::Camera* camera = &glfwOspWindow.camera;
    camera->setParam("aspect", glfwOspWindow.imgSize.x / (float)glfwOspWindow.imgSize.y);
    camera->setParam("position", pos);
    camera->setParam("direction", dir);
    camera->setParam("up", up);
    camera->commit(); // commit each object to indicate modifications are done*/
}

// void loadCamera(AnimatorKF &keyframe, GLFWOSPWindow &glfwOspWindow)
// {
//     // load camera
//     visuser::Camera c;
//     keyframe.get_current_cam(c);
//     vec3f pos(c.pos[0], c.pos[1], c.pos[2]);
//     vec3f dir(c.dir[0], c.dir[1], c.dir[2]);
//     vec3f up(c.up[0], c.up[1], c.up[2]);
//     ospray::cpp::Camera* camera = &glfwOspWindow.camera;
    
//     // Set consistent aspect ratio
//     camera->setParam("aspect", glfwOspWindow.imgSize.x / (float)glfwOspWindow.imgSize.y);
    
//     // Add a consistent field of view (in degrees)
//     camera->setParam("fovy", 45.0f);  // 45-degree vertical field of view
    
//     // Optional: Add a consistent focal distance if supported by your camera type
//     // camera->setParam("focusDistance", 30.0f);
    
//     camera->setParam("position", pos);
//     camera->setParam("direction", dir);
//     camera->setParam("up", up);
//     camera->commit(); // commit each object to indicate modifications are done
// }


void loadKF(Animator &animator, uint32_t idx, std::vector<float> &voxels, GLFWOSPWindow &glfwOspWindow){
    bool loadVol = false;
    bool loadDerived = false;
    for (int j = 0; j < animator.kfs[idx].get_data_list_size(); ++j) {
        vec3i volumeDimensions;
	std::string file_name;
	double spc[3] = {1.0, 1.0, 1.0};
	
	// The property describes how the data will look
	json data = animator.get_scene_data(idx, j);
	json info = animator.get_scene_info(idx, j);
	if ((data["type"] == "structured") || (data["type"] == "structuredSpherical")){
	    // load only one volume for now
	    if (!loadVol) loadVol = true;
	    else break;
	    // load data info
	    glm::vec3 data_dims = get_vec3f(data["src"]["dims"]);
	    volumeDimensions[0] = data_dims[0];
	    volumeDimensions[1] = data_dims[1];
	    volumeDimensions[2] = data_dims[2];
	    file_name = data["src"]["name"];
	}

	// load volume
	float min=std::numeric_limits<float>::infinity(), max=0;
	voxels.resize (volumeDimensions.long_product());

	std::fstream file;
	file.open(file_name, std::fstream::in | std::fstream::binary);
	std::cout <<"dim "<<volumeDimensions[0]<<" "
		  <<volumeDimensions[1]<<" "
		  <<volumeDimensions[2]<<"\nLoad "<< voxels.size()<< " :";
	for (size_t z=0; z<volumeDimensions[2]; z++){
	    long long offset = z * volumeDimensions[0] * volumeDimensions[1];
	    for (size_t y=0; y<volumeDimensions[1]; y++){
		for (size_t x =0 ; x < volumeDimensions[0]; x++){
		    float buff;
		    file.read((char*)(&buff), sizeof(buff));
		    voxels[offset + y*volumeDimensions[0] + x] = float(buff);
		    if (float(buff) > max) max = float(buff);
		    if (float(buff) < min) min = float(buff);
		}
	    }
	    for (int k=0; k<10; k++)
		if (z == (volumeDimensions[2]/10)*k)
		    std::cout <<z*volumeDimensions[0] * volumeDimensions[1]<<" "<< k<<"0% \n";    
		    }
	std::cout <<"End load \n";
	file.close();
	glfwOspWindow.voxel_data = &voxels;
	glfwOspWindow.tf_range[0] = min;
	glfwOspWindow.tf_range[1] = max;
	std::cout <<"range: "<< max <<" "<<min<<"\n";
	glfwOspWindow.tfn = loadTransferFunction(info, glfwOspWindow.tfn_widget);
	glfwOspWindow.volumeDimensions = volumeDimensions;
    }
    loadCamera(animator.kfs[idx], glfwOspWindow);
}


void init (void* fb, GLFWOSPWindow &glfwOspWindow){
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable( GL_BLEND );
    glGenTextures(1, &glfwOspWindow.texture);
    glBindTexture(GL_TEXTURE_2D, glfwOspWindow.texture);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture

    glTexImage2D(GL_TEXTURE_2D,
		 0,
		 GL_RGBA32F,
		 glfwOspWindow.imgSize.x,
		 glfwOspWindow.imgSize.y,
		 0,
		 GL_RGBA,
		 GL_FLOAT,
		 fb);
}

void renderOnce(GLFWOSPWindow &glfwOspWindow){
    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);      
    glEnable(GL_FRAMEBUFFER_SRGB); // Turn on sRGB conversion for OSPRay frame
    glfwOspWindow.display();
    glfwOspWindow.renderNewFrame();
    glDisable(GL_FRAMEBUFFER_SRGB); // Disable SRGB conversion for UI
    glDisable(GL_TEXTURE_2D);
    // Swap buffers
    glfwMakeContextCurrent(glfwOspWindow.glfwWindow);
    glfwSwapBuffers(glfwOspWindow.glfwWindow);
}


void renderLoop(GLFWOSPWindow &glfwOspWindow){
    std::cout << "Begin render loop\n";
    do{
	glfwPollEvents();
    
	auto t1 = std::chrono::high_resolution_clock::now();
      
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);      
	glEnable(GL_FRAMEBUFFER_SRGB); // Turn on sRGB conversion for OSPRay frame
	glfwOspWindow.display();
	glDisable(GL_FRAMEBUFFER_SRGB); // Disable SRGB conversion for UI
	glDisable(GL_TEXTURE_2D);
      
	// Start the Dear ImGui frame
	ImGui_ImplGlfwGL3_NewFrame();
	glfwOspWindow.buildUI();
	ImGui::Render();
	ImGui_ImplGlfwGL3_Render();
      
	// Swap buffers
	glfwMakeContextCurrent(glfwOspWindow.glfwWindow);
	glfwSwapBuffers(glfwOspWindow.glfwWindow);
      

	auto t2 = std::chrono::high_resolution_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	glfwSetWindowTitle(glfwOspWindow.glfwWindow, (std::string("Render FPS:")+std::to_string(int(1.f / time_span.count()))).c_str());

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(glfwOspWindow.glfwWindow, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
	   glfwWindowShouldClose(glfwOspWindow.glfwWindow) == 0 );
    
}

void run(std::string config_str, std::string output_dir, int header_sel){
    // initialize GLFW
    if (!glfwInit()) {
	throw std::runtime_error("Failed to initialize GLFW!");
    }
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    GLFWOSPWindow glfwOspWindow;

    // create GLFW window
    glfwOspWindow.glfwWindow = glfwCreateWindow(glfwOspWindow.windowSize.x, glfwOspWindow.windowSize.y, "Viewer", nullptr, nullptr);

    if (!glfwOspWindow.glfwWindow) {
	glfwTerminate();
	throw std::runtime_error("Failed to create GLFW window!");
    }
    Animator animator;
    animator.init(config_str.c_str());

    std::vector<float> voxels;
    loadKF(animator, 0, voxels, glfwOspWindow);

    // construct one time objects
    // init volume mesh
    ospray::cpp::Group group;
    // assume uniform data type now
	
    if (animator.get_scene_data(0, 0)["type"] == "structured"){
	glfwOspWindow.initVolume(glfwOspWindow.volumeDimensions, glfwOspWindow.world_size_x);
    }/*else if (widget.type_name == "unstructured"){
       glfwOspWindow.initVolumeOceanZMap(volumeDimensions, glfwOspWindow.world_size_x);
       }*/
    else if (animator.get_scene_data(0, 0)["type"] == "structuredSpherical"){
       std::filesystem::path currentPath = std::filesystem::current_path();
    
       // Build the relative path to the land.png file
       std::string path_to_bgmap = (currentPath / "renderingApps" / "mesh" / "land.png").string();
        
        // As a fallback, try to find it relative to a parent directory
       if (!std::filesystem::exists(path_to_bgmap)) {
           path_to_bgmap = (currentPath.parent_path() / "renderingApps" / "mesh" / "land.png").string();
       }
       
       glfwOspWindow.initVolumeSphere(glfwOspWindow.volumeDimensions, path_to_bgmap);
       group.setParam("geometry", ospray::cpp::CopiedData(glfwOspWindow.gmodel));
    }
    group.setParam("volume", ospray::cpp::CopiedData(glfwOspWindow.model));
    group.commit();

    glfwOspWindow.initClippingPlanes();
    std::cout << "volume loaded\n";

    // ospray objects init
    ospray::cpp::Light light("ambient");
    light.commit();
    glfwOspWindow.instance = ospray::cpp::Instance(group);
    glfwOspWindow.instance.commit();
    glfwOspWindow.world.setParam("instance", ospray::cpp::CopiedData(glfwOspWindow.instance));
    glfwOspWindow.world.setParam("light", ospray::cpp::CopiedData(light));
    glfwOspWindow.world.commit();
    ospray::cpp::Renderer *renderer = &glfwOspWindow.renderer;
    renderer->setParam("aoSamples", 1);
    renderer->setParam("volumeSamplingRate", 10.f);
    // for beige bakcground
    // vec4f bgColor = vec4f(pow(0.9607843137f, 2.2f), pow(0.870588235f, 2.2f), pow(0.7019607843, 2.2f), pow(1.0f, 2.2f));
    // renderer->setParam("backgroundColor", bgColor); 
    // renderer->setParam("backgroundColor", 0.0f); // black, transparent
    renderer->commit();

    glfwOspWindow.preRenderInit();

    if (header_sel >= 0){ // render selected keyframe
	glfwPollEvents();
	auto t1 = std::chrono::high_resolution_clock::now();
	renderOnce(glfwOspWindow);
	auto t2 = std::chrono::high_resolution_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
   
	// save file
    std::string outname;
    if (!output_dir.empty()) {
            outname = output_dir + "/img_kf" + std::to_string(header_sel) + ".png";
    } else {
            outname = "img_kf" + std::to_string(header_sel) + ".png";
    }
    glfwOspWindow.saveFrame(outname);
	std::cout <<"write: "<< outname <<" "<< time_span.count() <<"sec \n\n";

    }else{ // render all key frames

	// reload widget for each key frame
	for (int kf_idx=0; kf_idx<animator.kfs.size(); kf_idx++){
	
	    loadKF(animator, kf_idx, voxels, glfwOspWindow);

	    if (header_sel == -1){// render key frames
		glfwPollEvents();
		auto t1 = std::chrono::high_resolution_clock::now();
		renderOnce(glfwOspWindow);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		// save file
		std::string outname;
        if (!output_dir.empty()) {
            outname = output_dir + "/img_kf" + std::to_string(kf_idx) + ".png";
        } else {
             outname = "img_kf" + std::to_string(kf_idx) + ".png";
        }
		glfwOspWindow.saveFrame(outname);
		std::cout <<"write: "<< outname <<" "<< time_span.count() <<"sec \n\n";

	    }else if (header_sel == -2){//renderAllFrames
		glm::vec2 frameRange = animator.kfs[kf_idx].get_fRange();
		std::cout <<"\nrender frame "
			  << frameRange[0] <<" - "
			  << frameRange[1] <<" sec \n";
    
		for (int f = frameRange[0]; f <= frameRange[1]; f++){
		    glfwPollEvents();
		    auto t1 = std::chrono::high_resolution_clock::now();
		    renderOnce(glfwOspWindow);
		    auto t2 = std::chrono::high_resolution_clock::now();
		    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		    std::string outname;
            if (!output_dir.empty()) {
                outname = output_dir + "/img_kf" + std::to_string(kf_idx) + "f" + std::to_string(f) + ".png";
            } else {
                outname = "img_kf" + std::to_string(kf_idx) + "f" + std::to_string(f) + ".png";
            }
		    glfwOspWindow.saveFrame(outname);
		    std::cout <<"write: f"<< f  <<" "<< time_span.count() <<"sec \n";

		    if (f < frameRange[1]){
			// advance frame 
			animator.kfs[kf_idx].advanceFrame();
			loadCamera(animator.kfs[kf_idx], glfwOspWindow);
		    }
		}
    
		std::cout <<"\n";

	    }
	}
    }
    
    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();

}


void run_interactive_sel(std::string config_str, std::string fname, int header_sel){
    // initialize GLFW
    if (!glfwInit()) {
	throw std::runtime_error("Failed to initialize GLFW!");
    }

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    GLFWOSPWindow glfwOspWindow;

    // create GLFW window
    glfwOspWindow.glfwWindow = glfwCreateWindow(glfwOspWindow.windowSize.x, glfwOspWindow.windowSize.y, "Viewer", nullptr, nullptr);

    if (!glfwOspWindow.glfwWindow) {
	glfwTerminate();
	throw std::runtime_error("Failed to create GLFW window!");
    }
    Animator animator;
    animator.init(config_str.c_str());

    std::vector<float> voxels;
    loadKF(animator, header_sel, voxels, glfwOspWindow);
        

    // construct one time objects
    // init volume mesh
    ospray::cpp::Group group;
    // assume uniform data type now
	
    if (animator.get_scene_data(0, 0)["type"] == "structured"){
	glfwOspWindow.initVolume(glfwOspWindow.volumeDimensions, glfwOspWindow.world_size_x);
    }/*else if (widget.type_name == "unstructured"){
       glfwOspWindow.initVolumeOceanZMap(volumeDimensions, glfwOspWindow.world_size_x);
       }*/
    else if (animator.get_scene_data(0, 0)["type"] == "structuredSpherical"){
       std::string path_to_bgmap = "/home/{username}/projects/vis_interface/vis_user_tool/renderingApps/mesh/land.png";
       glfwOspWindow.initVolumeSphere(glfwOspWindow.volumeDimensions, path_to_bgmap);
       group.setParam("geometry", ospray::cpp::CopiedData(glfwOspWindow.gmodel));
    }
    group.setParam("volume", ospray::cpp::CopiedData(glfwOspWindow.model));
    group.commit();

    glfwOspWindow.initClippingPlanes();
    std::cout << "volume loaded\n";

    // ospray objects init
    ospray::cpp::Light light("ambient");
    light.commit();
    glfwOspWindow.instance = ospray::cpp::Instance(group);
    glfwOspWindow.instance.commit();
    glfwOspWindow.world.setParam("instance", ospray::cpp::CopiedData(glfwOspWindow.instance));
    glfwOspWindow.world.setParam("light", ospray::cpp::CopiedData(light));
    glfwOspWindow.world.commit();
    ospray::cpp::Renderer *renderer = &glfwOspWindow.renderer;;
    renderer->setParam("aoSamples", 1);
    renderer->setParam("volumeSamplingRate", 10.f);
    renderer->setParam("backgroundColor", 0.0f); // black, transparent
    renderer->commit();
    
    // set up arcball camera for ospray
    glfwOspWindow.arcballCamera.reset(new ArcballCamera(glfwOspWindow.world.getBounds<box3f>(), glfwOspWindow.windowSize));
    glfwOspWindow.arcballCamera->updateWindowSize(glfwOspWindow.windowSize);
    std::cout << glfwOspWindow.arcballCamera->eyePos() <<"\n";
    std::cout << glfwOspWindow.arcballCamera->lookDir() <<"\n";
    std::cout << glfwOspWindow.arcballCamera->upDir() <<"\n";
    ospray::cpp::Camera* camera = &glfwOspWindow.camera;
	    
    std::cout << "All osp objects committed\n";
	    
    std::cout << "All osp objects committed\n";

    glfwOspWindow.preRenderInit();
	
	std::cout << "Begin render loop\n";
	do{
	    glfwPollEvents();
    
	    auto t1 = std::chrono::high_resolution_clock::now();
      
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    glEnable(GL_TEXTURE_2D);      
	    glEnable(GL_FRAMEBUFFER_SRGB); // Turn on sRGB conversion for OSPRay frame
	    glfwOspWindow.display();
	    glDisable(GL_FRAMEBUFFER_SRGB); // Disable SRGB conversion for UI
	    glDisable(GL_TEXTURE_2D);
      
	    // Start the Dear ImGui frame
	    ImGui_ImplGlfwGL3_NewFrame();
	    glfwOspWindow.buildUI();
	    ImGui::Render();
	    ImGui_ImplGlfwGL3_Render();
      
	    // Swap buffers
	    glfwMakeContextCurrent(glfwOspWindow.glfwWindow);
	    glfwSwapBuffers(glfwOspWindow.glfwWindow);
      

	    auto t2 = std::chrono::high_resolution_clock::now();
	    auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	    glfwSetWindowTitle(glfwOspWindow.glfwWindow, (std::string("Render FPS:")+std::to_string(int(1.f / time_span.count()))).c_str());

	} // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(glfwOspWindow.glfwWindow, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
	       glfwWindowShouldClose(glfwOspWindow.glfwWindow) == 0 );
    
	ImGui_ImplGlfwGL3_Shutdown();
	glfwTerminate();

}
static bool savingFrame = false;
void run_interactive_in_place(float *input_array,
			      std::vector<std::string> &fnames,
			      int x, int y, int z, int count, int mode,
			      std::string path_to_bgmap){
    // initialize GLFW
    if (!glfwInit()) {
	throw std::runtime_error("Failed to initialize GLFW!");
    }

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    GLFWOSPWindow glfwOspWindow;

    // create GLFW window
    glfwOspWindow.glfwWindow = glfwCreateWindow(glfwOspWindow.windowSize.x, glfwOspWindow.windowSize.y, "Viewer", nullptr, nullptr);

    if (!glfwOspWindow.glfwWindow) {
	glfwTerminate();
	throw std::runtime_error("Failed to create GLFW window!");
    }
    glfwOspWindow.all_data_ptr = input_array;
    glfwOspWindow.count = count;
    glfwOspWindow.setFileNames(fnames);
    glfwOspWindow.volumeDimensions[0] = x; 
    glfwOspWindow.volumeDimensions[1] = y; 
    glfwOspWindow.volumeDimensions[2] = z;
    std::cout << "shape:" <<count <<" of "<< x <<" "<<y<<" "<<z <<std::endl;
    std::cout << "file names: ";
    for (auto f : glfwOspWindow.file_names)
	std::cout << f <<" ";
    std::cout <<"\n";

    // init vol containers
    float min=std::numeric_limits<float>::infinity(), max=0;
    std::vector<float> voxels(glfwOspWindow.volumeDimensions.long_product());
	
    // construct ospray variables
    ospray::cpp::Group group;
    {
	for (long long i =0 ; i < glfwOspWindow.volumeDimensions.long_product(); i++){
	    voxels[i] = glfwOspWindow.all_data_ptr[i];
	    min = std::min(min, voxels[i]);
	    max = std::max(max, voxels[i]);        
	}
	    
	std::cout <<"End load max: "<< max <<" min: "<<min<< "\n";
	glfwOspWindow.voxel_data = &voxels;
	glfwOspWindow.tf_range = vec2f(min, max);
    }
    		
    glfwOspWindow.tfn = makeTransferFunction(vec2f(min, max), glfwOspWindow. tfn_widget);
    	    
    // volume
    //ospray::cpp::Volume volume("structuredSpherical");
    if (mode == 0){
	glfwOspWindow.initVolume(glfwOspWindow.volumeDimensions, glfwOspWindow.world_size_x);
    }/*else if (mode == 1)
       glfwOspWindow.initVolumeOceanZMap(glfwOspWindow.volumeDimensions, glfwOspWindow.world_size_x);
     */
    else if (mode == 2){
       glfwOspWindow.initVolumeSphere(glfwOspWindow.volumeDimensions, path_to_bgmap);
       if (path_to_bgmap!="") group.setParam("geometry", ospray::cpp::CopiedData(glfwOspWindow.gmodel));
    }
	
    group.setParam("volume", ospray::cpp::CopiedData(glfwOspWindow.model));
    group.commit();

    glfwOspWindow.initClippingPlanes();

    // put the group into an instance (give the group a world transform)
    //ospray::cpp::Instance instance(group);
    //instance.commit();
    glfwOspWindow.instance = ospray::cpp::Instance(group);
    glfwOspWindow.instance.commit();

    // put the instance in the world
    ospray::cpp::World world;
    //world.setParam("instance", ospray::cpp::CopiedData(instance));
    glfwOspWindow.world.setParam("instance", ospray::cpp::CopiedData(glfwOspWindow.instance));

    // create and setup light for Ambient Occlusion
    ospray::cpp::Light light("ambient");
    light.setParam("visible", false);
    light.commit();
	
    //world.setParam("light", ospray::cpp::CopiedData(light));
    //world.commit();
    glfwOspWindow.world.setParam("light", ospray::cpp::CopiedData(light));
    glfwOspWindow.world.commit();
    	
    // set up arcball camera for ospray
    glfwOspWindow.arcballCamera.reset(new ArcballCamera(glfwOspWindow.world.getBounds<box3f>(), glfwOspWindow.windowSize));
    glfwOspWindow.arcballCamera->updateWindowSize(glfwOspWindow.windowSize);
    std::cout << "boundbox: "<< glfwOspWindow.world.getBounds<box3f>() << "\n";
    
	
    // create renderer, choose Scientific Visualization renderer
    ospray::cpp::Renderer *renderer = &glfwOspWindow.renderer;
	
    // renderer->setParam("aoSamples", 1);
    // renderer->setParam("backgroundColor", 0.0f); // white, transparent
    // Increase ambient occlusion samples for better boundary definition
    renderer->setParam("aoSamples", 1); // Increase from 1 to 4
    // vec4f bgColor = vec4f(pow(0.9607843137f, 2.2f), pow(0.870588235f, 2.2f), pow(0.7019607843, 2.2f), pow(1.0f, 2.2f));
    // renderer->setParam("backgroundColor", bgColor);
    renderer->setParam("backgroundColor", 0.0f); 
    renderer->commit();
	
    ospray::cpp::Camera* camera = &glfwOspWindow.camera;
	    
    camera->setParam("aspect", glfwOspWindow.imgSize.x / (float)glfwOspWindow.imgSize.y);
    camera->setParam("position", glfwOspWindow.arcballCamera->eyePos());
    camera->setParam("direction", glfwOspWindow.arcballCamera->lookDir());
    camera->setParam("up", glfwOspWindow.arcballCamera->upDir());
    camera->commit(); // commit each object to indicate modifications are done
	
    std::cout << "All osp objects committed\n";
    glfwOspWindow.preRenderInit();	
	
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Begin render loop\n";
    do{
	glfwPollEvents();

    
	t1 = std::chrono::high_resolution_clock::now();
      
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
      
	glEnable(GL_FRAMEBUFFER_SRGB); // Turn on sRGB conversion for OSPRay frame
	glfwOspWindow.display();
	glDisable(GL_FRAMEBUFFER_SRGB); // Disable SRGB conversion for UI
	glDisable(GL_TEXTURE_2D);
      
	// Start the Dear ImGui frame
    if(savingFrame)
    {
        glfwOspWindow.saveFrame("savedFrame.png");
        savingFrame = false;
    }
    else
    {
        ImGui_ImplGlfwGL3_NewFrame();
        glfwOspWindow.buildUI();
        ImGui::Render();
        ImGui_ImplGlfwGL3_Render();
    }

    //poll for right shift and  number 1 key press
    if (glfwGetKey(glfwOspWindow.glfwWindow, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS &&
        glfwGetKey(glfwOspWindow.glfwWindow, GLFW_KEY_1) == GLFW_PRESS)
        {
            std::cout << "Saving frame\n";
            savingFrame = true;
            // glfwOspWindow.saveFrame("savedFrame.png");
        }
	
      
	// Swap buffers
	glfwMakeContextCurrent(glfwOspWindow.glfwWindow);
	glfwSwapBuffers(glfwOspWindow.glfwWindow);
      

	t2 = std::chrono::high_resolution_clock::now();
	auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	glfwSetWindowTitle(glfwOspWindow.glfwWindow, (std::string("Render FPS:")+std::to_string(int(1.f / time_span.count()))).c_str());

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(glfwOspWindow.glfwWindow, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
	   glfwWindowShouldClose(glfwOspWindow.glfwWindow) == 0 );

	
    glfwOspWindow.printSessionSummary();
    
    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();
	
}
    


