#include "GLFWOSPWindow.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//
// Lower level rendering contents
//


void GLFWOSPWindow::display()
{
    if (kf_widget.play) playAnimationFrame(); 
  
    glBindTexture(GL_TEXTURE_2D, texture);
    // render textured quad with OSPRay frame buffer contents
    renderNewFrame();
    auto fb = framebuffer.map(OSP_FB_COLOR);

    glTexImage2D(GL_TEXTURE_2D,
		 0,
		 GL_RGBA32F,
		 imgSize.x,
		 imgSize.y,
		 0,
		 GL_RGBA,
		 GL_FLOAT,
		 fb);
    framebuffer.unmap(fb);
   
   
    glBegin(GL_QUADS);

    glTexCoord2f(0.f, 0.f);
    glVertex2f(0.f, 0.f);

    glTexCoord2f(0.f, 1.f);
    glVertex2f(0.f, windowSize.y);

    glTexCoord2f(1.f, 1.f);
    glVertex2f(windowSize.x, windowSize.y);

    glTexCoord2f(1.f, 0.f);
    glVertex2f(windowSize.x, 0.f);

    glEnd();
}

void GLFWOSPWindow::saveFrame(std::string filename){
    int width, height;
    glfwGetFramebufferSize(glfwWindow, &width, &height);
    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(filename.c_str(), width, height, nrChannels, buffer.data(), stride);


	// uint32_t *fb = (uint32_t *)framebuffer.map(OSP_FB_COLOR);
	// stbi_flip_vertically_on_write(true);
	// stbi_write_png(filename.c_str(), width, height, nrChannels, fb, stride);
	// framebuffer.unmap(fb);
	
}

void GLFWOSPWindow::initBGMap(){
    // triangle mesh data
    std::vector<vec3f> vertex;
    std::vector<vec4f> color;
    std::vector<vec4ui> index;

    int mapx = 864;
    int mapy = 648;
    const char mapfile[] = "/home/{username}/projects/vis_interface/vis_user_tool/renderingApps/mesh/land.png";

    int comp;
    stbi_set_flip_vertically_on_load(1);
    float* image = stbi_loadf(mapfile, &mapx, &mapy, &comp, 0);

    if(image == nullptr)
	throw(std::string("Failed to load texture"));
    else std::cout << "load earth image: " << mapx <<" "<<mapy <<" "<<comp <<"\n";
    
    for (size_t j=0; j<mapy; j++){
	for (size_t i=0; i<mapx; i++){
	    float angle_h = i/float(mapx)*M_PI*2; // radius h, 0-PI
	    float angle_v = j/float(mapy)*M_PI; // radius v, 0-PI/2;
	    float r = 10; // radius r, 0-1
	    float p_x = r*sin(angle_v)*cos(angle_h);
	    float p_y = r*sin(angle_v)*sin(angle_h);
	    float p_z = r*cos(angle_v);

	    vertex.push_back(vec3f(p_x, p_y, p_z));
	    size_t c_i = (j*mapx+i)*comp;
	    color.push_back(vec4f(image[c_i], image[c_i+1], image[c_i+2], image[c_i+3]));
	    //color.push_back(vec4f(image[c_i], image[c_i+1], image[c_i+2], 1.0f));
	    //color.push_back(vec4f(float(i)/mapx, float(j)/mapy, 0, 1.f));
	}
    }

    for (size_t i=0; i<mapx-1; i++){
	for (size_t j=0; j<mapy-1; j++){
	    index.push_back(vec4ui(i + j*mapx, i+1+j*mapx, i+1+(j+1)*mapx, i+(j+1)*mapx));
	}
    }
    
    // create and setup model and mesh
    ospray::cpp::Geometry mesh("mesh");
    mesh.setParam("vertex.position", ospray::cpp::CopiedData(vertex));
    mesh.setParam("vertex.color", ospray::cpp::CopiedData(color));
    mesh.setParam("index", ospray::cpp::CopiedData(index));
    mesh.commit();
    
    
    // put the mesh into a model
    ospray::cpp::GeometricModel model_(mesh);
    model_.commit();
    gmodel = model_;
}

void GLFWOSPWindow::initBGMap(const std::string mapfile){
    // triangle mesh data
    std::vector<vec3f> vertex;
    std::vector<vec4f> color;
    std::vector<vec4ui> index;

    int mapx = 100;
    int mapy = 100;

    int comp;
    stbi_set_flip_vertically_on_load(1);
    float* image = stbi_loadf(mapfile.c_str(), &mapx, &mapy, &comp, 0);

    if(image == nullptr)
	throw(std::string("Failed to load texture"));
    else std::cout << "load earth image: " << mapx <<" "<<mapy <<" "<<comp <<"\n";
    
    for (size_t j=0; j<mapy; j++){
	for (size_t i=0; i<mapx; i++){
	    float angle_h = i/float(mapx)*M_PI*2; // radius h, 0-PI
	    float angle_v = j/float(mapy)*M_PI; // radius v, 0-PI/2;
	    float r = 11.f; // radius r, 0-1
	    float p_x = r*sin(angle_v)*cos(angle_h);
	    float p_y = r*sin(angle_v)*sin(angle_h);
	    float p_z = r*cos(angle_v);
	    vertex.push_back(vec3f(p_x, p_y, p_z));
	    
	    size_t c_i = (j*mapx+i)*comp;
	    color.push_back(vec4f(image[c_i], image[c_i+1], image[c_i+2], image[c_i+3]));
	}
    }
    for (size_t i=0; i<mapx-1; i++){
	for (size_t j=0; j<mapy-1; j++){
	    size_t offset = 0;
	    index.push_back(vec4ui(offset+i + j*mapx,
				   offset+i+1+j*mapx,
				   offset+i+1+(j+1)*mapx,
				   offset+i+(j+1)*mapx));
	}
    }
    
    for (size_t j=0; j<mapy; j++){
	for (size_t i=0; i<mapx; i++){
	    float angle_h = i/float(mapx)*M_PI*2; // radius h, 0-PI
	    float angle_v = j/float(mapy)*M_PI; // radius v, 0-PI/2;
	    float r_low = 10; // radius r, 0-1
	    float p_x_low = r_low*sin(angle_v)*cos(angle_h);
	    float p_y_low = r_low*sin(angle_v)*sin(angle_h);
	    float p_z_low = r_low*cos(angle_v);

	    vertex.push_back(vec3f(p_x_low, p_y_low, p_z_low));
	    color.push_back(vec4f(0.f, 0.f, 0.f, 1.f));
	}
    }

    for (size_t i=0; i<mapx-1; i++){
	for (size_t j=0; j<mapy-1; j++){
	    size_t offset = mapx*mapy;
	    index.push_back(vec4ui(offset+i + j*mapx,
				   offset+i+1+j*mapx,
				   offset+i+1+(j+1)*mapx,
				   offset+i+(j+1)*mapx));
	}
    }
    
    // create and setup model and mesh
    ospray::cpp::Geometry mesh("mesh");
    mesh.setParam("vertex.position", ospray::cpp::CopiedData(vertex));
    mesh.setParam("vertex.color", ospray::cpp::CopiedData(color));
    mesh.setParam("index", ospray::cpp::CopiedData(index));
    mesh.commit();
    
    // put the mesh into a model
    ospray::cpp::GeometricModel model_(mesh);
    model_.commit();

    gmodel = model_;
}

void GLFWOSPWindow::initVolume(vec3i volumeDimensions, visuser::AniObjWidget &widget){
 
	// volume
	//ospray::cpp::Volume volume("structuredSpherical");
	ospray::cpp::Volume vol("structuredRegular");
        float world_scale_xy = 1;
	for (size_t i=0; i<2; i++)
	    world_scale_xy = std::min(world_scale_xy, float(widget.world_bbox[i]/volumeDimensions[i]));
	float world_scale_z = widget.world_bbox[0] / widget.world_bbox[2] * world_scale_xy;
	vol.setParam("gridOrigin", vec3f(0.f,0.f,0.f));
	vol.setParam("gridSpacing", vec3f(world_scale_xy, world_scale_xy, world_scale_z));
	//volume.setParam("gridSpacing", vec3f(10.f, 180.f / volumeDimensions.y, 360.f/volumeDimensions.z));
	vol.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
	vol.setParam("dimensions", volumeDimensions);
	vol.commit();
	
	// put the mesh into a model
	ospray::cpp::VolumetricModel mdl(vol);
	mdl.setParam("transferFunction", tfn);
	mdl.commit();
	    
	volume = vol;
	model = mdl;

	volume_type = "structured";
   
}

void GLFWOSPWindow::initVolume(vec3i volumeDimensions, float bb_x){
	ospray::cpp::Volume vol("structuredRegular");
	vol.setParam("gridOrigin", vec3f(0.f,0.f,0.f));
	vol.setParam("gridSpacing", vec3f(bb_x / volumeDimensions.x));
	vol.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
	vol.setParam("dimensions", volumeDimensions);
	vol.commit();
	
	// put the mesh into a model
	ospray::cpp::VolumetricModel mdl(vol);
	mdl.setParam("transferFunction", tfn);
	mdl.commit();
	    
	volume = vol;
	model = mdl;
	volume_type = "structured";
}


void GLFWOSPWindow::initVolumeSphere(vec3i volumeDimensions, const std::string path_to_bgmap){
    // volume
    ospray::cpp::Volume vol("structuredSpherical");
    //ospray::cpp::Volume vol("structuredRegular");
    float world_scale_xy = 1;
    vol.setParam("gridOrigin", vec3f(world_size_x+1.f, 10.f, 0.f));
    //vol.setParam("gridSpacing", vec3f(world_scale_xy, world_scale_xy, world_scale_z));
    vol.setParam("gridSpacing", vec3f(-1.f/volumeDimensions.x,
				      (80+50.f) / volumeDimensions.y,
				      360.f/volumeDimensions.z));
    vol.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
    vol.setParam("dimensions", volumeDimensions);
    vol.commit();
	
    // put the mesh into a model
    ospray::cpp::VolumetricModel mdl(vol);
    mdl.setParam("transferFunction", tfn);
    mdl.commit();
	    
    volume = vol;
    model = mdl;
    volume_type = "structuredSpherical";
    
    if (path_to_bgmap != "")
    	initBGMap(path_to_bgmap);
}


void GLFWOSPWindow::initVolumeOceanZMap(vec3i volumeDimensions, float bbx){
    ospray::cpp::Volume vol("unstructured");
    float world_scale_xy = bbx/volumeDimensions.x;
    float world_scale_z = 0.0005;
    const float zMap_full[] = {0.5,1.6,2.8,4.2,5.8,7.6,9.7,12,14.7,17.7,21.1,25,29.3,34.2,39.7,45.8,52.7,60.3,68.7,78,88.2,99.4,112,125,139,155,172,190,209,230,252,275,300,325,352,381,410,441,473,507,541,576,613,651,690,730,771,813,856,900,946,992,1040,1089,1140,1192,1246,1302,1359,1418,1480,1544,1611,1681,1754,1830,1911,1996,2086,2181,2281,2389,2503,2626,2757,2898,3050,3215,3392,3584,3792,4019,4266,4535,4828,5148,5499,5882,6301,6760};

    if (volumeDimensions[2] == 90){
	for (size_t i=0; i<90; i++)
	    zMap.push_back(zMap_full[i] * world_scale_z);
    }else if(volumeDimensions[2] == 45){
	for (size_t i=0; i<45; i++)
	    zMap.push_back(zMap_full[i*2] * world_scale_z);
    }else if(volumeDimensions[2] == 23){
	for (size_t i=0; i<23; i++)
	    zMap.push_back(zMap_full[i*4] * world_scale_z);
    }else if(volumeDimensions[2] == 12){
	for (size_t i=0; i<12; i++)
	    zMap.push_back(zMap_full[i*8] * world_scale_z);
    }else std::cout << "z dim not matched!\n";

    std::cout << "end zload\n";

    //if (!isSphere)
    rectMesh.initMesh(volumeDimensions, zMap, vec2f(world_scale_xy));
    //else
    //	rectMesh.initSphereMesh(volumeDimensions, zMap, vec2f(world_scale_xy));
    vol.setParam("vertex.position", ospray::cpp::SharedData(rectMesh.vertices));
    vol.setParam("vertex.data", ospray::cpp::SharedData(*(voxel_data)));
    vol.setParam("background", ospray::cpp::SharedData(vec3f(1.0f, 0.0f, 1.0f)));
    vol.setParam("index", ospray::cpp::SharedData(rectMesh.indices));
    vol.setParam("cell.index", ospray::cpp::SharedData(rectMesh.cells));
    vol.setParam("cell.type", ospray::cpp::SharedData(rectMesh.cellTypes));
    vol.commit();
	    
    // put the mesh into a model
    ospray::cpp::VolumetricModel mdl(vol);
    mdl.setParam("transferFunction", tfn);
    mdl.commit();
	    
    volume = vol;
    model = mdl;
    volume_type = "unstructured";
}

void GLFWOSPWindow::initClippingPlanes(){    
    // clipping plane
    clipping_params = std::array<ClippingPlaneParams, 4>{
	ClippingPlaneParams(0, vec3f(world_size_x, 0, 0)),
	ClippingPlaneParams(0, vec3f(-world_size_x, 0, 0)),
	ClippingPlaneParams(1, vec3f(0, world_size_x, 0)),
	ClippingPlaneParams(1, vec3f(0, -world_size_x, 0))};
    
    clipping_params.changed = false;
    clipping_planes = {
	ClippingPlane(clipping_params.param[0]),
	ClippingPlane(clipping_params.param[1]),
	ClippingPlane(clipping_params.param[2]),
	ClippingPlane(clipping_params.param[3])};
    
    for (int i=0; i<clipping_params.param.size()/2; i++){
	clipping_params.param[i*2].flip_plane = true;
    }

}

void GLFWOSPWindow::setFileNames(std::vector<std::string> names){
    file_names.resize(0);
    for (auto i : names) file_names.push_back(i);
}

void GLFWOSPWindow::preRenderInit(){
    renderNewFrame();
		
    glfwMakeContextCurrent(glfwWindow);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    ImGui_ImplGlfwGL3_Init(glfwWindow, true);
    ImGui::StyleColorsDark();

	ImGuiIO& io = ImGui::GetIO();
	io.FontGlobalScale = 1.75;
    
    auto fb = framebuffer.map(OSP_FB_COLOR);
	
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture

    glTexImage2D(GL_TEXTURE_2D,
		 0,
		 GL_RGBA32F,
		 imgSize.x,
		 imgSize.y,
		 0,
		 GL_RGBA,
		 GL_FLOAT,
		 fb);

    framebuffer.unmap(fb);
    setFunc();
    reshape(windowSize.x, windowSize.y);
    glfwSetInputMode(glfwWindow, GLFW_STICKY_KEYS, GL_TRUE);
}



//
//               User input
//

void GLFWOSPWindow::motion(double x, double y)
{
    const vec2f mouse(x, y);
    if (previousMouse != vec2f(-1)) {
	const bool leftDown =
	    glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	const bool rightDown =
	    glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
	const bool middleDown =
	    glfwGetMouseButton(glfwWindow, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
	const vec2f prev = previousMouse;

	bool cameraChanged = leftDown || rightDown || middleDown;

	// don't modify camera with mouse on ui window
	if (ImGui::GetIO().WantCaptureMouse){ 
	    return;
	}
    
	if (leftDown) {
	    const vec2f mouseFrom(std::clamp(prev.x * 2.f / windowSize.x - 1.f, -1.f, 1.f),
				  std::clamp(prev.y * 2.f / windowSize.y - 1.f, -1.f, 1.f));
	    const vec2f mouseTo(std::clamp(mouse.x * 2.f / windowSize.x - 1.f, -1.f, 1.f),
				std::clamp(mouse.y * 2.f / windowSize.y - 1.f, -1.f, 1.f));
	    arcballCamera->rotate(mouseFrom, mouseTo);
	} else if (rightDown) {
	    arcballCamera->zoom(mouse.y - prev.y);
	} else if (middleDown) {
	    arcballCamera->pan(vec2f(mouse.x - prev.x, prev.y - mouse.y));
	}

	if (cameraChanged) {
	    //updateCamera();
	    //addObjectToCommit(camera.handle());
	    camera.setParam("aspect", windowSize.x / float(windowSize.y));
	    camera.setParam("position", arcballCamera->eyePos());
	    camera.setParam("direction", arcballCamera->lookDir());
	    camera.setParam("up", arcballCamera->upDir());
	    camera.commit();
	}
    }

    previousMouse = mouse;

}


void GLFWOSPWindow::reshape(int w, int h)
{
  
    windowSize.x = w;
    windowSize.y = h;

	printf("Reshape window size: %d %d\n", w, h);
  
    // create new frame buffer
    auto buffers = OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM | OSP_FB_ALBEDO
	| OSP_FB_NORMAL;
    framebuffer =
	ospray::cpp::FrameBuffer(imgSize.x, imgSize.y, OSP_FB_RGBA32F, buffers);
    framebuffer.commit();
  
    glViewport(0, 0, windowSize.x, windowSize.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, windowSize.x, 0.0, windowSize.y, -1.0, 1.0);

    // update camera
    // arcballCamera->updateWindowSize(windowSize);

    camera.setParam("aspect", windowSize.x / float(windowSize.y));
    camera.commit();

}

//
//               GUI
//

void GLFWOSPWindow::buildUI(){
    static float f = 1.f;
    static bool changeF = false;
    static int curTime = 0;
    static int data_time = 0;
    static float slider_tf_max = tf_range[1];
    static float slider_tf_min = tf_range[0];
    static float map_on = true;
    static ImGui::FileBrowser fileDialog;
    // (optional) set browser properties
    fileDialog.SetTitle("land map");
    fileDialog.SetTypeFilters({ ".png"});
    
	// ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
	// Replace with:
	ImGuiWindowFlags flags = 0; // Remove auto resize
	ImGui::SetNextWindowSize(ImVec2(600, 1150)); // Set preferred width and height
    bool camera_need_commit = false;
    bool tf_need_commit = false;

	
    ImGui::Begin("Menu Window", nullptr, flags);
    
    if (volume_type == "structuredSpherical"){
    	ImGui::Text("Spherical view options"); 
    	if (ImGui::Button("toggle bg map")) {
    	    map_on = !map_on;
    	    ospray::cpp::Group tmp_group;
    	    if (map_on) tmp_group.setParam("geometry", ospray::cpp::SharedData(gmodel));
    	    tmp_group.setParam("volume", ospray::cpp::SharedData(model));
    	    tmp_group.commit();
    	    instance = ospray::cpp::Instance(tmp_group);
    	    instance.commit();
	    world.setParam("instance", ospray::cpp::SharedData(instance));
	    world.commit();
    	}
    	ImGui::SameLine();
    	if (ImGui::Button("load bg image")){
    	    fileDialog.Open();
    	}
    	fileDialog.Display();
        
        if(fileDialog.HasSelected())
        {
            std::cout << "Selected filename" << fileDialog.GetSelected().string() << std::endl;
            bgImg = fileDialog.GetSelected().string();
            initBGMap(bgImg);
            ospray::cpp::Group tmp_group;
    	    if (map_on) tmp_group.setParam("geometry", ospray::cpp::SharedData(gmodel));
    	    tmp_group.setParam("volume", ospray::cpp::SharedData(model));
    	    tmp_group.commit();
    	    instance = ospray::cpp::Instance(tmp_group);
    	    instance.commit();
	    world.setParam("instance", ospray::cpp::SharedData(instance));
	    world.commit();
            fileDialog.ClearSelected();
        }
    }

    ImGui::Text("Options");               // Display some text (you can use a format strings too)
    if (ImGui::TreeNode("clipping planes")){
	for (size_t i = 0; i < clipping_params.param.size(); ++i) {
	    ImGui::PushID(i);
	    ImGui::Separator();
	    auto &plane = clipping_params.param[i];

	    ImGui::Text("Clipping Plane on axis %d", plane.axis);
	    clipping_params.changed |= ImGui::Checkbox("Enabled", &plane.enabled);
		
	    if (ImGui::SliderFloat("Position",
				   &clippingBox[i],
				   -1, 0)) {
		plane.position[plane.axis] = -clippingBox[i]*world_size_x;
		clipping_params.changed = true;
	    }

	    ImGui::PopID();
	}
	ImGui::TreePop();
    }
	
    if (clipping_params.changed) {
	clipping_params.changed = false;

	for (size_t i = 0; i < clipping_params.param.size(); ++i) {
	    clipping_planes[i].update(clipping_params.param[i]);
	    clipping_planes[i].geom.commit();
	    clipping_planes[i].model.commit();
	    clipping_planes[i].group.commit();
	    clipping_planes[i].instance.commit();
	}

	std::vector<cpp::Instance> active_instances = {instance};
	for (auto &p : clipping_planes) {
	    if (p.params.enabled) {
		active_instances.push_back(p.instance);
	    }
	}
	world.setParam("instance", cpp::CopiedData(active_instances));
	world.commit();
    }

    if (ImGui::TreeNode("Transfer Function")){
	if (ImGui::SliderFloat("tf min", &slider_tf_min, tf_range[0], tf_range[1])){
	    tfn.setParam("valueRange", vec2f(slider_tf_min, slider_tf_max));
	    tfn.commit();
	    model.commit();
	}

	if (ImGui::SliderFloat("tf max", &slider_tf_max, tf_range[0], tf_range[1])){
	    tfn.setParam("valueRange", vec2f(slider_tf_min, slider_tf_max));
	    tfn.commit();
	    model.commit();
	}
    
	//if (ImGui::SliderFloat("float", &f, 0.01f, 30.0f)){changeF = true;}
    
  	if (tfn_widget.changed() || changeF) {
	    std::vector<float> tmpOpacities, tmpColors; 
	    tfn_widget.get_osp_colormapf(tmpColors, tmpOpacities);
	    for (uint32_t i=0;i<tmpOpacities.size();i++){
	    	tmpOpacities[i] = tmpOpacities[i]*f;
	    }
	    
	    tfn_widget.setUnchanged();
	    changeF = false;

	    // update opacities only
	    tfn.setParam("opacity", ospray::cpp::CopiedData(tmpOpacities));
	    tfn.commit();
	    model.commit();
	}

  
	tfn_widget.draw_ui();
	ImGui::TreePop();
    }
    
    if (ImGui::TreeNode("Data time")){
  	if (ImGui::SliderInt("time", &data_time, 0, count-1)) {
	    long long offset = data_time * volumeDimensions.long_product();
	    for (long long i =0 ; i < volumeDimensions.long_product(); i++){
                (*voxel_data)[i] = all_data_ptr[i+offset];
            }
	    volume.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
	    volume.commit();
	    model.commit();
	}
	ImGui::TreePop();
    }
    
    if (ImGui::TreeNode("Animation keyframe")){
	ImGui::SameLine();
        if (ImGui::Button("export")) {
	    int dims[3] = {volumeDimensions.x, volumeDimensions.y, volumeDimensions.z};
	    int world_bbox[3] = {world_size_x, world_size_x, world_size_x};
	    kf_widget.exportKFs(dims, volume_type, world_bbox, file_names, slider_tf_min, slider_tf_max, bgImg);
	    std::cout << "keyframes exported, data dims: "
		      << dims[0] <<" "<<dims[1] <<" "<< dims[2]
		      << " tf ranges" << slider_tf_min <<" "<<slider_tf_max <<"\n";
	}


	ImGui::Text("Generate Key frames");
	const char* items[] = { "fixedCam"};
	static int item_current_idx = 0; // Here we store our selection data as an index.
	const char* combo_preview_value = items[item_current_idx];

	if (ImGui::BeginCombo("", combo_preview_value))
	    {
		for (int n = 0; n < IM_ARRAYSIZE(items); n++)
		    {
			const bool is_selected = (item_current_idx == n);
			if (ImGui::Selectable(items[n], is_selected))
			    item_current_idx = n;

			if (is_selected)
			    ImGui::SetItemDefaultFocus();
		    }
		ImGui::EndCombo();
	    }
	ImGui::SameLine();
	if (ImGui::Button("set")) {
	    std::vector<float> tmpOpacities, tmpColors;
	    tfn_widget.get_osp_colormapf(tmpColors, tmpOpacities);
	    for (uint32_t i=0;i<tmpOpacities.size();i++){
	    	tmpOpacities[i] = tmpOpacities[i]*f;
	    }
	    
	    kf_widget.generatePresetKF(items[item_current_idx], *arcballCamera, tmpColors, tmpOpacities, count, clippingBox);
	    
	    std::cout << "Generate " << items[item_current_idx] <<"\n";
	}
	ImGui::NewLine();
	
	kf_widget.draw_ui();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255,255,0,255));
	ImGui::Text("active kf:");
	ImGui::PopStyleColor();
	ImGui::SameLine();
	if (ImGui::Button("setF")) {
	    std::vector<float> tmpOpacities, tmpColors;
	    tfn_widget.get_osp_colormapf(tmpColors, tmpOpacities);
	    for (uint32_t i=0;i<tmpOpacities.size();i++){
	    	tmpOpacities[i] = tmpOpacities[i]*f;
	    }
	    
	    kf_widget.setKeyFrame(*arcballCamera, tmpColors, tmpOpacities, data_time);
	    
	    // update camera
	    camera_need_commit = true;

	}
	ImGui::SameLine();
	if (ImGui::Button("loadF")) {
	  
	    ArcballCamera cam;
	    std::vector<float> tmpOpacities, tmpColors;
	    kf_widget.loadKeyFrame(cam, tmpColors, tmpOpacities, data_time);
	    
	    // update camera
	    arcballCamera->set(cam);
	    camera_need_commit = true;

	    //update tf
	    //tfn_widget.set_osp_colormapf(tmpColors, tmpOpacities);
	    //tfn_widget.changed();
	    
	    // TODO update scene
	    long long offset = data_time * volumeDimensions.long_product();
	    for (long long i =0 ; i < volumeDimensions.long_product(); i++){
                (*voxel_data)[i] = all_data_ptr[i+offset];
            }
	    volume.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
	    volume.commit();
	    model.commit();
        }
	ImGui::SameLine();
        if (ImGui::Button("play")) {kf_widget.play = true;}

	if (kf_widget.kfs.size() == 0) kf_widget.record_frame = 0;
	
	// record keyframe if requested
	if (kf_widget.record_frame != -1){
	    std::vector<float> tmpOpacities, tmpColors; 
	    tfn_widget.get_osp_colormapf(tmpColors, tmpOpacities);
	    kf_widget.recordKeyFrame(*arcballCamera, tmpColors, tmpOpacities, data_time, clippingBox);

	}

	if (camera_need_commit){
	    camera.setParam("aspect", windowSize.x / float(windowSize.y));
	    camera.setParam("position", arcballCamera->eyePos());
	    camera.setParam("direction", arcballCamera->lookDir());
	    camera.setParam("up", arcballCamera->upDir());
	    camera.commit();
	}
	
	ImGui::TreePop();
    }
	  
	  
	  
    ImGui::End();
}

void GLFWOSPWindow::playAnimationFrame(){
    if (kf_widget.playFrame >= kf_widget.kfs.back().timeFrame){
	kf_widget.playFrame = -1;
	kf_widget.play = false;
    }
    
    // load current frame info
    float cam[9];
    std::vector<float> tmpOpacities, tmpColors;
    int data_index;
    kf_widget.getFrameFromKF(cam, tmpColors, tmpOpacities, data_index, kf_widget.playFrame);
    vec3f pos(cam[0], cam[1], cam[2]);
    vec3f dir(cam[3], cam[4], cam[5]);
    vec3f up(cam[6], cam[7], cam[8]);

    // update timestep
    long long offset = data_index * volumeDimensions.long_product();
    for (long long i =0 ; i < volumeDimensions.long_product(); i++){
	(*voxel_data)[i] = all_data_ptr[i+offset];
    }
    volume.setParam("data", ospray::cpp::SharedData(voxel_data->data(), volumeDimensions));
    volume.commit();
    //model.commit();
    
    // update camera
    camera.setParam("aspect", windowSize.x / float(windowSize.y));
    camera.setParam("position", pos);
    camera.setParam("direction", dir);
    camera.setParam("up", up);
    camera.commit();
	    
    // update opacities only
    tfn.setParam("opacity", ospray::cpp::CopiedData(tmpOpacities));
    tfn.commit();
    model.commit();

    kf_widget.playFrame++;
}


void GLFWOSPWindow::printSessionSummary(){
    float bbox[4];
    std::vector<float> data_indices;
    kf_widget.getDataFilterFromKF(bbox, data_indices);

    float y_scale = float(volumeDimensions[1])/volumeDimensions[0];

    bbox[0] += 1;
    bbox[1] += 1;
    // flip y start end
    float tmp = bbox[2];
    bbox[2] = -bbox[3]/y_scale;
    bbox[3] = -tmp/y_scale;

    std::cerr << "bbox:" << bbox[0] <<" "<<bbox[1]<<" "<<bbox[2] <<" "<<bbox[3]<<"\nData: ";
    for (auto &i : data_indices)
	std::cerr << i <<" ";
    std::cerr <<"\n";
}
