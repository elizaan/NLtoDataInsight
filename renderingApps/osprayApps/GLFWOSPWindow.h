#include "ospray/ospray_cpp.h"
#include "ospray/ospray_cpp/ext/rkcommon.h"
#include <vector>


#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>
// imgui
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw_gl3.h"
#include "../imgui/imfilebrowser.h"

#include "../../loader.h"
#include "ArcballCamera.h"
#include "TransferFunctionWidget.h"
#include "clipping_plane.h"
#include "app_params.h"

#include "KeyframeWidget.h"
#include "rectMesh.h"

using namespace rkcommon::math;

class GLFWOSPWindow{
public:
    // image size
    // vec2i imgSize{2048, 2048};
    // vec2i windowSize{2048, 2048};
    vec2i imgSize{2048, };
    vec2i windowSize{2048, 2048};
    // For high-quality rendering at 16:9 aspect ratio
    // vec2i imgSize{1920, 1080};  // 2K rendering resolution
    // vec2i windowSize{1280, 720}; // Smaller window with same aspect ratio
    unsigned int texture;
    unsigned int guiTextures[128];
    unsigned int guiTextureSize = 0;
    GLFWwindow *glfwWindow = nullptr;

    int world_size_x = 10;
    
    ospray::cpp::Camera camera{"perspective"};
    ospray::cpp::Renderer renderer{"scivis"};
    ospray::cpp::World world;
    ospray::cpp::Instance instance;
    ospray::cpp::VolumetricModel model;
    ospray::cpp::GeometricModel gmodel;
    ospray::cpp::Volume volume;
    std::vector<float> *voxel_data; // pointer to voxels data
    std::vector<std::string> file_names; // file names of all data 
    vec3i volumeDimensions;
    float* all_data_ptr; // pointer to all data
    int count = 1;
    std::string volume_type="structured";
    std::string bgImg="";
  
    static GLFWOSPWindow *activeWindow;
    ospray::cpp::FrameBuffer framebuffer;
    std::unique_ptr<ArcballCamera> arcballCamera;
    vec2f previousMouse{vec2f(-1)};

    vec2f tf_range;
    ospray::cpp::TransferFunction tfn{"piecewiseLinear"};
    tfnw::TransferFunctionWidget tfn_widget;
    keyframe::KeyframeWidget kf_widget;
    visuser::RectMesh rectMesh;
    std::vector<float> zMap;
    std::vector<float> clippingBox = {-1,0,-1,0};
    std::array<ClippingPlane, 4> clipping_planes;
    AppParam<std::array<ClippingPlaneParams, 4>> clipping_params;
  
    GLFWOSPWindow(){
	activeWindow = this;
    
	/// prepare framebuffer
	auto buffers = OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM | OSP_FB_ALBEDO
	    | OSP_FB_NORMAL;
	framebuffer = ospray::cpp::FrameBuffer(imgSize.x, imgSize.y, OSP_FB_RGBA32F, buffers);
    std::cout << "Framebuffer initialized with size: " << imgSize.x << "x" << imgSize.y << std::endl;
    
    }
  
    void display();
    void saveFrame(std::string filename);
    void motion(double, double);
    void mouse(int, int, int, int);
    void reshape(int, int);
    
    void setFunc(){

	glfwSetCursorPosCallback(
				 glfwWindow, [](GLFWwindow *, double x, double y) {
				     activeWindow->motion(x, y);
				 });
	glfwSetFramebufferSizeCallback(
				       glfwWindow, [](GLFWwindow *, int newWidth, int newHeight) {
					   activeWindow->reshape(newWidth, newHeight);
				       });

    
    }
    
    void initVolume(vec3i dim, visuser::AniObjWidget &widget);
    void initVolume(vec3i dim, float bb_x);
    void initVolumeOceanZMap(vec3i volumeDimensions, float bb_x);
    void initVolumeSphere(vec3i dim, const std::string s);
    void initBGMap();
    void initBGMap(const std::string s);
    void initClippingPlanes();
    void setFileNames(std::vector<std::string> names);
    
    void preRenderInit();    
    void renderNewFrame(){

	framebuffer.clear();
	// render one frame
	framebuffer.renderFrame(renderer, camera, world);
    }

    void buildUI();
    void playAnimationFrame();
    void printSessionSummary();

};
