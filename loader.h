#pragma once

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "json.hpp"
#include "util.h"
#include "ext/glm/glm.hpp"

namespace visuser{

    struct Camera {
    	glm::vec3 pos;
    	glm::vec3 dir;
	glm::vec3 focalPoint; // when JSON provides position+focalPoint this stores the world focal point
	glm::vec3 dir_unit;   // unit direction vector (from pos toward focalPoint). Added for convenience
    	glm::vec3 up;
    	uint32_t frame; 

	Camera();
    	Camera(const glm::vec3 &pos, const glm::vec3 &dir, const glm::vec3 &up);
    	Camera(const glm::vec3 &pos, const glm::vec3 &dir, const glm::vec3 &up, uint32_t f);
    	
	void print();
    };

    Camera interpolate(Camera &a, Camera &b, glm::vec2 range, uint32_t f);
    
    void jsonFromFile(const char* name, nlohmann_loader::json &j);
    void writeSampleJsonFile(std::string meta_file_name);
    
    
    // old data structure
    
    struct AniObjWidget{
    	// input data
    	nlohmann_loader::json config;
    	std::string file_name;		// path to input raw file
    	std::string type_name;		// unstructured grid for now
    	glm::vec3 dims;			// x, y, z
	glm::vec3 world_bbox;
	std::string bgmap_name;		// path to background map
    	
    	// animation widget settings
    	glm::vec2 frameRange;				// the range of animating this object
    	std::vector<Camera> cameras; 			// list of camera keyframes
    	glm::vec2 tfRange;				// TF range 
    	
    	// init
    	AniObjWidget(){};
    	AniObjWidget(const nlohmann_loader::json in_file);
	AniObjWidget(std::string type_name, int x, int y, int z, std::vector<float> &z_m);
	void init();
	void init_from_json(const nlohmann_loader::json in_file);
	void init_from_json_modular(const nlohmann_loader::json in_file);
    	void load_info();
    	void load_info_modular();
    	void print_info();
    	void load_cameras();
    	void load_tfs();
    	void overwrite_data_info(std::string file_name, glm::vec3 dims);
    	
    	// animation
    	void getFrameCam(Camera &cam) const {cam = currentCam;}
    	void getFrameTF (std::vector<float> &c, std::vector<float> &o) const {c = colors; o = opacities;} 
    	void advanceFrame();
    	
    	private:
    	uint32_t currentF = 0;
    	Camera currentCam;
    	std::vector<float> colors;
    	std::vector<float> opacities;
    	
    };
    
       
    struct AniObjHandler{
    	// input data
    	bool is_header = false;
    	nlohmann_loader::json header_config;
    	std::vector<AniObjWidget> widgets;
    	
    	// init
    	AniObjHandler(){};
    	AniObjHandler(const char* filename);

	void init(const char* filename);
	void init_modular(const char* filename);
    };
	
    
    struct AnimatorKF{
	// init
    	AnimatorKF(){};
    	AnimatorKF(const char* filename, const char* data_file_name);
	void init(const char* filename, const char* data_file_name);
	void init_from_json(nlohmann_loader::json in_file, nlohmann_loader::json &in_file_data);
	
	// get params
	glm::vec3 get_bbox() const {return get_vec3f(config_json["world_bbox"]);}
	glm::vec3 get_max_dims() const {return max_dims;}
	glm::vec2 get_fRange() const {return get_vec2i(config_json["frameRange"]);}
	std::string get_scalar_field() const {
		return config_json.contains("scalar_field") ? config_json["scalar_field"].get<std::string>() : "salinity";
	}
	uint32_t get_data_list_size() const;	
	void print_info();
	
	// read data and render info as json
	nlohmann_loader::json get_scene_data(uint32_t idx, nlohmann_loader::json &data_list_json);
	nlohmann_loader::json get_scene_info(uint32_t idx);
	std::vector<nlohmann_loader::json> get_scene_data_list() const;

	// animation
	void advanceFrame(){currentF++;}
	void get_current_cam(Camera &cam);

	// Return the raw camera JSON entries for this keyframe (as written in the script)
	std::vector<nlohmann_loader::json> get_camera_json() const;



	private:
	// input data
    	nlohmann_loader::json config_json;
    	
    	uint32_t currentF = 0;
    	// set max dims for this scene to fit in the bounding box 
    	glm::vec3 max_dims = {0,0,0};
    	void init_max_dims(nlohmann_loader::json &data_list_json);
    	
    };

    struct Animator{
    	// key frames
	std::vector<AnimatorKF> kfs;
	
	// init
    	Animator(){};
    	Animator(const char* filename);
	void init(const char* filename);

	// Return global metadata read from the data list JSON (if present)
	nlohmann_loader::json get_global_metadata() const;
	
	// read kf pieces
	bool isHeader(){return is_header;}
	nlohmann_loader::json get_scene_data(uint32_t i, uint32_t j){
	    return kfs[i].get_scene_data(j, data_list_json);
	}
	nlohmann_loader::json get_scene_info(uint32_t i, uint32_t j){
	    return kfs[i].get_scene_info(j);
	}
	
	private:
	// input data
    	bool is_header;
    	nlohmann_loader::json header_json;
	nlohmann_loader::json data_list_json;
	
	
    };
}


