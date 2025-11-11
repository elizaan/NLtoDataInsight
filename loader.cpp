#include "loader.h"
#include <chrono>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "json.hpp"

//#include "stb_image.h"
//#include "util.h"
#include "ext/glm/gtx/string_cast.hpp"

using json = nlohmann_loader::json;

visuser::Camera::Camera(): pos(glm::vec3(1,0,0)), dir(glm::vec3(-1,0,0)), up(glm::vec3(0,0,1)){
    frame = 0;
    // preserve legacy semantics: dir historically held either a focalPoint or a direction
    focalPoint = dir;
    // compute dir_unit robustly
    glm::vec3 diff = focalPoint - pos;
    if (glm::length(diff) > 1e-6f) dir_unit = glm::normalize(diff);
    else if (glm::length(dir) > 1e-6f) dir_unit = glm::normalize(dir);
    else dir_unit = glm::vec3(0.0f, 0.0f, -1.0f);
}

visuser::Camera::Camera(const glm::vec3 &pos, const glm::vec3 &dir, const glm::vec3 &up)
    : pos(pos), dir(dir), up(up){
    frame = 0;
    // Keep legacy behavior: store provided 'dir' value as focalPoint by default
    focalPoint = dir;
    glm::vec3 diff = focalPoint - pos;
    if (glm::length(diff) > 1e-6f) dir_unit = glm::normalize(diff);
    else if (glm::length(dir) > 1e-6f) dir_unit = glm::normalize(dir);
    else dir_unit = glm::vec3(0.0f, 0.0f, -1.0f);
}

visuser::Camera::Camera(const glm::vec3 &pos, const glm::vec3 &dir, const glm::vec3 &up, uint32_t f)
    : pos(pos), dir(dir), up(up), frame(f){
    // Keep legacy behavior: store provided 'dir' value as focalPoint by default
    focalPoint = dir;
    glm::vec3 diff = focalPoint - pos;
    if (glm::length(diff) > 1e-6f) dir_unit = glm::normalize(diff);
    else if (glm::length(dir) > 1e-6f) dir_unit = glm::normalize(dir);
    else dir_unit = glm::vec3(0.0f, 0.0f, -1.0f);
}

    
void visuser::Camera::print(){
	std::cout << "Camera: \n pos: [" <<glm::to_string(this->pos) << "], "
		  << "\n dir: [" <<glm::to_string(this->dir) << "], "
		  << "\n up : [" <<glm::to_string(this->up) << "]\n";
	
}

visuser::Camera visuser::interpolate(visuser::Camera &a, visuser::Camera &b, glm::vec2 range, uint32_t f){
	float val = (f - range[0]) / float(range[1] - range[0]);
	glm::vec3 pos = mix(a.pos, b.pos, val);
	glm::vec3 dir = glm::normalize(mix(a.dir, b.dir, val));
	glm::vec3 up  = glm::normalize(mix(a.up, b.up, val));
	return visuser::Camera(pos, dir, up);
}
    
void visuser::jsonFromFile(const char* name, nlohmann_loader::json &j){
    std::ifstream cfg_file(name);
    // std::cout << "\nReading " << name <<"\n";
    if (!cfg_file) {
        std::cerr << "[error]: Failed to open config file " << name << "\n";
        throw std::runtime_error("Failed to open input config file");
    }
    cfg_file >> j;
    //std::cout << "Done reading " << name <<"\n\n";
}
    
    
visuser::AniObjWidget::AniObjWidget(const nlohmann_loader::json in_file){
    config = in_file;
}


void visuser::AniObjWidget::init(){
    load_info(); 
    load_cameras(); 
    load_tfs();
}


void visuser::AniObjWidget::init_from_json(const nlohmann_loader::json in_file){
    config = in_file;
    load_info();
    load_cameras();
    load_tfs();
}

void visuser::AniObjWidget::init_from_json_modular(const nlohmann_loader::json in_file){
    config = in_file;
    load_info();
    load_cameras();
    load_tfs();
}

void visuser::AniObjWidget::load_info(){
    file_name 	= config["data"]["name"];
    type_name 	= config["data"]["type"];
    world_bbox  = get_vec3f(config["data"]["world_bbox"]);
    dims 	= get_vec3i(config["data"]["dims"]);
    frameRange 	= get_vec2i(config["data"]["frameRange"]);
    currentF 	= frameRange[0];
    
    if (config["data"].contains("backgroundMap")) {
    	bgmap_name = config["data"]["backgroundMap"];
    }else bgmap_name = "";
	
    std::cout << "end load info\n";
}

void visuser::AniObjWidget::load_info_modular(){
    type_name 	= config["data"]["type"];
    file_name 	= config["data"]["name"];
    type_name 	= config["data"]["type"];
    world_bbox  = get_vec3f(config["data"]["world_bbox"]);
    dims 	= get_vec3i(config["data"]["dims"]);
    frameRange 	= get_vec2i(config["data"]["frameRange"]);
    currentF 	= frameRange[0];
    
    if (config["data"].contains("backgroundMap")) {
    	bgmap_name = config["data"]["backgroundMap"];
    }else bgmap_name = "";
	
    std::cout << "end load info\n";
}

void visuser::AniObjWidget::print_info(){
    std::cout << "data info...\n\n"
	      << "input file: " << file_name << "\n"
	      << "type: " << type_name <<"\n"
	      << "dims: " << dims[0] <<" "<< dims[1] <<" "<<dims[2]<<"\n"
	      << "output frame range: " << frameRange[0] <<" "<<frameRange[1]<<"\n";
    std::cout << "]\n\nEnd data log...\n\n";
}

void visuser::AniObjWidget::load_cameras(){
    const std::vector<json> &camera_set = config["camera"].get<std::vector<json>>();
    for (size_t i = 0; i < camera_set.size(); ++i) {
        const auto &c = camera_set[i];
        
        // Support both old format (pos/dir/up) and new format (position/focalPoint or direction/up)
        glm::vec3 pos, dir, up;
        
        // Read position
        if (c.contains("position")) {
            pos = get_vec3f(c["position"]);
        } else if (c.contains("pos")) {
            pos = get_vec3f(c["pos"]);
        }
        
        // Read direction or focalPoint
        if (c.contains("focalPoint")) {
            // VTK format uses focalPoint
            dir = get_vec3f(c["focalPoint"]);
        } else if (c.contains("direction")) {
            // OSPRay format uses direction
            dir = get_vec3f(c["direction"]);
        } else if (c.contains("dir")) {
            // Old format
            dir = get_vec3f(c["dir"]);
        }
        
        // Read up vector
        if (c.contains("up")) {
            up = get_vec3f(c["up"]);
        }
        
        cameras.push_back(Camera(pos, dir, up, c["frame"].get<uint32_t>()));
    }
    currentCam = cameras[0];
    std::cout << "end load cames\n";
}

void visuser::AniObjWidget::load_tfs(){
    const std::vector<json> &tf_set = config["transferFunc"].get<std::vector<json>>();
    colors = tf_set[0]["colors"].get<std::vector<float>>();
    opacities = tf_set[0]["opacities"].get<std::vector<float>>();
    tfRange = get_vec2f(tf_set[0]["range"]);
    std::cout << "end load tfs\n";
}

void visuser::AniObjWidget::overwrite_data_info(std::string f_name, glm::vec3 d){
    file_name = f_name;
    dims = d;
}

void visuser::AniObjWidget::advanceFrame(){
    currentCam = interpolate(cameras[0], cameras[1], frameRange, currentF);
	
    // do nothing for transfer function now
	
    currentF++;
}


// object handler

visuser::AniObjHandler::AniObjHandler(const char* filename){
    init(filename);
}

void visuser::AniObjHandler::init(const char* filename){
    jsonFromFile(filename, header_config);
    is_header = header_config["isheader"];
    
    if (!is_header){
    	// not a header, read again as plain kf file
    	widgets.resize(1);
    	widgets[0].init_from_json(header_config);
    }else{
    	// is a header, read all kf files
    	auto filenames = header_config["file_list"].get<std::vector<json>>();
    	auto datanames = header_config["data_list"].get<std::vector<json>>();
	std::filesystem::path p = std::filesystem::absolute(filename).parent_path();
	std::string p_str = p.generic_string() + "/";
	// std::cout << "path: "<< p_str <<"\n";
    	widgets.resize(filenames.size());
    	for (size_t i=0; i<filenames.size(); i++){
	    nlohmann_loader::json config;
	    std::string kf_name = filenames[i]["keyframe"];
	    jsonFromFile((p_str+kf_name).c_str(), config);
	    widgets[i].init_from_json(config);
	    if (!filenames[i]["data_i"].is_null()){
		uint32_t data_i = filenames[i]["data_i"];
		if (datanames.size() > data_i){ 
		    widgets[i].overwrite_data_info(datanames[data_i]["name"], 
						   get_vec3i(datanames[data_i]["dims"]));
		    std::cout << "overwriting " << kf_name 
			      << " to \n  data: " << widgets[i].file_name
			      << " \n  dims: " << glm::to_string(widgets[i].dims)
			      << "\n";
		}	
	    }
    	}
    }
}

void visuser::AniObjHandler::init_modular(const char* filename){
    jsonFromFile(filename, header_config);
    is_header = header_config["isheader"];
    
    if (!is_header){
    	// not a header, read again as plain kf file
    	widgets.resize(1);
    	widgets[0].init_from_json(header_config);
    }else{
    	// is a header, read all kf files
    	// parse current path first
    	std::filesystem::path p = std::filesystem::absolute(filename).parent_path();
	std::string p_str = p.generic_string() + "/";
	// std::cout << "path: "<< p_str <<"\n";
	
    	// read data list and kf list
    	nlohmann_loader::json data_list_config;
    	std::string data_list_name = header_config["data_list"];
    	jsonFromFile((p_str+data_list_name).c_str(), data_list_config);
    	auto filenames = header_config["kf_list"].get<std::vector<json>>();
    	auto datanames = data_list_config["list"].get<std::vector<json>>();
    	widgets.resize(filenames.size());
    	
    	// load all kfs
		std::cout << "Logging rendering information..." << std::endl;
    	for (size_t i=0; i<filenames.size(); i++){
	    nlohmann_loader::json config;
	    std::string kf_name = filenames[i];
	    jsonFromFile((p_str+kf_name).c_str(), config);
	    
		

	    std::cout << "key frame " << kf_name << " with data [ ";
	    auto scene_data_list = config["scene_data_list"].get<std::vector<json>>();
	    for (size_t idx=0; idx < scene_data_list.size(); idx++){
	    	std::cout << scene_data_list[idx]["index_in_list"] <<" ";
	    }
	    std::cout << "]\n";
	    //widgets[i].init_from_json_modular(config);
	    /*if (!filenames[i]["data_i"].is_null()){
		uint32_t data_i = filenames[i]["data_i"];
		if (datanames.size() > data_i){ 
		    widgets[i].overwrite_data_info(datanames[data_i]["src"]["name"], 
						   get_vec3i(datanames[data_i]["src"]["dims"]));
		    std::cout << "overwriting " << kf_name 
			      << " to \n  data: " << widgets[i].file_name
			      << " \n  dims: " << glm::to_string(widgets[i].dims)
			      << "\n";
		}	
	    }*/
    	}
    }
}


void visuser::writeSampleJsonFile(std::string meta_file_name){
    std::vector<uint32_t> data_i_list_kf = {0, 0, 1};
    std::map<uint32_t, uint32_t > data_i_list;
    nlohmann_loader::ordered_json j;
    std::string base_file_name = meta_file_name+"_kf";
    
    j["isheader"] = true;
    
    for (auto i: data_i_list_kf){
	if (data_i_list.find(i) == data_i_list.end()){
	    uint32_t idx = data_i_list.size();
	    j["data_list"][idx]["name"] = "<file "+std::to_string(idx)+">";
	    j["data_list"][idx]["dims"] = {(idx+1)*100, (idx+1)*100, (idx+1)*100};
	    data_i_list[i] = idx;
	}
    }
	
    // export all key frames to json file
    // write a header of file names 
    for (size_t i=0; i<data_i_list_kf.size();i++){
	std::string file_name = base_file_name + std::to_string(i) + ".json";
	j["file_list"][i]["keyframe"] = file_name;
	j["file_list"][i]["data_i"] = data_i_list[data_i_list_kf[i]];

	// write json for each keyframe interval
	nlohmann_loader::ordered_json tmp_j;
	tmp_j["isheader"] = false;
	tmp_j["data"]["type"] = "structured";
	tmp_j["data"]["name"] = "";
	tmp_j["data"]["dims"] = {100, 100, 100};
	tmp_j["data"]["world_bbox"] = {10, 10, 10};
	tmp_j["data"]["frameRange"] = {i*5, (i+1)*5};

	// cameras
	for (size_t j=0; j<2; j++)
	    {
		nlohmann_loader::ordered_json tmp_cam;
		tmp_cam["frame"] = (i+j)*5;
		for (size_t c=0; c<3; c++){
		    tmp_cam["pos"].push_back(c);
		    tmp_cam["dir"].push_back(c);
		    tmp_cam["up"].push_back(c);
		}
		tmp_j["camera"].push_back(tmp_cam);
	    }

	// tf
	tmp_j["transferFunc"][0]["frame"] = i*5;
	tmp_j["transferFunc"][0]["range"] = {-1, 1};
        tmp_j["transferFunc"][0]["colors"].push_back(0);
	tmp_j["transferFunc"][0]["colors"].push_back(0);
	tmp_j["transferFunc"][0]["colors"].push_back(255);
        tmp_j["transferFunc"][0]["opacities"].push_back(0);
	tmp_j["transferFunc"][0]["opacities"].push_back(1);
	    
	std::ofstream o(file_name);
	o << std::setw(4)<< tmp_j <<std::endl;
	o.close();
    }
    std::ofstream o_meta(meta_file_name+".json");
    o_meta << std::setw(4) << j <<std::endl;
    o_meta.close();
}


    
visuser::AnimatorKF::AnimatorKF(const char* file_name, const char* data_file_name){
    json data_config;
    jsonFromFile(file_name, config_json);
    jsonFromFile(data_file_name, data_config);
    init_max_dims(data_config);
}


void visuser::AnimatorKF::init(const char* file_name, const char* data_file_name){
    json data_config;
    jsonFromFile(file_name, config_json);
    jsonFromFile(data_file_name, data_config);
    init_max_dims(data_config);
}

void visuser::AnimatorKF::init_from_json(json in_file, json &data_list_json){
    config_json = in_file;
    init_max_dims(data_list_json);
}

void visuser::AnimatorKF::init_max_dims(json &data_list_json){
    if (data_list_json.is_null()) return;
    for (size_t idx=0; idx < get_data_list_size(); idx++){
    	auto data = get_scene_data(idx, data_list_json);
    	if (!data["src"]["dims"].is_null()){
    	    glm::vec3 d = get_vec3f(data["src"]["dims"]);
    	    for (size_t k=0; k<3; k++)
    	    	max_dims[k] = std::max(max_dims[k], d[k]);
    	}
    }
}

void visuser::AnimatorKF::get_current_cam(Camera &cam) {
    std::vector<Camera> cameras;
    auto camera_set = config_json["camera"].get<std::vector<nlohmann_loader::json>>();
    for (size_t i = 0; i < camera_set.size(); ++i) {
        const auto &c = camera_set[i];
        
        // Support both old format (pos/dir/up) and new format (position/focalPoint or direction/up)
        glm::vec3 pos, dir, up;
        
        // Read position
        if (c.contains("position")) {
            pos = get_vec3f(c["position"]);
        } else if (c.contains("pos")) {
            pos = get_vec3f(c["pos"]);
        }
        
        // Read direction or focalPoint
        if (c.contains("focalPoint")) {
            // VTK format uses focalPoint
            dir = get_vec3f(c["focalPoint"]);
        } else if (c.contains("direction")) {
            // OSPRay format uses direction
            dir = get_vec3f(c["direction"]);
        } else if (c.contains("dir")) {
            // Old format
            dir = get_vec3f(c["dir"]);
        }
        
        // Read up vector
        if (c.contains("up")) {
            up = get_vec3f(c["up"]);
        }
        
        cameras.push_back(Camera(pos, dir, up, c["frame"].get<uint32_t>()));
    }
    // Determine whether the source cameras provided position+focalPoint (VTK style) or pos+dir (old style)
    bool has_focal = false;
    if (camera_set.size() >= 2) {
        const auto &c0 = camera_set[0];
        const auto &c1 = camera_set[1];
        if ((c0.contains("position") && c0.contains("focalPoint")) &&
            (c1.contains("position") && c1.contains("focalPoint"))) {
            has_focal = true;
        }
    }

    if (has_focal) {
        // Interpolate position and focalPoint, then compute dir = normalize(focalPoint - position)
        glm::vec3 pos0 = get_vec3f(camera_set[0]["position"]);
        glm::vec3 fp0  = get_vec3f(camera_set[0]["focalPoint"]);
        glm::vec3 pos1 = get_vec3f(camera_set[1]["position"]);
        glm::vec3 fp1  = get_vec3f(camera_set[1]["focalPoint"]);

        float val = (currentF - get_fRange()[0]) / float(get_fRange()[1] - get_fRange()[0]);
        // Compute a safe, clamped interpolation parameter
        float denom = get_fRange()[1] - get_fRange()[0];
        float t = 0.0f;
        if (std::fabs(denom) < 1e-6f) t = 0.0f;
        else t = (currentF - get_fRange()[0]) / denom;
        t = glm::clamp(t, 0.0f, 1.0f);

        glm::vec3 pos = mix(pos0, pos1, t);
        glm::vec3 fp  = mix(fp0, fp1, t);

        // Compute a robust unit direction from eye -> focalPoint
        glm::vec3 dir_unit;
        glm::vec3 diff = fp - pos;
        if (glm::length(diff) > 1e-6f) {
            dir_unit = glm::normalize(diff);
        } else {
            // Fallback: try mixing per-camera position->focal vectors
            glm::vec3 raw0 = fp0 - pos0;
            glm::vec3 raw1 = fp1 - pos1;
            glm::vec3 raw = mix(raw0, raw1, t);
            if (glm::length(raw) > 1e-6f) dir_unit = glm::normalize(raw);
            else if (glm::length(fp) > 1e-6f) dir_unit = glm::normalize(fp);
            else dir_unit = glm::vec3(0.0f, 0.0f, -1.0f);
        }

        // Interpolate up and then orthonormalize it relative to dir_unit
        glm::vec3 up  = glm::vec3(0.0f, 1.0f, 0.0f);
        if (camera_set[0].contains("up") && camera_set[1].contains("up")) {
            glm::vec3 up0 = get_vec3f(camera_set[0]["up"]);
            glm::vec3 up1 = get_vec3f(camera_set[1]["up"]);
            up = glm::normalize(mix(up0, up1, t));
        }
        if (glm::length(dir_unit) > 1e-6f) {
            up = up - dir_unit * glm::dot(dir_unit, up);
            if (glm::length(up) < 1e-6f) up = glm::vec3(0.0f, 1.0f, 0.0f);
            else up = glm::normalize(up);
        }

        // Preserve legacy behavior: store focalPoint in cam.dir (so VTK consumers get SetFocalPoint)
        cam = Camera(pos, fp, up);
        // Also explicitly keep the new fields current for convenience
        cam.focalPoint = fp;
        cam.dir_unit = dir_unit;
        std::cout << "[DEBUG] get_current_cam: used position+focalPoint interpolation.\n";
    } else {
        cam = interpolate(cameras[0], cameras[1], get_fRange(), currentF);
        std::cout << "[DEBUG] get_current_cam: used pos/dir interpolation (legacy).\n";
    }
}

std::vector<nlohmann_loader::json> visuser::AnimatorKF::get_camera_json() const {
    std::vector<nlohmann_loader::json> cameras;
    if (config_json.contains("camera")) {
        try {
            cameras = config_json["camera"].get<std::vector<nlohmann_loader::json>>();
        } catch (...) {
            // ignore parse errors
        }
    }
    return cameras;
}

void visuser::AnimatorKF::print_info(){
    std::cout << "\nkf basic info...\n"
	      << "bbox: " << get_bbox()[0] <<" "<< get_bbox()[1] <<" "<<get_bbox()[2]<<"\n"
	      << "max dims: " << get_max_dims()[0] <<" "<< get_max_dims()[1] <<" "<<get_max_dims()[2]<<"\n"
	      << "output frame range: [" << get_fRange()[0] <<" "<<get_fRange()[1]<<"]\n";
    std::cout << "Cams: \n";
    auto camera_set = config_json["camera"].get<std::vector<nlohmann_loader::json>>();
    for (size_t i = 0; i < camera_set.size(); ++i) {
	const auto &c = camera_set[i];
	std::cout << c["pos"] <<" "<<c["dir"] <<" "<< c["up"]<< std::endl;
    }
    std::cout << "End kf basic info...\n\n";
}

uint32_t visuser::AnimatorKF::get_data_list_size() const{
    return (config_json["scene_data_list"].get<std::vector<nlohmann_loader::json>>()).size();
}

std::vector<nlohmann_loader::json> visuser::AnimatorKF::get_scene_data_list() const{
	return config_json["scene_data_list"].get<std::vector<nlohmann_loader::json>>();
}

json visuser::AnimatorKF::get_scene_data(uint32_t idx, json &data_list_json){
    return data_list_json["list"].get<std::vector<json>>()[config_json["scene_data_list"].get<std::vector<nlohmann_loader::json>>()[idx]["index_in_list"]];
}

json visuser::AnimatorKF::get_scene_info(uint32_t idx){
    return config_json["scene_data_list"].get<std::vector<json>>()[idx]["scene_info"];
}



visuser::Animator::Animator(const char* file_name){
    jsonFromFile(file_name, header_json);
    jsonFromFile(to_string(header_json["data_list"]).c_str(), data_list_json);
}


void visuser::Animator::init(const char* file_name){
    jsonFromFile(file_name, header_json);
    is_header = header_json["isheader"];
    
    if (!is_header){
    	// not a header, read again as plain kf file
    	kfs.resize(1);
    	kfs[0].init_from_json(header_json, data_list_json);
    }else{
    	// is a header, read all kf files
    	// parse current path first
    	std::filesystem::path p = std::filesystem::absolute(file_name).parent_path();
	std::string p_str = p.generic_string() + "/";
	// std::cout << "path: "<< p_str <<"\n";
	
    	// read data list and kf list
    	std::string data_list_name = header_json["data_list"];
    	jsonFromFile((p_str+data_list_name).c_str(), data_list_json);
    	auto filenames = header_json["kf_list"].get<std::vector<json>>();
    	auto datanames = data_list_json["list"].get<std::vector<json>>();
    	kfs.resize(filenames.size());
    	
		std::cout << "Logging rendering information..." << std::endl;
    	// load all kfs
    	for (size_t i=0; i<filenames.size(); i++){
	    nlohmann_loader::json config;
	    std::string kf_name = filenames[i];
	    jsonFromFile((p_str+kf_name).c_str(), config);
	    
	    
		std::cout << "key frame " << kf_name << " with data [ ";
	    auto scene_data_list = config["scene_data_list"].get<std::vector<json>>();
	    for (size_t idx=0; idx < scene_data_list.size(); idx++){
	    	std::cout << scene_data_list[idx]["index_in_list"] <<" ";
	    }
	    std::cout << "]\n";
	    kfs[i].init_from_json(config, data_list_json);
    	}
    }
}

nlohmann_loader::json visuser::Animator::get_global_metadata() const {
    if (data_list_json.is_null()) return nlohmann_loader::json();
    if (data_list_json.contains("global_metadata")) return data_list_json["global_metadata"];
    return nlohmann_loader::json();
}










