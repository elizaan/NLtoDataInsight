#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../../ext/pybind11_json/pybind11_json.hpp"
#include <iostream>

namespace py = pybind11;

void echo(int i) {
    std::cout << i <<std::endl;
}

template<class T>
std::vector<T>makeVectorFromPyArray( py::array_t<T>py_array )
{
    return std::vector<T>(py_array.data(), py_array.data() + py_array.size());
}


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

#include "../../loader.h"

using json = nlohmann_loader::json;
using namespace visuser;


nlohmann_loader::json generateScript(){
    nlohmann_loader::json j = {{"value", 1}};

    std::cout << "This function returns an nlohmann_loader::json instance: "  << j << std::endl;

    return j;
}

nlohmann_loader::json fixedCamHelper(std::string meta_file_name,
			      std::vector<std::string> &filenames,
			      int kf_interval,
			      std::vector<int> dims,
			      std::string meshType,
			      int world_bbx_len,
			      std::vector<float> &cam,
			      std::vector<float> &tf_range,
			      std::string bgImg)

{
    nlohmann_loader::ordered_json j;
    // std::string base_file_name = meta_file_name+"_kf";
    // std::filesystem::path p = std::filesystem::current_path();
    // std::string p_str = p.generic_string() + "Out_text"+"/";
	// std::cout << "DEBUG: p_str = " << p_str << std::endl; //
	std::filesystem::path meta_path(meta_file_name);
    std::string base_name = meta_path.stem().generic_string(); // "abcd" if input is "abcd.json"

    // Get the output directory (e.g., "GAD_text/")
    std::string gad_dir = meta_path.parent_path().generic_string() + "/";

    // Path to Out_text/ (sibling of GAD_text/)
    std::string out_text_dir = meta_path.parent_path().parent_path().generic_string() + "/Out_text/";

    j["isheader"] = true;
    // j["data_list"] = meta_file_name+"_data_list.json";
	j["data_list"] = base_name + "_data_list.json";
    {
    	nlohmann_loader::ordered_json data_j;
    	for (int i=0; i<filenames.size();i++){
			data_j["list"][i]["type"] = meshType;
			// data_j["list"][i]["src"]["name"] = p_str+filenames[i];
			data_j["list"][i]["src"]["name"] = out_text_dir + filenames[i];
			data_j["list"][i]["src"]["dims"] = {dims[0], dims[1], dims[2]};
    	}
    	// std::ofstream o(j["data_list"]);
		std::string data_list_path = gad_dir + base_name + "_data_list.json";
        std::ofstream o(data_list_path);
		o << std::setw(4)<< data_j <<std::endl;
		o.close();
    }
       
    // export all key frames to json file
    // write a header of file names 
    for (size_t i=0; i<filenames.size();i++){
		// std::string file_name = base_file_name + std::to_string(i) + ".json";
		// j["kf_list"][i] = file_name;
		std::string file_name = base_name + "_kf" + std::to_string(i) + ".json";
		j["kf_list"][i] = file_name;

		std::string kf_path = gad_dir + file_name;

		// write json for each keyframe interval
		nlohmann_loader::ordered_json tmp_j;
		tmp_j["isheader"] = false;
		tmp_j["world_bbox"] = {world_bbx_len, world_bbx_len, world_bbx_len};
		tmp_j["frameRange"] = {i*kf_interval, (i+1)*kf_interval};
		if (bgImg != "") tmp_j["data"]["backgroundMap"] = bgImg;

		// cameras
		for (size_t j=0; j<2; j++){
			nlohmann_loader::ordered_json tmp_cam;
			tmp_cam["frame"] = (i+j)*kf_interval;
			for (size_t c=0; c<3; c++){
				tmp_cam["pos"].push_back(cam[c]);
				tmp_cam["dir"].push_back(cam[3+c]);
				tmp_cam["up"].push_back(cam[6+c]);
			}
			tmp_j["camera"].push_back(tmp_cam);
		}
	
		// one data per kf in this template
		// tf
		tmp_j["scene_data_list"][0]["index_in_list"] = i;
		tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["frame"] = i*kf_interval;
		tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["range"] = {tf_range[0], tf_range[1]};
         
		// Grayscale colormap
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {
		// 	0.0, 0.0, 0.0,   // Black
		// 	0.2, 0.2, 0.2,   // Dark gray
		// 	0.4, 0.4, 0.4,   // Medium dark gray
		// 	0.5, 0.5, 0.5,   // Medium gray
		// 	0.8, 0.8, 0.8,   // Light gray
		// 	1.0, 1.0, 1.0    // White
		// };
		// // Adjust opacity points to match the number of color points (6 instead of 7)
		// for (int i=0; i<6; i++)
		// 	tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(i/5.0);
		
		// Set opacity points - transparent land, visible ocean
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"] = {
		// 	0.0,    // Completely transparent for land (values near 0)
		// 	0.0,    // Still transparent at around 30
		// 	0.7,    // Good opacity for lower salinity ocean (~35)
		// 	0.85,   // Higher opacity for mid salinity
		// 	1.0     // Full opacity for higher salinity (~38)
		// };

		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {
		// 	0.0, 0.0, 0.0,   // Black 
		// 	0.1, 0.1, 0.1,  // Medium dark gray
		// 	1.0, 1.0, 1.0    // White
		// };
		// // Adjust opacity points to match the number of color points (6 instead of 7)
		
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.008f);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1.0f);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1.0f);
		
		// fixed tf for now
        tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {0, 0, 0.562493,  0, 0, 1,  0, 1, 1,  0.500008, 1, 0.500008,  1, 1, 0,  1, 0, 0,  0.500008, 0, 0};
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.1);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.7);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.85);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.2);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.1);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.45);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.4);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.95);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.1);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.2);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.5);

		for (int i=0; i<7; i++)
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(i/float(7));

		// 	// for (int i=0; i<7; i++)
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1);

		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.05);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.2);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.7);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.8);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.05);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.02);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.4);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.35);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.85);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.93);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.01);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.07);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.9);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1);

		// Assuming tmp_j is a json object (like nlohmann::json)
		// and the structure already exists

		// std::vector<float> opacities = {
		// 	0.0, 0.007843137718737125, 0.01568627543747425, 0.0235294122248888,
		// 	0.0313725508749485, 0.03921568766236305, 0.0470588244497776,
		// 	0.054901961237192154, 0.062745101749897, 0.07058823853731155,
		// 	0.0784313753247261, 0.08627451211214066, 0.0941176488995552,
		// 	0.10196078568696976, 0.10980392247438431, 0.11764705926179886,
		// 	0.125490203499794, 0.13333334028720856, 0.1411764770746231,
		// 	0.14901961386203766, 0.1568627506494522, 0.16470588743686676,
		// 	0.1725490242242813, 0.18039216101169586, 0.1882352977991104,
		// 	0.19607843458652496, 0.20392157137393951, 0.21176470816135406,
		// 	0.21960784494876862, 0.22745098173618317, 0.23529411852359772,
		// 	0.24313725531101227, 0.250980406999588, 0.25882354378700256,
		// 	0.2666666805744171, 0.27450981736183167, 0.2823529541492462,
		// 	0.29019609093666077, 0.2980392277240753, 0.30588236451148987,
		// 	0.3137255012989044, 0.32156863808631897, 0.3294117748737335,
		// 	0.33725491166114807, 0.3450980484485626, 0.3529411852359772,
		// 	0.3607843220233917, 0.3686274588108063, 0.3764705955982208,
		// 	0.3843137323856354, 0.3921568691730499, 0.4000000059604645,
		// 	0.40784314274787903, 0.4156862795352936, 0.42352941632270813,
		// 	0.4313725531101227, 0.43921568989753723, 0.4470588266849518,
		// 	0.45490196347236633, 0.4627451002597809, 0.47058823704719543,
		// 	0.47843137383461, 0.48627451062202454, 0.4941176474094391,
		// 	0.501960813999176, 0.5098039507865906, 0.5176470875740051,
		// 	0.5254902243614197, 0.5333333611488342, 0.5411764979362488,
		// 	0.5490196347236633, 0.5568627715110779, 0.5647059082984924,
		// 	0.572549045085907, 0.5803921818733215, 0.5882353186607361,
		// 	0.5960784554481506, 0.6039215922355652, 0.6117647290229797,
		// 	0.6196078658103943, 0.6274510025978088, 0.6352941393852234,
		// 	0.6431372761726379, 0.6509804129600525, 0.658823549747467,
		// 	0.6666666865348816, 0.6745098233222961, 0.6823529601097107,
		// 	0.6901960968971252, 0.6980392336845398, 0.7058823704719543,
		// 	0.7137255072593689, 0.7215686440467834, 0.729411780834198,
		// 	0.7372549176216125, 0.7450980544090271, 0.7529411911964417,
		// 	0.7607843279838562, 0.7686274647712708, 0.7764706015586853,
		// 	0.7843137383460999, 0.7921568751335144, 0.800000011920929,
		// 	0.8078431487083435, 0.8156862854957581, 0.8235294222831726,
		// 	0.8313725590705872, 0.8392156958580017, 0.8470588326454163,
		// 	0.8549019694328308, 0.8627451062202454, 0.8705882430076599,
		// 	0.8784313797950745, 0.886274516582489, 0.8941176533699036,
		// 	0.9019607901573181, 0.9098039269447327, 0.9176470637321472,
		// 	0.9254902005195618, 0.9333333373069763, 0.9411764740943909,
		// 	0.9490196108818054, 0.95686274766922, 0.9647058844566345,
		// 	0.9725490212440491, 0.9803921580314636, 0.9882352948188782
		// };

		// // Add all opacity values
		// for (const float opacity : opacities) {
		// 	tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(opacity);
		// }
	
		
		// Colors - 7 colors from blue to red, with smooth transition
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {
		// 	0, 0, 0.562493,     // Dark blue
		// 	0, 0, 1,       // Blue
		// 	0, 1, 1,       // Cyan
		// 	0.5, 1, 0.5,       // Green
		// 	1, 1, 0,       // Yellow
		// 	1, 0.5, 0,     // Orange
		// 	1, 0, 0        // Red
		// };

		// Opacities - 7 values with gradual increase, matching color count
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].clear(); // Clear existing values
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0);  // Low values
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.2);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.45);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.6);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.75);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(0.85);
		// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(1);  // High values
			// tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {
		// 	0.0, 0.0, 0.0,      // Black for land (values near 0)
		// 	0.0, 0.0, 0.0,      // Black still at around 30 (threshold between land and ocean)
		// 	0.5, 0.5, 0.5,      // Medium gray for lower salinity (~35)
		// 	0.7, 0.7, 0.7,      // Light gray for mid salinity (~36.5)
		// 	1.0, 1.0, 1.0       // Pure white for higher salinity (~38)
		// };
		// std::ofstream o(file_name);
		std::ofstream o(kf_path);
		o << std::setw(4)<< tmp_j <<std::endl;
		o.close();
    }
    std::ofstream o_meta(meta_file_name+".json");
    o_meta << std::setw(4) << j <<std::endl;
    o_meta.close();

    return j;
}

nlohmann_loader::json fixedCamHelperStreamline(std::string meta_file_name,
			      std::vector<std::string> &filenames,
			      int kf_interval,
			      std::vector<int> dims,
			      std::string meshType,
			      int world_bbx_len,
			      std::vector<float> &cam,
			      std::vector<float> &tf_range,
			      std::vector<float> &tf_colors,
			      std::vector<float> &tf_opacities,
			      std::string scalar_field,
			      std::string bgImg,
			      float frame_rate,
			      std::vector<std::string> &required_modules,
			      std::vector<float> &file_sizes_mb,
			      std::string grid_type,
			      std::vector<float> &spacing,
			      std::vector<float> &origin,
			      float view_angle,
			      std::string rendering_backend,
			      nlohmann_loader::json &volume_config,
			      nlohmann_loader::json &streamline_config,
			      nlohmann_loader::json &isosurface_config)

{
    nlohmann_loader::ordered_json j;

	std::filesystem::path meta_path(meta_file_name);
    std::string base_name = meta_path.stem().generic_string();

    // Get the output directory (e.g., "GAD_text/")
    std::string gad_dir = meta_path.parent_path().generic_string() + "/";

    // Path to Out_text/ (sibling of GAD_text/)
    std::string out_text_dir = meta_path.parent_path().parent_path().generic_string() + "/Out_text/";

    j["isheader"] = true;
	j["data_list"] = base_name + "_data_list.json";
    
    // Create data_list.json
    {
    	nlohmann_loader::ordered_json data_j;
    	for (size_t i=0; i<filenames.size(); i++){
            data_j["list"][i]["type"] = meshType;
            // data_j["list"][i]["src"]["name"] = out_text_dir + filenames[i];
			data_j["list"][i]["src"]["name"] = filenames[i];
            data_j["list"][i]["src"]["dims"] = {dims[0], dims[1], dims[2]};
            
            // Extract timestep from filename (e.g., "ocean_290_198_23_t2184.vtk" -> 2184)
            std::string filename = filenames[i];
            size_t t_pos = filename.find("_t");
            if (t_pos != std::string::npos) {
                size_t num_start = t_pos + 2;
                size_t num_end = filename.find('.', num_start);
                if (num_end == std::string::npos) {
                    num_end = filename.length();
                }
                std::string timestep_str = filename.substr(num_start, num_end - num_start);
                try {
                    int timestep = std::stoi(timestep_str);
                    data_j["list"][i]["time"] = timestep;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not extract timestep from filename: " << filename << std::endl;
                }
            }
            
            // Add file size metadata if available
            if (i < file_sizes_mb.size()) {
                data_j["list"][i]["metadata"]["file_size_mb"] = file_sizes_mb[i];
            }
    	}
    	
    	// Add global metadata
    	data_j["global_metadata"]["grid_type"] = grid_type;
    	data_j["global_metadata"]["coordinate_system"] = "cartesian";
    	
    	if (spacing.size() == 3) {
    	    data_j["global_metadata"]["spacing"] = {spacing[0], spacing[1], spacing[2]};
    	} else {
    	    data_j["global_metadata"]["spacing"] = {1.0, 1.0, 1.0};
    	}
    	
    	if (origin.size() == 3) {
    	    data_j["global_metadata"]["origin"] = {origin[0], origin[1], origin[2]};
    	} else {
    	    data_j["global_metadata"]["origin"] = {0.0, 0.0, 0.0};
    	}
    	
		std::string data_list_path = gad_dir + base_name + "_data_list.json";
		std::ofstream o(data_list_path);
		o << std::setw(4) << data_j << std::endl;
		o.close();
    }
       
    // Generate keyframe list and individual keyframe files
    for (size_t i=0; i<filenames.size(); i++){
		std::string file_name = base_name + "_kf" + std::to_string(i) + ".json";
		j["kf_list"][i] = file_name;

		std::string kf_path = gad_dir + file_name;
		
		// Write json for each keyframe interval
		nlohmann_loader::ordered_json tmp_j;
		tmp_j["isheader"] = false;
		tmp_j["world_bbox"] = {world_bbx_len, world_bbx_len, world_bbx_len};
		tmp_j["frameRange"] = {i*kf_interval, (i+1)*kf_interval};
        
        // Add background texture if provided
        if (bgImg != "") tmp_j["data"]["backgroundMap"] = bgImg;

		// cameras - format depends on rendering backend
		bool use_vtk_format = (rendering_backend == "vtk");
		
		for (size_t j=0; j<2; j++){
            nlohmann_loader::ordered_json tmp_cam;
            tmp_cam["frame"] = (i+j)*kf_interval;
            
            tmp_cam["position"] = {cam[0], cam[1], cam[2]};
            
            if (use_vtk_format) {
                tmp_cam["focalPoint"] = {cam[3], cam[4], cam[5]};
            } else {
                tmp_cam["direction"] = {cam[3], cam[4], cam[5]};
            }
            
            tmp_cam["up"] = {cam[6], cam[7], cam[8]};
            tmp_cam["viewAngle"] = view_angle;
            
            tmp_j["camera"].push_back(tmp_cam);
		}
	
		// Ensure we have placeholders for three possible representations so
		// we don't emit null entries in the scene_data_list. We'll overwrite
		// each placeholder below depending on whether the representation is
		// enabled. This guarantees an object exists for every index.
		tmp_j["scene_data_list"] = nlohmann_loader::ordered_json::array();
		for (int _si = 0; _si < 3; ++_si) {
			nlohmann_loader::ordered_json ph = nlohmann_loader::ordered_json::object();
			tmp_j["scene_data_list"].push_back(ph);
		}

		// ==================================================================
		// REPRESENTATION 0: VOLUME (Ocean volume rendering)
		// ==================================================================
		if (!volume_config.is_null() && volume_config.contains("enabled") && volume_config["enabled"].get<bool>()) {
		    int idx = 0;
		    tmp_j["scene_data_list"][idx]["index_in_list"] = i;
		    tmp_j["scene_data_list"][idx]["representation"] = "volume";
		    tmp_j["scene_data_list"][idx]["enabled"] = true;
		    tmp_j["scene_data_list"][idx]["field"] = scalar_field;
		    
		    // Transfer function
		    tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"]["frame"] = i*kf_interval;
		    tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"]["range"] = {tf_range[0], tf_range[1]};
		    tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"]["colors"] = tf_colors;
		    tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"]["opacities"] = tf_opacities;
		    
		    // Add volume properties from config
		    if (volume_config.contains("volumeProperties")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["volumeProperties"] = volume_config["volumeProperties"];
		    }
		    if (volume_config.contains("mapperProperties")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["mapperProperties"] = volume_config["mapperProperties"];
		    }
		} else {
			// Explicitly write a disabled placeholder object so we don't leave
			// a null entry in the JSON array (which breaks downstream parsers).
			int idx = 0;
			tmp_j["scene_data_list"][idx]["index_in_list"] = i;
			tmp_j["scene_data_list"][idx]["representation"] = "volume";
			tmp_j["scene_data_list"][idx]["enabled"] = false;
			tmp_j["scene_data_list"][idx]["field"] = scalar_field;
			tmp_j["scene_data_list"][idx]["scene_info"] = nlohmann_loader::ordered_json::object();
		}
		
		// ==================================================================
		// REPRESENTATION 1: STREAMLINE (Velocity flow)
		// ==================================================================
		if (!streamline_config.is_null() && streamline_config.contains("enabled") && streamline_config["enabled"].get<bool>()) {
		    int idx = 1;
		    tmp_j["scene_data_list"][idx]["index_in_list"] = i;
		    tmp_j["scene_data_list"][idx]["representation"] = "streamline";
		    tmp_j["scene_data_list"][idx]["enabled"] = true;
		    tmp_j["scene_data_list"][idx]["field"] = "velocity";  // Streamlines use velocity field
		    
		    // Add streamline-specific properties from config
		    if (streamline_config.contains("integrationProperties")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["integrationProperties"] = streamline_config["integrationProperties"];
		    }
		    if (streamline_config.contains("seedPlane") && streamline_config["seedPlane"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["seedPlane"] = streamline_config["seedPlane"];
		    }
		    if (streamline_config.contains("seedPoints") && streamline_config["seedPoints"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["seedPoints"] = streamline_config["seedPoints"];
		    }
		    if (streamline_config.contains("seedLine") && streamline_config["seedLine"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["seedLine"] = streamline_config["seedLine"];
		    }
		    if (streamline_config.contains("seedRake") && streamline_config["seedRake"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["seedRake"] = streamline_config["seedRake"];
		    }
		    if (streamline_config.contains("streamlineProperties")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["streamlineProperties"] = streamline_config["streamlineProperties"];
		    }
		    if (streamline_config.contains("colorMapping")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["colorMapping"] = streamline_config["colorMapping"];
		    }
		    if (streamline_config.contains("outline") && streamline_config["outline"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["outline"] = streamline_config["outline"];
		    }
		    if (streamline_config.contains("transferFunc")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"] = streamline_config["transferFunc"];
		    }
		} else {
			int idx = 1;
			tmp_j["scene_data_list"][idx]["index_in_list"] = i;
			tmp_j["scene_data_list"][idx]["representation"] = "streamline";
			tmp_j["scene_data_list"][idx]["enabled"] = false;
			tmp_j["scene_data_list"][idx]["field"] = "velocity";
			tmp_j["scene_data_list"][idx]["scene_info"] = nlohmann_loader::ordered_json::object();
		}
		
		// ==================================================================
		// REPRESENTATION 2: ISOSURFACE (Land mask)
		// ==================================================================
		if (!isosurface_config.is_null() && isosurface_config.contains("enabled") && isosurface_config["enabled"].get<bool>()) {
		    int idx = 2;
		    tmp_j["scene_data_list"][idx]["index_in_list"] = i;
		    tmp_j["scene_data_list"][idx]["representation"] = "isosurface";
		    tmp_j["scene_data_list"][idx]["enabled"] = true;
		    tmp_j["scene_data_list"][idx]["field"] = scalar_field;  // Isosurface uses salinity for land detection
		    
		    // Add isosurface properties from config
		    if (isosurface_config.contains("isoMethod")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["isoMethod"] = isosurface_config["isoMethod"];
		    }
		    if (isosurface_config.contains("thresholdRange")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["thresholdRange"] = isosurface_config["thresholdRange"];
		    }
		    if (isosurface_config.contains("isoValues")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["isoValues"] = isosurface_config["isoValues"];
		    }
		    if (isosurface_config.contains("numberOfContours")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["numberOfContours"] = isosurface_config["numberOfContours"];
		    }
		    if (isosurface_config.contains("surfaceProperties")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["surfaceProperties"] = isosurface_config["surfaceProperties"];
		    }
		    if (isosurface_config.contains("texture") && isosurface_config["texture"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["texture"] = isosurface_config["texture"];
		    }
		    if (isosurface_config.contains("colorMapping")) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["colorMapping"] = isosurface_config["colorMapping"];
		    }
		    if (isosurface_config.contains("transferFunc") && isosurface_config["transferFunc"]["enabled"].get<bool>()) {
		        tmp_j["scene_data_list"][idx]["scene_info"]["transferFunc"] = isosurface_config["transferFunc"];
		    }

		} else {
			int idx = 2;
			tmp_j["scene_data_list"][idx]["index_in_list"] = i;
			tmp_j["scene_data_list"][idx]["representation"] = "isosurface";
			tmp_j["scene_data_list"][idx]["enabled"] = false;
			tmp_j["scene_data_list"][idx]["field"] = scalar_field;
			tmp_j["scene_data_list"][idx]["scene_info"] = nlohmann_loader::ordered_json::object();
		}

		std::ofstream o(kf_path);
		o << std::setw(4) << tmp_j << std::endl;
		o.close();
    }
    
    // Add metadata section
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    struct tm *parts = std::localtime(&now_c);
    char date_buffer[11];
    std::strftime(date_buffer, sizeof(date_buffer), "%Y-%m-%d", parts);
    
    j["metadata"]["creation_date"] = std::string(date_buffer);
    j["metadata"]["modification_date"] = std::string(date_buffer);
    j["metadata"]["total_frames"] = static_cast<int>(filenames.size());
    j["metadata"]["frame_rate"] = frame_rate;
    
    // Add rendering backend section
    j["rendering_backend"]["preferred"] = rendering_backend;
    j["rendering_backend"]["version"] = "9.0+";
    
    if (!required_modules.empty()) {
        j["rendering_backend"]["required_modules"] = required_modules;
    } else {
        j["rendering_backend"]["required_modules"] = {
            "vtkRenderingVolume",
            "vtkFiltersFlowPaths",
            "vtkRenderingCore",
            "vtkFiltersCore",
            "vtkCommonCore",
            "vtkIOXML"
        };
    }
    
    // Write main metadata file
    std::ofstream o_meta(meta_file_name + ".json");
    o_meta << std::setw(4) << j << std::endl;
    o_meta.close();

    return j;
}

nlohmann_loader::json generateScriptFixedCamStreamline(std::string meta_file_name,
				      py::list &input_names,
				      int kf_interval,
				      py::array_t<int> dims_in,
				      std::string meshType,
				      int world_bbx_len,
				      py::array_t<float> cam_in,
				      py::array_t<float> tf_range_in,
				      py::array_t<float> tf_colors_in,
				      py::array_t<float> tf_opacities_in,
				      std::string scalar_field,
				      std::string bgImg,
				      float frame_rate,
				      py::list &required_modules_in,
				      py::array_t<float> file_sizes_mb_in,
				      std::string grid_type,
				      py::array_t<float> spacing_in,
				      py::array_t<float> origin_in,
				      float view_angle,
				      std::string rendering_backend,
				      nlohmann_loader::json volume_config,
				      nlohmann_loader::json streamline_config,
				      nlohmann_loader::json isosurface_config)
{
    std::vector<std::string> filenames = input_names.cast<std::vector<std::string>>();
    std::vector<int> dims = makeVectorFromPyArray(dims_in);
    std::vector<float> cam = makeVectorFromPyArray(cam_in);
    std::vector<float> tf_range = makeVectorFromPyArray(tf_range_in);
    std::vector<float> tf_colors = makeVectorFromPyArray(tf_colors_in);
    std::vector<float> tf_opacities = makeVectorFromPyArray(tf_opacities_in);
    std::vector<std::string> required_modules = required_modules_in.cast<std::vector<std::string>>();
    std::vector<float> file_sizes_mb = makeVectorFromPyArray(file_sizes_mb_in);
    std::vector<float> spacing = makeVectorFromPyArray(spacing_in);
    std::vector<float> origin = makeVectorFromPyArray(origin_in);

    if (1){
        std::cout << "output file name "   << meta_file_name << "\n";
        std::cout << "filenames (vtk): "; for (auto s : filenames) std::cout << s <<" "; std::cout << "\n";
        std::cout << "dims: ";      for (auto d : dims)      std::cout << d <<" "; std::cout << "\n";
        std::cout << "cam: ";       for (auto c : cam)       std::cout << c <<" "; std::cout << "\n";
        std::cout << "tf_range: ";  for (auto v : tf_range)  std::cout << v <<" "; std::cout << "\n";
        std::cout << "tf_colors: "; for (auto c : tf_colors) std::cout << c <<" "; std::cout << "\n";
        std::cout << "tf_opacities: "; for (auto o : tf_opacities) std::cout << o <<" "; std::cout << "\n";
        std::cout << "scalar_field: " << scalar_field << "\n";
        std::cout << "key frame interval = " << kf_interval << "\n";
        std::cout << "mesh type = "          << meshType << "\n";
        std::cout << "world bbox size = "    << world_bbx_len << "\n";
        std::cout << "bg img file name = "   << bgImg << "\n";
        std::cout << "frame_rate = "         << frame_rate << "\n";
        std::cout << "required_modules: ";  for (auto m : required_modules) std::cout << m <<" "; std::cout << "\n";
        std::cout << "file_sizes_mb: ";  for (auto f : file_sizes_mb) std::cout << f <<" "; std::cout << "\n";
        std::cout << "grid_type: " << grid_type << "\n";
        std::cout << "spacing: ";  for (auto s : spacing) std::cout << s <<" "; std::cout << "\n";
        std::cout << "origin: ";  for (auto o : origin) std::cout << o <<" "; std::cout << "\n";
        std::cout << "view_angle: " << view_angle << "\n";
        std::cout << "rendering_backend: " << rendering_backend << "\n";
    }

    return fixedCamHelperStreamline(meta_file_name, filenames, kf_interval, dims, meshType, 
                                   world_bbx_len, cam, tf_range, tf_colors, tf_opacities, 
                                   scalar_field, bgImg, frame_rate, required_modules,
                                   file_sizes_mb, grid_type, spacing, origin, 
                                   view_angle, rendering_backend, 
                                   volume_config, streamline_config, isosurface_config);
}

nlohmann_loader::json generateScriptFixedCam(std::string meta_file_name,
				      py::list &input_names,
				      int kf_interval,
				      py::array_t<int> dims_in,
				      std::string meshType,
				      int world_bbx_len,
				      py::array_t<float> cam_in,
				      py::array_t<float> tf_range_in,
				      std::string bgImg
				      )
{
    std::vector<std::string> filenames = input_names.cast<std::vector<std::string>>();
    std::vector<int> dims = makeVectorFromPyArray(dims_in);
    std::vector<float> cam = makeVectorFromPyArray(cam_in);
    std::vector<float> tf_range = makeVectorFromPyArray(tf_range_in);

    if (1){
	// std::cout << "output file name "   << meta_file_name << "\n";
	// std::cout << "filenames: "; for (auto s : filenames) std::cout << s <<" "; std::cout << "\n";
	// std::cout << "dims: ";      for (auto d : dims)      std::cout << d <<" "; std::cout << "\n";
	// std::cout << "cam: ";       for (auto c : cam)       std::cout << c <<" "; std::cout << "\n";
	// std::cout << "tf_range: ";  for (auto v : tf_range)  std::cout << v <<" "; std::cout << "\n";
	// std::cout << "key frame interval = " << kf_interval << "\n";
	// std::cout << "mesh type = "          << meshType << "\n";
	// std::cout << "world bbox size = "    << world_bbx_len << "\n";
	// std::cout << "bg img file name = "   << bgImg << "\n";
    }

    return fixedCamHelper(meta_file_name, filenames, kf_interval, dims, meshType, world_bbx_len, cam, tf_range, bgImg);
}



nlohmann_loader::json generateScriptRotate(std::string meta_file_name,
					   py::list &input_names,
					   int kf_interval,
					   py::array_t<int> dims_in,
					   std::string meshType,
					   int world_bbx_len,
					   float start_degree,
					   float end_degree,
					   float dist,
					   py::array_t<float> tf_range_in,
					   std::string bgImg
					   )
{
    std::vector<std::string> filenames = input_names.cast<std::vector<std::string>>();
    std::vector<int> dims = makeVectorFromPyArray(dims_in);
    std::vector<float> tf_range = makeVectorFromPyArray(tf_range_in);

    size_t kf_nums = filenames.size();

    if (1){
	std::cout << "output file name "   << meta_file_name << "\n";
	std::cout << "filenames: "; for (auto s : filenames) std::cout << s <<" "; std::cout << "\n";
	std::cout << "dims: ";      for (auto d : dims)      std::cout << d <<" "; std::cout << "\n";
	std::cout << "rotate: ";     std::cout << start_degree <<" " <<end_degree <<" dist=" <<dist<< "\n";
	std::cout << "tf_range: ";  for (auto v : tf_range)  std::cout << v <<" "; std::cout << "\n";
	std::cout << "key frame nums = "     << kf_nums << "\n";
	std::cout << "mesh type = "          << meshType << "\n";
	std::cout << "world bbox size = "    << world_bbx_len << "\n";
	std::cout << "bg img file name = "   << bgImg << "\n";
    }
    
    nlohmann_loader::ordered_json j;
    //std::string base_file_name = meta_file_name+"_kf";
    //std::filesystem::path p = std::filesystem::current_path();
    //std::string p_str = p.generic_string() + "/";
    std::filesystem::path meta_path(meta_file_name);
    std::string base_name = meta_path.stem().generic_string(); // "abcd" if input is "abcd.json"

    // Get the output directory (e.g., "GAD_text/")
    std::string gad_dir = meta_path.parent_path().generic_string() + "/";

    // Path to Out_text/ (sibling of GAD_text/)
    std::string out_text_dir = meta_path.parent_path().parent_path().generic_string() + "/Out_text/";

    j["isheader"] = true;
    //j["data_list"] = meta_file_name+"_data_list.json";
	j["data_list"] = base_name + "_data_list.json";
    {
    	nlohmann_loader::ordered_json data_j;
    	for (int i=0; i<filenames.size();i++){
			data_j["list"][i]["type"] = meshType;
			//data_j["list"][i]["src"]["name"] = p_str+filenames[i];
			data_j["list"][i]["src"]["name"] = out_text_dir + filenames[i];
			data_j["list"][i]["src"]["dims"] = {dims[0], dims[1], dims[2]};
    	}
    	//std::ofstream o(j["data_list"]);
		std::string data_list_path = gad_dir + base_name + "_data_list.json";
		std::ofstream o(data_list_path);
		o << std::setw(4)<< data_j <<std::endl;
		o.close();
    }

    float rot_ratio = (end_degree - start_degree)/360.f;
       
    // export all key frames to json file
    // write a header of file names 
    for (size_t i=0; i<filenames.size();i++){
		//std::string file_name = base_file_name + std::to_string(i) + ".json";
		std::string file_name = base_name + "_kf" + std::to_string(i) + ".json";
		j["kf_list"][i] = file_name;
		std::string kf_path = gad_dir + file_name;
		// write json for each keyframe interval
		nlohmann_loader::ordered_json tmp_j;
		tmp_j["isheader"] = false;
		tmp_j["world_bbox"] = {world_bbx_len, world_bbx_len, world_bbx_len};
		tmp_j["frameRange"] = {i*kf_interval, (i+1)*kf_interval};
		if (bgImg != "") tmp_j["data"]["backgroundMap"] = bgImg;

		// cameras
		for (size_t j=0; j<2; j++){
			nlohmann_loader::ordered_json tmp_cam;
			tmp_cam["frame"] = (i+j)*kf_interval;
			
			// float angle_h = (j)/float(kf_nums-1)*M_PI*2*rot_ratio; // radius h, 0-PI
			// Modified angle calculation for rotating camera
			// float angle_h = ((i + j) / float(kf_nums * 2 - 1)) * M_PI * 2 * rot_ratio + (start_degree * M_PI / 180.0);
			float angle_h;
			if (i == kf_nums-1 && j == 1) {
				// For the last camera position, explicitly set to end_degree
				angle_h = end_degree * M_PI/180.0;
			} else {
				// For all other positions, use a formula that distributes evenly
				float steps = kf_nums * 2 - 1; // Total number of camera positions (5 with 3 keyframes)
				angle_h = ((i * 2 + j) / steps) * (end_degree - start_degree) * M_PI/180.0 + (start_degree * M_PI/180.0);
			}
			std::cout << "angle: " << angle_h << "\n";
			float r = dist; // radius r, 0-1
			float p_x = cos(angle_h);
			float p_y = sin(angle_h);
			tmp_cam["pos"] =  {p_x*r, p_y*r, 0};
			tmp_cam["dir"] =  {-p_x ,-p_y, 0};
			tmp_cam["up"]  = {0,0,-1};
				
			tmp_j["camera"].push_back(tmp_cam);
		}
	
		// one data per kf in this template
		// tf
		tmp_j["scene_data_list"][0]["index_in_list"] = i;
		tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["frame"] = i*kf_interval;
		tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["range"] = {tf_range[0], tf_range[1]};

		// fixed tf for now
        tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {0, 0, 0.562493,  0, 0, 1,  0, 1, 1,  0.500008, 1, 0.500008,  1, 1, 0,  1, 0, 0,  0.500008, 0, 0};
		for (int i=0; i<7; i++)
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(i/float(7));
			
		//std::ofstream o(file_name);
		std::ofstream o(kf_path);
		o << std::setw(4)<< tmp_j <<std::endl;
		o.close();
    }
    std::ofstream o_meta(meta_file_name+".json");
    o_meta << std::setw(4) << j <<std::endl;
    o_meta.close();

    return j;
}

nlohmann_loader::json readScript(std::string path){
    nlohmann_loader::json j;
    std::ifstream f(path);
    f >> j;

    return j;
}


PYBIND11_MODULE(vistool_py3, m) {
    // Optional docstring
    m.doc() = "the util library";

    m.def("generateScript", &generateScript, "generate preset script");
    m.def("generateScriptFixedCam", &generateScriptFixedCam, "generate preset script");
    m.def("generateScriptFixedCamStreamline", &generateScriptFixedCamStreamline, "generate preset streamline script");
    m.def("generateScriptRotate", &generateScriptRotate, "generate preset rotate script");
    m.def("readScript", &readScript, "read key frame script from file");
}
