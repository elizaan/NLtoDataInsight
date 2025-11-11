#include "TransferFunctionWidget.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include "KeyframeWidget.h"
#include "../../loader.h"
//#ifndef TFN_WIDGET_NO_STB_IMAGE_IMPL
//#define STB_IMAGE_IMPLEMENTATION
//#endif
//#define STB_IMAGE_IMPLEMENTATION

//#include "stb_image.h"
#include <cstdlib>
#include <string>

namespace keyframe {
  
    template <typename T>
    inline T clamp(T x, T min, T max)
    {
	if (x < min) {
	    return min;
	}
	if (x > max) {
	    return max;
	}
	return x;
    }

    inline float srgb_to_linear(const float x)
    {
	if (x <= 0.04045f) {
	    return x / 12.92f;
	} else {
	    return std::pow((x + 0.055f) / 1.055f, 2.4f);
	}
    }

    Keyframe::Keyframe(ArcballCamera &cam, std::vector<float> &c_in, std::vector<float> &o_in, int time_in, int data_i_in, std::vector<float> &cbox)
    {
        arcballCam.set(cam);
	for (uint32_t i=0; i<c_in.size(); i++) tf_colors.push_back(c_in[i]);
	for (uint32_t i=0; i<o_in.size(); i++) tf_opacities.push_back(o_in[i]);
	for (uint32_t i=0; i<cbox.size(); i++) clippingBox.push_back(cbox[i]);
	timeFrame = time_in;
	data_i = data_i_in;
    }


    KeyframeWidget::vec2f::vec2f(float c) : x(c), y(c) {}

    KeyframeWidget::vec2f::vec2f(float x, float y) : x(x), y(y) {}

    KeyframeWidget::vec2f::vec2f(const ImVec2 &v) : x(v.x), y(v.y) {}

    float KeyframeWidget::vec2f::length() const
    {
	return std::sqrt(x * x + y * y);
    }

    KeyframeWidget::vec2f KeyframeWidget::vec2f::operator+(
							   const KeyframeWidget::vec2f &b) const
    {
	return vec2f(x + b.x, y + b.y);
    }

    KeyframeWidget::vec2f KeyframeWidget::vec2f::operator-(
							   const KeyframeWidget::vec2f &b) const
    {
	return vec2f(x - b.x, y - b.y);
    }

    KeyframeWidget::vec2f KeyframeWidget::vec2f::operator/(
							   const KeyframeWidget::vec2f &b) const
    {
	return vec2f(x / b.x, y / b.y);
    }

    KeyframeWidget::vec2f KeyframeWidget::vec2f::operator*(
							   const KeyframeWidget::vec2f &b) const
    {
	return vec2f(x * b.x, y * b.y);
    }

    KeyframeWidget::vec2f::operator ImVec2() const
    {
	return ImVec2(x, y);
    }

    KeyframeWidget::KeyframeWidget()
    {
    }

    void KeyframeWidget::insert_point(float x){
	time_ctrl_points.push_back(x);
	dataIndex_ctrl_points.push_back(x);
	cam_ctrl_points.push_back(x);
	tfn_ctrl_points.push_back(x);
    }

    void KeyframeWidget::sort_points_by_x(){
	std::sort(time_ctrl_points.begin(), time_ctrl_points.end());
	std::sort(dataIndex_ctrl_points.begin(), dataIndex_ctrl_points.end());
	std::sort(cam_ctrl_points.begin(), cam_ctrl_points.end());
	std::sort(tfn_ctrl_points.begin(), tfn_ctrl_points.end());
    }

    void KeyframeWidget::clear_points(){
	time_ctrl_points.resize(0);
	dataIndex_ctrl_points.resize(0);
	cam_ctrl_points.resize(0);
	tfn_ctrl_points.resize(0);
    }
    
    
    void KeyframeWidget::draw_attribute_line(ImDrawList *draw_list, 
    					const vec2f view_scale,
    					const vec2f view_offset,
    					std::vector<float>& pts, 
    					uint32_t index,
    					std::string txt){
	const float point_radius = 5.f;
	std::vector<ImVec2> polyline_pts;
	draw_list->AddText(ImVec2(view_offset + vec2f(0.f, display_offsets[index]+0.15f)* view_scale), 0xFFFFFFFF, txt.c_str());
	polyline_pts.push_back(vec2f(0, display_offsets[index]) * view_scale + view_offset);
	polyline_pts.push_back(vec2f(1, display_offsets[index]) * view_scale + view_offset);
	draw_list->AddPolyline(polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);
	
	for (size_t i=0; i<pts.size()-1; i++) {
	    const vec2f pt_pos = vec2f(pts[i], display_offsets[index]) * view_scale + view_offset;
	    if (i == selected_point) draw_list->AddCircleFilled(pt_pos, point_radius, IM_COL32(255,255,0,255));
	    else draw_list->AddCircleFilled(pt_pos, point_radius, 0xFFFFFFFF);
	}
    }

    void KeyframeWidget::draw_ui()
    {
	
	const ImGuiIO &io = ImGui::GetIO();

	ImGui::Text(guiText.c_str());
	ImGui::TextWrapped("Left click to add a point, right click remove. ");
	//update_colormap();

	vec2f canvas_size = ImGui::GetContentRegionAvail();
	canvas_size.y = 100;
	// Note: If you're not using OpenGL for rendering your UI, the setup for
	// displaying the colormap texture in the UI will need to be updated.
	//size_t tmp = colormap_img;
	//ImGui::Image(reinterpret_cast<void *>(tmp), ImVec2(canvas_size.x, 16));
	ImDrawList *draw_list = ImGui::GetWindowDrawList();
	vec2f canvas_pos = ImGui::GetCursorScreenPos();
   
	draw_list->PushClipRect(canvas_pos, canvas_pos + canvas_size);

	const vec2f view_scale(canvas_size.x, -canvas_size.y);
	const vec2f view_offset(canvas_pos.x, canvas_pos.y + canvas_size.y);
	const float point_radius = 5.f;

	draw_list->AddRect(canvas_pos, canvas_pos + canvas_size, ImColor(180, 180, 180, 255));

	ImGui::InvisibleButton("tfn_canvas", canvas_size);

	bool clicked_on_item = false;
	bool hover_on_timeline = false;
	if (!io.MouseDown[0] && !io.MouseDown[1]) {
	    clicked_on_item = false;
	}
	if (ImGui::IsItemHovered() && (io.MouseDown[0] || io.MouseDown[1])) {
	    clicked_on_item = true;
	}
	
	if (ImGui::IsItemHovered() && !(io.MouseDown[0] || io.MouseDown[1])) {
	    hover_on_timeline = true;
	}

	ImVec2 bbmin = ImGui::GetItemRectMin();
	ImVec2 bbmax = ImGui::GetItemRectMax();
	ImVec2 clipped_mouse_pos = ImVec2(std::min(std::max(io.MousePos.x, bbmin.x), bbmax.x),
					  std::min(std::max(io.MousePos.y, bbmin.y), bbmax.y));

	if (clicked_on_item) {
	    vec2f mouse_pos = (vec2f(clipped_mouse_pos) - view_offset) / view_scale;
	    mouse_pos.x = clamp(mouse_pos.x, 0.f, 1.f);
	    mouse_pos.y = clamp(mouse_pos.y, 0.f, 1.f);

	    if (io.MouseDown[0]) {
		vec2f v_line_start = vec2f(0, display_offsets[0]) * view_scale + view_offset;
	        if (abs(clipped_mouse_pos.y - v_line_start.y) <= point_radius){
		    auto fnd = std::find_if(time_ctrl_points.begin(), time_ctrl_points.end(), [&](const float &p) {
						const vec2f pt_pos = vec2f(p, display_offsets[0]) * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
						return dist <= point_radius;
					    });
		    
		    // No nearby point, we're adding a new one
		    if (fnd == time_ctrl_points.end()) {
			time_ctrl_points.push_back(mouse_pos.x);
			dataIndex_ctrl_points.push_back(mouse_pos.x);
			cam_ctrl_points.push_back(mouse_pos.x);
			tfn_ctrl_points.push_back(mouse_pos.x);
			// mark this frame to get param from osp
			record_frame = mouse_pos.x * timeFrameMax;
		    }else // there is a selected point, reset value
			selected_point = -1;
		}

		// set selected point
		
		// Keep alpha control points ordered by x coordinate, update
		// selected point index to match
		std::sort(time_ctrl_points.begin(), time_ctrl_points.end());
		std::sort(dataIndex_ctrl_points.begin(), dataIndex_ctrl_points.end());
		std::sort(cam_ctrl_points.begin(), cam_ctrl_points.end());
		std::sort(tfn_ctrl_points.begin(), tfn_ctrl_points.end());
		
		if (selected_point != 0 && selected_point != time_ctrl_points.size() - 1) {
		    auto fnd = std::find_if(time_ctrl_points.begin(), time_ctrl_points.end(), [&](const float &p) {
						const vec2f pt_pos = vec2f(p, display_offsets[0]) * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
						return dist <= point_radius;
					    });
		    selected_point = std::distance(time_ctrl_points.begin(), fnd);
		}
	    } else if (ImGui::IsMouseClicked(1)) {
		selected_point = -1;
		// Find and remove the point
		auto fnd = std::find_if(time_ctrl_points.begin(), time_ctrl_points.end(), [&](const float &p) {
					    const vec2f pt_pos = vec2f(p, display_offsets[0]) * view_scale + view_offset;
					    float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
					    return dist <= point_radius;
					});
		// We also want to prevent erasing the first and last points
		if (fnd != time_ctrl_points.end() && fnd != time_ctrl_points.begin() &&
		    fnd != time_ctrl_points.end() - 1) {
		    dataIndex_ctrl_points.erase(std::remove(dataIndex_ctrl_points.begin(),
							    dataIndex_ctrl_points.end(), *fnd),
						dataIndex_ctrl_points.end());
		    cam_ctrl_points.erase(std::remove(cam_ctrl_points.begin(),
						      cam_ctrl_points.end(), *fnd),
					  cam_ctrl_points.end());
		    tfn_ctrl_points.erase(std::remove(tfn_ctrl_points.begin(),
						      tfn_ctrl_points.end(), *fnd),
						      tfn_ctrl_points.end());
		    
		    time_ctrl_points.erase(fnd);
		}
	    } else {
		// other mouse click, clear selected point
		selected_point = -1;
	    }
	} else {
	    // not clicking
	}

	// Draw the alpha control points, and build the points for the polyline
	// which connects them
	draw_attribute_line(draw_list, view_scale, view_offset, time_ctrl_points, 0, "time");
	draw_attribute_line(draw_list, view_scale, view_offset, dataIndex_ctrl_points, 1, "data_i");
	draw_attribute_line(draw_list, view_scale, view_offset, cam_ctrl_points, 2, "camera");
	draw_attribute_line(draw_list, view_scale, view_offset, tfn_ctrl_points, 3, "tfn");
	
	// draw the hover line
	if (hover_on_timeline) {
		vec2f mouse_pos = (vec2f(clipped_mouse_pos) - view_offset) / view_scale;
	    	mouse_pos.x = clamp(mouse_pos.x, 0.f, 1.f);
	    	mouse_pos.y = clamp(mouse_pos.y, 0.f, 1.f);
		std::vector<ImVec2> polyline_pts;
		polyline_pts.push_back(view_scale*vec2f(mouse_pos.x, 0.f) + view_offset);
		polyline_pts.push_back(view_scale*vec2f(mouse_pos.x, 1.f) + view_offset);
		draw_list->AddPolyline(polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);	
	}
	draw_list->PopClipRect();
	
    }

    
    void KeyframeWidget::recordKeyFrame(ArcballCamera &cam,
					std::vector<float> &tf_colors,
					std::vector<float> &tf_opacities,
					int data_i,
					std::vector<float> &cbox)
    {
        
	kfs.push_back(Keyframe(cam, tf_colors, tf_opacities, record_frame, data_i, cbox));
	std::cout << "adding frame[" << kfs.back().timeFrame<<"] with cam ";
        kfs.back().arcballCam.print();

	// sort key frame by record time
	std::sort(kfs.begin(), kfs.end(), [](Keyframe a, Keyframe b)
	{return a.timeFrame < b.timeFrame;});
	
	record_frame = -1;
    }
    
    void KeyframeWidget::setKeyFrame(ArcballCamera &cam, std::vector<float> &tf_colors, std::vector<float> &tf_opacities, int &data_i){
	if (selected_point > time_ctrl_points.size()-2) return;
	if (selected_point < 0) return;

	// find keyframe in list, both vector sorted by time
	//Keyframe thisKF = kfs[selected_point];
	kfs[selected_point].data_i = data_i;
	kfs[selected_point].arcballCam.set(cam);

	kfs[selected_point].tf_colors.resize(0);
	kfs[selected_point].tf_opacities.resize(0);
	for (uint32_t i=0; i<tf_colors.size(); i++)
	    kfs[selected_point].tf_colors.push_back(tf_colors[i]);
	for (uint32_t i=0; i<tf_opacities.size(); i++)
	    kfs[selected_point].tf_opacities.push_back(tf_opacities[i]);

	std::cout << "set frame["<<kfs[selected_point].timeFrame<<"]  with cam ";
        cam.print();
   
    }
    
    void KeyframeWidget::loadKeyFrame(ArcballCamera &cam, std::vector<float> &tf_colors, std::vector<float> &tf_opacities, int &data_i){
	if (selected_point > time_ctrl_points.size()-2) return;
	if (selected_point < 0) return;

	// find keyframe in list, both vector sorted by time
	Keyframe thisKF = kfs[selected_point];
	data_i = thisKF.data_i;
	cam.set(thisKF.arcballCam);
	for (uint32_t i=0; i<tf_colors.size(); i++) tf_colors.push_back(thisKF.tf_colors[i]);
	for (uint32_t i=0; i<tf_opacities.size(); i++) tf_opacities.push_back(thisKF.tf_opacities[i]);

	std::cout << "load frame["<<thisKF.timeFrame<<"]  with cam ";
        cam.print();
	std::cout << "  data_i=" <<data_i<<"\n";
    }
	
	
    void KeyframeWidget::exportKFs(int dims[3], std::string meshType, int world_bbox[3], std::vector<std::string> &data_fnames, float tf_range_x, float tf_range_y, std::string bgImg){
	
	// Get the output path from environment variable if it exists
	const char* env_path = std::getenv("KF_WIDGET_OUTPUT_PATH");
	if (env_path != nullptr) {
		outputName = env_path;
		std::cout << "Using output path from environment: " << outputName << std::endl;
	}
		// create a header file
	std::string meta_file_name = outputName;
	nlohmann_loader::ordered_json j;
	std::filesystem::path meta_path(meta_file_name);
	std::string base_name = meta_path.stem().generic_string(); // "abcd" if input is "abcd.json"
	std::string gad_dir = meta_path.parent_path().generic_string() + "/";
	std::string out_viewer_dir = meta_path.parent_path().parent_path().generic_string() + "/Out_viewer/"; 
	// std::string base_file_name = meta_file_name+"_kf";
	// std::filesystem::path p = std::filesystem::current_path();
	// std::string p_str = p.generic_string() + "/";
	std::string base_file_name = base_name + "_kf";
	j["isheader"] = true;
	// j["data_list"][0] = {};
	// j["data_list"] = outputName + "_data_list.json";
	j["data_list"] = base_name + "_data_list.json";

	{
        nlohmann_loader::ordered_json data_j;
        for (int i = 0; i < data_fnames.size(); i++) {
            data_j["list"][i]["type"] = meshType;
            // data_j["list"][i]["src"]["name"] = p_str + data_fnames[i];
			data_j["list"][i]["src"]["name"] = out_viewer_dir + data_fnames[i];
            data_j["list"][i]["src"]["dims"] = {dims[0], dims[1], dims[2]};
        }
        // std::ofstream o(j["data_list"]);
		std::string data_list_path = gad_dir + base_name + "_data_list.json";
		std::ofstream o(data_list_path);
        o << std::setw(4) << data_j << std::endl;
        o.close();
    }
	
	// export all key frames to json file 
	for (size_t i=0; i<kfs.size()-1;i++){
	    std::string file_name = base_file_name + std::to_string(i) + ".json";
	    j["kf_list"][i] = file_name;

		std::string kf_path = gad_dir + file_name;

	    // write json for each keyframe interval
	    nlohmann_loader::ordered_json tmp_j;
	    tmp_j["isheader"] = false;
	 
	    tmp_j["world_bbox"] = {world_bbox[0], world_bbox[1], world_bbox[2]};
	    tmp_j["frameRange"] = {kfs[i].timeFrame, kfs[i+1].timeFrame};
	    if (bgImg != "") tmp_j["data"]["backgroundMap"] = bgImg;

	    // cameras
	    for (size_t j=0; j<2; j++)
	    {
			nlohmann_loader::ordered_json tmp_cam;
			tmp_cam["frame"] = kfs[i+j].timeFrame;
			for (size_t c=0; c<3; c++){
				tmp_cam["pos"].push_back(kfs[i+j].arcballCam.eyePos()[c]);
				tmp_cam["dir"].push_back(kfs[i+j].arcballCam.lookDir()[c]);
				tmp_cam["up"].push_back(kfs[i+j].arcballCam.upDir()[c]);
			}
			tmp_j["camera"].push_back(tmp_cam);
	    }

	    // scene data and transfer function
		tmp_j["scene_data_list"][0]["index_in_list"] = 0; // Assuming first dataset
        tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["frame"] = kfs[i].timeFrame;
        tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["range"] = {tf_range_x, tf_range_y};

		// default values for colors and opacities
		if (kfs[i].tf_colors.empty()) {
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = {
				0, 0, 0.562493,  0, 0, 1,  0, 1, 1,  0.500008, 1, 0.500008,  1, 1, 0,  1, 0, 0,  0.500008, 0, 0
			};
		} else {
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"] = nlohmann_loader::json::array();
			for (auto &z : kfs[i].tf_colors) 
				tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["colors"].push_back(z);
		}

		if (kfs[i].tf_opacities.empty()) {
			// Default opacity values (7 values from 0 to 1)
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"] = nlohmann_loader::json::array();
			for (int o = 0; o < 7; o++)
				tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(o/float(7));
		} else {
			tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"] = nlohmann_loader::json::array();
			for (auto &z : kfs[i].tf_opacities) 
				tmp_j["scene_data_list"][0]["scene_info"]["transferFunc"]["opacities"].push_back(z);
		}
	    
	    // std::ofstream o(file_name);
		std::ofstream o(kf_path);
	    o << std::setw(4)<< tmp_j <<std::endl;
	    o.close();
	}
	std::ofstream o_meta(meta_file_name+".json");
	o_meta << std::setw(4) << j <<std::endl;
	o_meta.close();
    }


    void KeyframeWidget::getFrameFromKF(float cam_params[9], std::vector<float> &tf_colors, std::vector<float> &tf_opacities, int &data_i, int f){
	if (f > kfs.back().timeFrame) return;

	int i_prev=0, i_next=0;
	for (size_t i=0; i<kfs.size()-1;i++){
	    if (( f >= kfs[i].timeFrame) && (f <= kfs[i+1].timeFrame))
		{ i_prev = i; i_next = i+1;}
	}

	glm::vec2 range(kfs[i_prev].timeFrame, kfs[i_next].timeFrame);
	visuser::Camera a, b;
	
	for (uint32_t i=0; i<3; i++){
	    a.pos[i] = kfs[i_prev].arcballCam.eyePos()[i];
	    a.dir[i] = kfs[i_prev].arcballCam.lookDir()[i];
	    a.up[i] = kfs[i_prev].arcballCam.upDir()[i];
	    b.pos[i] = kfs[i_next].arcballCam.eyePos()[i];
	    b.dir[i] = kfs[i_next].arcballCam.lookDir()[i];
	    b.up[i] = kfs[i_next].arcballCam.upDir()[i];
	}
	visuser::Camera currentCam = visuser::interpolate(a, b, range, f);

	for (uint32_t i=0; i<3; i++){
	    cam_params[i] = currentCam.pos[i];
	    cam_params[3+i] = currentCam.dir[i];
	    cam_params[6+i] = currentCam.up[i];
	}
	
	// select previous keyframe's tf and data id
	tf_colors.resize(kfs[i_prev].tf_colors.size());	
	tf_opacities.resize(kfs[i_prev].tf_opacities.size());
	for (uint32_t i=0; i<kfs[i_prev].tf_colors.size(); i++)
	    tf_colors[i] = kfs[i_prev].tf_colors[i];
	for (uint32_t i=0; i<kfs[i_prev].tf_opacities.size(); i++)
	    tf_opacities[i] = kfs[i_prev].tf_opacities[i];
	data_i = kfs[i_prev].data_i;
    }

    
    void KeyframeWidget::getDataFilterFromKF(float bbox[4], std::vector<float> &data_indices){
	// init return bbox
	bbox[0] = 1.f; bbox[1] = 0.f;
	bbox[2] = 1.f; bbox[2] = 0.f;
	
	for (uint32_t i=0; i<kfs.size(); i++){
	    // union all camera view
	    bbox[0] = kfs[i].clippingBox[0];
	    bbox[1] = kfs[i].clippingBox[1];
	    bbox[2] = kfs[i].clippingBox[2];
	    bbox[3] = kfs[i].clippingBox[3];

	    // get all data index in use
	    // spread by timeline
	    data_indices.push_back(kfs[i].data_i);
	}
	
    }

    void KeyframeWidget::generatePresetKF(std::string presetName,
					  ArcballCamera &cam,
					  std::vector<float> &tf_colors,
					  std::vector<float> &tf_opacities,
					  uint32_t data_count,
					  std::vector<float> &cbox){
	if (presetName == "fixedCam"){
	    clear_points();
	    kfs.resize(0);
	    
	    uint32_t interval = timeFrameMax/data_count;
	    for (uint32_t i=0; i<data_count; i++){
		record_frame = i*interval;
		recordKeyFrame(cam, tf_colors, tf_opacities, i, cbox);
		insert_point(float(i)/data_count);
	    }

	    // insert an end padding point to keep the last key frame
	    record_frame = timeFrameMax;
	    recordKeyFrame(cam, tf_colors, tf_opacities, data_count-1, cbox);
	    insert_point(1.f);

	    // should be sorted by x already
	    // sort_points_by_x();
	}else{
	    std::cout << "Key frame preset not recognized:" << presetName <<"\n";
	}

	std::cout << kfs.size() << "\n";

    }

}
