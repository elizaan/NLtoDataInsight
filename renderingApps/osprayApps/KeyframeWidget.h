#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "../imgui/imgui.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "ArcballCamera.h"

namespace keyframe {
    struct Keyframe{
	ArcballCamera arcballCam;
	std::vector<float> tf_colors;
	std::vector<float> tf_opacities;
	int timeFrame;
	int data_i;
	std::string filename;
	std::vector<float> clippingBox;
	
	Keyframe(){};
	Keyframe(ArcballCamera &cam, std::vector<float> &tf_colors, std::vector<float> &tf_opacities, int timeFrame, int data_i, std::vector<float> &cbox);
    };
    
    class KeyframeWidget {
	struct vec2f {
	    float x, y;

	    vec2f(float c = 0.f);
	    vec2f(float x, float y);
	    vec2f(const ImVec2 &v);

	    float length() const;

	    vec2f operator+(const vec2f &b) const;
	    vec2f operator-(const vec2f &b) const;
	    vec2f operator/(const vec2f &b) const;
	    vec2f operator*(const vec2f &b) const;
	    operator ImVec2() const;
	};
  
	size_t selected_point = -1;

	bool clicked_on_item = false;
	std::string guiText = "kf timeline";

    public:
	KeyframeWidget();
	std::vector<float> time_ctrl_points = {0.f, 1.f};
	std::vector<float> dataIndex_ctrl_points = {0.f, 1.f};
	std::vector<float> cam_ctrl_points = {0.f, 1.f};
	std::vector<float> tfn_ctrl_points = {0.f, 1.f};
	std::vector<Keyframe> kfs;
	uint32_t timeFrameMax = 20;
	uint32_t viewFrame = 0;
	int record_frame = -1;
	uint32_t playFrame = 0;
	bool play = false;
	std::string outputName = "viewer_script";

	void draw_ui();
	void setGuiText(std::string in) { guiText = in;}

	void recordKeyFrame(ArcballCamera &cam,
			    std::vector<float> &tf_colors,
			    std::vector<float> &tf_opacities,
			    int i,
			    std::vector<float> &cbox);
	
	void setKeyFrame(   ArcballCamera &cam,
			    std::vector<float> &tf_colors,
			    std::vector<float> &tf_opacities,
			    int &data_i);
	
	void loadKeyFrame(  ArcballCamera &cam,
			    std::vector<float> &tf_colors,
			    std::vector<float> &tf_opacities,
			    int &data_i);
	
	void getFrameFromKF(float cam[9],
			    std::vector<float> &tf_colors,
			    std::vector<float> &tf_opacities,
			    int &data_i,
			    int f);
	
	void exportKFs(     int dim[3],
			    std::string meshType,
			    int world_bbox[3],
			    std::vector<std::string> &data_fnames,
			    float tf_range_x,
			    float tf_range_y,
			    std::string bgImg);
	
	void getDataFilterFromKF(float bbox[4],
				 std::vector<float>
				 &data_indices);

	void generatePresetKF(std::string presetName,
			      ArcballCamera &cam,
			      std::vector<float> &tf_colors,
			      std::vector<float> &tf_opacities,
			      uint32_t data_count,
			      std::vector<float> &cbox);
	
    private:
	void insert_point(float x);
	void sort_points_by_x();
	void clear_points();
	void draw_attribute_line(ImDrawList *draw_list, const vec2f view_scale, const vec2f view_offset, std::vector<float>& pts, uint32_t index, std::string txt);
	float display_offsets[4]  = {0.85f, 0.5f, 0.3f, 0.1f};
    };
}
