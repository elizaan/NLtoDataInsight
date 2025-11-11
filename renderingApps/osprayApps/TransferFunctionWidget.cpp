#include "TransferFunctionWidget.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include "embedded_colormaps.h"

//#ifndef TFN_WIDGET_NO_STB_IMAGE_IMPL
//#define STB_IMAGE_IMPLEMENTATION
//#endif
//#define STB_IMAGE_IMPLEMENTATION

//#include "stb_image.h"

namespace tfnw {
  
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

    Colormap::Colormap(const std::string &name,
		       const std::vector<uint8_t> &img,
		       const ColorSpace color_space)
	: name(name), colormap(img), color_space(color_space)
    {
    }

    TransferFunctionWidget::vec2f::vec2f(float c) : x(c), y(c) {}

    TransferFunctionWidget::vec2f::vec2f(float x, float y) : x(x), y(y) {}

    TransferFunctionWidget::vec2f::vec2f(const ImVec2 &v) : x(v.x), y(v.y) {}

    float TransferFunctionWidget::vec2f::length() const
    {
	return std::sqrt(x * x + y * y);
    }

    TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator+(
									   const TransferFunctionWidget::vec2f &b) const
    {
	return vec2f(x + b.x, y + b.y);
    }

    TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator-(
									   const TransferFunctionWidget::vec2f &b) const
    {
	return vec2f(x - b.x, y - b.y);
    }

    TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator/(
									   const TransferFunctionWidget::vec2f &b) const
    {
	return vec2f(x / b.x, y / b.y);
    }

    TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator*(
									   const TransferFunctionWidget::vec2f &b) const
    {
	return vec2f(x * b.x, y * b.y);
    }

    TransferFunctionWidget::vec2f::operator ImVec2() const
    {
	return ImVec2(x, y);
    }

    TransferFunctionWidget::TransferFunctionWidget()
    {
	// Load up the embedded colormaps as the default options
	// NOT in use for now
	load_embedded_preset(paraview_cool_warm, sizeof(paraview_cool_warm), "color");

	// Initialize the colormap alpha channel w/ a linear ramp
	update_colormap();
    }

    void TransferFunctionWidget::add_colormap(const Colormap &map)
    {
	colormaps.push_back(map);

	if (colormaps.back().color_space == SRGB) {
	    Colormap &cmap = colormaps.back();
	    cmap.color_space = LINEAR;
	    for (size_t i = 0; i < cmap.colormap.size() / 4; ++i) {
		for (size_t j = 0; j < 3; ++j) {
		    const float x = srgb_to_linear(cmap.colormap[i * 4 + j] / 255.f);
		    cmap.colormap[i * 4 + j] = static_cast<uint8_t>(clamp(x * 255.f, 0.f, 255.f));
		}
	    }
	}
    }

    void TransferFunctionWidget::draw_ui()
    {
	update_gpu_image();

	const ImGuiIO &io = ImGui::GetIO();

	ImGui::Text(guiText.c_str());
	ImGui::TextWrapped(
			   "Left click to add a point, right click remove. "
			   "Left click + drag to move points.");
	//update_colormap();

	vec2f canvas_size = ImGui::GetContentRegionAvail();
	canvas_size.y = 100;
	// Note: If you're not using OpenGL for rendering your UI, the setup for
	// displaying the colormap texture in the UI will need to be updated.
	//size_t tmp = colormap_img;
	//ImGui::Image(reinterpret_cast<void *>(tmp), ImVec2(canvas_size.x, 16));
	ImDrawList *draw_list = ImGui::GetWindowDrawList();
	vec2f canvas_pos = ImGui::GetCursorScreenPos();
    
	const float point_radius = 10.f;
   
	draw_list->PushClipRect(canvas_pos, canvas_pos + canvas_size);

	const vec2f view_scale(canvas_size.x, -canvas_size.y);
	const vec2f view_offset(canvas_pos.x, canvas_pos.y + canvas_size.y);

	draw_list->AddRect(canvas_pos, canvas_pos + canvas_size, ImColor(180, 180, 180, 255));

	ImGui::InvisibleButton("tfn_canvas", canvas_size);

	bool clicked_on_item = false;
	if (!io.MouseDown[0] && !io.MouseDown[1]) {
	    clicked_on_item = false;
	}
	if (ImGui::IsItemHovered() && (io.MouseDown[0] || io.MouseDown[1])) {
	    clicked_on_item = true;
	}

	ImVec2 bbmin = ImGui::GetItemRectMin();
	ImVec2 bbmax = ImGui::GetItemRectMax();
	ImVec2 clipped_mouse_pos = ImVec2(std::min(std::max(io.MousePos.x, bbmin.x), bbmax.x),
					  std::min(std::max(io.MousePos.y, bbmin.y), bbmax.y));

	if (clicked_on_item) {
	    colormap_changed = true;
	    vec2f mouse_pos = (vec2f(clipped_mouse_pos) - view_offset) / view_scale;
	    mouse_pos.x = clamp(mouse_pos.x, 0.f, 1.f);
	    mouse_pos.y = clamp(mouse_pos.y, 0.f, 1.f);

	    if (io.MouseDown[0]) {
		if (selected_point != (size_t)-1) {
		    alpha_control_pts[selected_point] = mouse_pos;

		    // Keep the first and last control points at the edges
		    if (selected_point == 0) {
			alpha_control_pts[selected_point].x = 0.f;
		    } else if (selected_point == alpha_control_pts.size() - 1) {
			alpha_control_pts[selected_point].x = 1.f;
		    }
		} else {
		    auto fnd = std::find_if(
					    alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f &p) {
						const vec2f pt_pos = p * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
						return dist <= point_radius;
					    });
		    // No nearby point, we're adding a new one
		    if (fnd == alpha_control_pts.end()) {
			alpha_control_pts.push_back(mouse_pos);
		    }
		}

		// Keep alpha control points ordered by x coordinate, update
		// selected point index to match
		std::sort(alpha_control_pts.begin(),
			  alpha_control_pts.end(),
			  [](const vec2f &a, const vec2f &b) { return a.x < b.x; });
		if (selected_point != 0 && selected_point != alpha_control_pts.size() - 1) {
		    auto fnd = std::find_if(
					    alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f &p) {
						const vec2f pt_pos = p * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
						return dist <= point_radius;
					    });
		    selected_point = std::distance(alpha_control_pts.begin(), fnd);
		}
		update_colormap();
	    } else if (ImGui::IsMouseClicked(1)) {
		selected_point = -1;
		// Find and remove the point
		auto fnd = std::find_if(
					alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f &p) {
					    const vec2f pt_pos = p * view_scale + view_offset;
					    float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
					    return dist <= point_radius;
					});
		// We also want to prevent erasing the first and last points
		if (fnd != alpha_control_pts.end() && fnd != alpha_control_pts.begin() &&
		    fnd != alpha_control_pts.end() - 1) {
		    alpha_control_pts.erase(fnd);
		}
		update_colormap();
	    } else {
		selected_point = -1;
	    }
	} else {
	    selected_point = -1;
	}

	// Draw the alpha control points, and build the points for the polyline
	// which connects them
	std::vector<ImVec2> polyline_pts;
	for (const auto &pt : alpha_control_pts) {
	    const vec2f pt_pos = pt * view_scale + view_offset;
	    polyline_pts.push_back(pt_pos);
	    draw_list->AddCircleFilled(pt_pos, point_radius, 0xFFFFFFFF);
	}
	draw_list->AddPolyline(
			       polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);
	draw_list->PopClipRect();
    }

    bool TransferFunctionWidget::changed() const
    {
	return colormap_changed;
    }

    std::vector<uint8_t> TransferFunctionWidget::get_colormap()
    {
	colormap_changed = false;
	return current_colormap;
    }

    std::vector<float> TransferFunctionWidget::get_colormapf()
    {
	colormap_changed = false;
	std::vector<float> colormapf(current_colormap.size(), 0.f);
	for (size_t i = 0; i < current_colormap.size(); ++i) {
	    colormapf[i] = current_colormap[i] / 255.f;
	}
	return colormapf;
    }

    void TransferFunctionWidget::get_colormapf(std::vector<float> &color,
					       std::vector<float> &opacity)
    {
	colormap_changed = false;
	color.resize((current_colormap.size() / 4) * 3);
	opacity.resize(current_colormap.size() / 4);
	for (size_t i = 0; i < current_colormap.size() / 4; ++i) {
	    color[i * 3] = current_colormap[i * 4] / 255.f;
	    color[i * 3 + 1] = current_colormap[i * 4 + 1] / 255.f;
	    color[i * 3 + 2] = current_colormap[i * 4 + 2] / 255.f;
	    opacity[i] = current_colormap[i * 4 + 3] / 255.f;
	}
    }
    void TransferFunctionWidget::set_osp_colormapf(std::vector<float> &colors, std::vector<float> &opacities)
    {
	osp_colors.resize(colors.size());
	
	for (uint32_t i=0; i<colors.size(); i++){
	    osp_colors[i] = colors[i];
	}

	if (opacities.size() > 1){
	    alpha_control_pts.resize(opacities.size());
	    for (uint32_t i=0; i<opacities.size(); i++){
		float cur_x = i/float(opacities.size()-1);
		float cur_y = clamp(float(opacities[i]), 0.f, 1.f);
	        alpha_control_pts[i].x = cur_x;
	        alpha_control_pts[i].y = cur_y;
	    }
	}else{
	    alpha_control_pts.resize(2);
	    alpha_control_pts[0].x = 0.f;
	    alpha_control_pts[0].y = clamp(float(opacities[0]), 0.f, 1.f);
	    alpha_control_pts[1].x = 1.f;
	    alpha_control_pts[1].y = clamp(float(opacities[0]), 0.f, 1.f);
	}
	update_colormap();

    }


    void TransferFunctionWidget::get_osp_colormapf(std::vector<float> &color,
					       std::vector<float> &opacity)
    {
	colormap_changed = false;
	color.resize(osp_colors.size());
	opacity.resize(current_colormap.size() / 4);
	for (size_t i = 0; i < current_colormap.size() / 4; ++i) {
	    opacity[i] = current_colormap[i * 4 + 3] / 255.f;
	}
	for (size_t i = 0; i < color.size() ; ++i) {
	    color[i] = osp_colors[i];
	}
	
    }


    void TransferFunctionWidget::update_gpu_image()
    {
	GLint prev_tex_2d = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex_2d);

	if (colormap_img == (GLuint)-1) {
	    glGenTextures(1, &colormap_img);
	    glBindTexture(GL_TEXTURE_2D, colormap_img);
	
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	if (gpu_image_stale) {
	    gpu_image_stale = false;
	    glBindTexture(GL_TEXTURE_2D, colormap_img);
	    glTexImage2D(GL_TEXTURE_2D,
			 0,
			 GL_RGB8,
			 (GLsizei)(current_colormap.size() / 4),
			 1,
			 0,
			 GL_RGBA,
			 GL_UNSIGNED_BYTE,
			 current_colormap.data());
	}
	glBindTexture(GL_TEXTURE_2D, prev_tex_2d);
    }

    void TransferFunctionWidget::update_colormap()
    {
	//colormap_changed = false;
	gpu_image_stale = true;
	current_colormap = colormaps[selected_colormap].colormap;
	// We only change opacities for now, so go through and update the opacity
	// by blending between the neighboring control points
	auto a_it = alpha_control_pts.begin();
	const size_t npixels = current_colormap.size() / 4;
	for (size_t i = 0; i < npixels; ++i) {
	    float x = static_cast<float>(i) / npixels;
	    auto high = a_it + 1;
	    if (x > high->x) {
		++a_it;
		++high;
	    }
	    float t = (x - a_it->x) / (high->x - a_it->x);
	    float alpha = (1.f - t) * a_it->y + t * high->y;
	    current_colormap[i * 4 + 3] = static_cast<uint8_t>(clamp(alpha * 255.f, 0.f, 255.f));
	}
    }

    void TransferFunctionWidget::load_embedded_preset(const uint8_t *buf,
						      size_t size,
						      const std::string &name)
    {
	int w=127, h, n;
	uint8_t img_data[] = { // rainbow
	    0x89,0x50,0x4e,0x47,0xd,0xa,0x1a,0xa,0x0,0x0,0x0,0xd,0x49,0x48,0x44,0x52,0x0,0x0,0x0,0xb4,0x0,0x0,0x0,0x1,0x8,0x6,0x0,0x0,0x0,0x77,0xca,0x84,0xf4,0x0,0x0,0x0,0x46,0x49,0x44,0x41,0x54,0x78,0x5e,0x63,0x64,0x60,0xf8,0xff,0x9f,0x81,0x85,0x81,0x1,0x8c,0x39,0x70,0xd0,0x20,0xf9,0xc1,0x24,0xc7,0xfe,0x1f,0xe8,0xd0,0x3f,0x40,0xfc,0x3,0x8d,0xc6,0x26,0x6,0x53,0x33,0xb0,0x72,0x4c,0xc,0xff,0x18,0xd0,0x83,0x70,0xb0,0x5,0x2b,0xba,0xfb,0x18,0x7f,0x42,0x83,0x77,0xf0,0x6,0x2b,0x3,0x3,0xb2,0xdb,0xfe,0x30,0x30,0x0,0x0,0x85,0x9e,0x5d,0x0,0x63,0x5b,0x7,0x6f,0x0,0x0,0x0,0x0,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82
	};;
	auto img = std::vector<uint8_t>(img_data, img_data + w * 1 * 4);
	colormaps.emplace_back(name, img, SRGB);
	Colormap &cmap = colormaps.back();
   
	for (size_t i = 0; i < cmap.colormap.size() / 4; ++i) {
	    for (size_t j = 0; j < 3; ++j) {
		const float x = srgb_to_linear(cmap.colormap[i * 4 + j] / 255.f);
		cmap.colormap[i * 4 + j] = static_cast<uint8_t>(clamp(x * 255.f, 0.f, 255.f));
	    }
	}
    }

}
