#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "../imgui/imgui.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace tfnw {

enum ColorSpace { LINEAR, SRGB };

struct Colormap {
    std::string name;
    // An RGBA8 1D image
    std::vector<uint8_t> colormap;
    ColorSpace color_space;

    Colormap(const std::string &name,
             const std::vector<uint8_t> &img,
             const ColorSpace color_space);
};

class TransferFunctionWidget {
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

    std::vector<Colormap> colormaps;
    size_t selected_colormap = 0;
    std::vector<uint8_t> current_colormap;
  
    size_t selected_point = -1;

    bool clicked_on_item = false;
    bool gpu_image_stale = true;
    bool colormap_changed = false;
    GLuint colormap_img = -1;
    std::string guiText = "color map";

public:
    TransferFunctionWidget();
    std::vector<vec2f> alpha_control_pts = {vec2f(0.f), vec2f(1.f)};
    std::vector<float> osp_colors;
    
    // Add a colormap preset. The image should be a 1D RGBA8 image, if the image
    // is provided in sRGBA colorspace it will be linearized
    void add_colormap(const Colormap &map);

    // Add the transfer function UI into the currently active window
    void draw_ui();

    // Returns true if the colormap was updated since the last
    // call to draw_ui
    bool changed() const;

    void setUnchanged(){colormap_changed = false;}

    // Get back the RGBA8 color data for the transfer function
    std::vector<uint8_t> get_colormap();

    // Get back the RGBA32F color data for the transfer function
    std::vector<float> get_colormapf();

    // Get back the RGBA32F color data for the transfer function
    // as separate color and opacity vectors
    void get_colormapf(std::vector<float> &color, std::vector<float> &opacity);

    // get piecewise linear colormap
    void set_osp_colormapf(std::vector<float> &color, std::vector<float> &opacity);
    void get_osp_colormapf(std::vector<float> &color, std::vector<float> &opacity);

  std::vector<vec2f> get_alpha_control_pts(){ return alpha_control_pts;}

  void setGuiText(std::string in) { guiText = in;}
private:
    void update_gpu_image();

    void update_colormap();

    void load_embedded_preset(const uint8_t *buf, size_t size, const std::string &name);
};
}
