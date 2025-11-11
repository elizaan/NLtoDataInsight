#pragma once

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>
#include <ospray/ospray_cpp/ext/rkcommon.h>
#include <rkcommon/math/box.h>
#include <rkcommon/math/vec.h>

using namespace ospray;
using namespace rkcommon;

struct ClippingPlaneParams {
    int axis = 0;
    bool flip_plane = false;
    bool enabled = false;
    math::vec3f position;

    ClippingPlaneParams() = default;

    ClippingPlaneParams(int axis, const math::vec3f &pos);
};

struct ClippingPlane {
    ClippingPlaneParams params;

    cpp::Geometry geom;
    cpp::GeometricModel model;
    cpp::Group group;
    cpp::Instance instance;

    ClippingPlane(const ClippingPlaneParams &params = ClippingPlaneParams());

    void update(const ClippingPlaneParams &params);
};

