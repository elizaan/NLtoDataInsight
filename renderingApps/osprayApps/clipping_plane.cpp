#include "clipping_plane.h"

ClippingPlaneParams::ClippingPlaneParams(int axis, const math::vec3f &pos)
    : axis(axis), position(pos)
{
}

ClippingPlane::ClippingPlane(const ClippingPlaneParams &params) : params(params), geom("plane")
{
    math::vec4f normal(0.f);
    normal[params.axis] = 1.f;
    geom.setParam("plane.coefficients", cpp::CopiedData(normal));
    geom.commit();

    model = cpp::GeometricModel(geom);
    model.commit();

    group.setParam("clippingGeometry", cpp::CopiedData(model));
    group.commit();

    instance = cpp::Instance(group);
    const math::affine3f xfm = math::affine3f::translate(
        math::vec3f(params.position.x, params.position.y, params.position.z));
    instance.setParam("xfm", xfm);
    instance.commit();
}

void ClippingPlane::update(const ClippingPlaneParams &p) {
    params = p;

    math::vec4f normal(0.f);
    normal[params.axis] = 1.f;
    geom.setParam("plane.coefficients", cpp::CopiedData(normal));
    model.setParam("invertNormals", params.flip_plane);
    geom.commit();

    model.commit();

    group.setParam("clippingGeometry", cpp::CopiedData(model));

    const math::affine3f xfm = math::affine3f::translate(
        math::vec3f(params.position.x, params.position.y, params.position.z));
    instance.setParam("xfm", xfm);
    instance.commit();
}

