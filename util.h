#pragma once

#include <string>
#include <vector>
#include "json.hpp"
#include "ext/glm/glm.hpp"

using namespace glm;

// Read the contents of a file into the string
std::string get_file_content(const std::string &fname);

std::string get_file_extension(const std::string &fname);

std::string get_file_basename(const std::string &path);

std::string get_file_basepath(const std::string &path);

bool starts_with(const std::string &str, const std::string &prefix);

std::vector<glm::vec3> generate_fibonacci_sphere(const size_t n_points, const float radius);

inline glm::vec3 get_vec3f(const nlohmann_loader::json &j)
{
    glm::vec3 v;
    for (size_t i = 0; i < 3; ++i) {
        v[i] = j[i].get<float>();
    }
    return v;
}

inline glm::vec3 get_vec2f(const nlohmann_loader::json &j)
{
    glm::vec3 v;
    for (size_t i = 0; i < 2; ++i) {
        v[i] = j[i].get<float>();
    }
    return v;
}

inline glm::vec3 get_vec3i(const nlohmann_loader::json &j)
{
    glm::vec3 v;
    for (size_t i = 0; i < 3; ++i) {
        v[i] = j[i].get<int>();
    }
    return v;
}

inline glm::vec2 get_vec2i(const nlohmann_loader::json &j)
{
    glm::vec3 v;
    for (size_t i = 0; i < 2; ++i) {
        v[i] = j[i].get<int>();
    }
    return v;
}


