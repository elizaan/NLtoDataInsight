#include "ospray/ospray_cpp/ext/rkcommon.h"

namespace visuser
{
    namespace rkm = rkcommon::math;
    class RectMesh
    {
    public:
	RectMesh(){};
    	RectMesh(rkm::vec3ui dimensions, 
		 std::vector<float> z_mapping,
		 std::vector<float> &v,
		 rkm::vec2f xy_scaling = {1.0f, 1.0f});
        RectMesh(rkm::vec3ui dimensions, 
		 std::vector<float> z_mapping,
		 std::string rawFilePath,
		 rkm::vec2f xy_scaling = {1.0f, 1.0f});

	void initMesh(rkm::vec3ui dimensions, 
		      std::vector<float> &z_mapping,
		      rkm::vec2f xy_scaling);
	void initSphereMesh(rkm::vec3ui dimensions, 
			    std::vector<float> &z_mapping,
			    rkm::vec2f xy_scaling);

	void setValue(std::vector<float> &v);
	
        std::vector<float> scalars;
        std::vector<rkm::vec3f> vertices;
        std::vector<size_t> indices;
	std::vector<size_t> cells;
        std::vector<uint8_t> cellTypes;
    private:
    };
};
