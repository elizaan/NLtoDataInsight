#ifndef VTK_FUNS2_H
#define VTK_FUNS2_H

#include <string>
#include <vector>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkImageData.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkLookupTable.h>
#include <vtkActor.h>
#include "../../loader.h"

using namespace visuser;
using namespace nlohmann_loader;

// =============================================================================
// IMPROVED VISUALIZATION FUNCTIONS
// =============================================================================

/**
 * Get output filename with improved naming convention
 */
std::string getOutName2(std::string str, uint32_t idx);

/**
 * Setup oceanographic colormaps for different field types
 * @param colorTF - VTK color transfer function to configure
 * @param range - Data value range [min, max]
 * @param field_type - "salinity", "temperature", or "default"
 */
void setupOceanicColormap(vtkColorTransferFunction* colorTF, double* range, const std::string& field_type, const std::vector<float>& json_colors = std::vector<float>());

/**
 * Setup depth-based opacity for realistic ocean visualization
 * @param opacityTF - VTK opacity transfer function to configure
 * @param range - Data value range [min, max]
 */
void setupDepthBasedOpacity(vtkPiecewiseFunction* opacityTF, double* range, const std::vector<float>& json_opacities = std::vector<float>());

/**
 * Setup streamline colormaps based on scalar type
 * @param lut - VTK lookup table to configure
 * @param scalar_type - "velocity", "salinity", or "default"
 */
void setupStreamlineColormap(vtkLookupTable* lut, const std::string& scalar_type);

/**
 * Detect land areas based on salinity/temperature values
 * @param salinity_value - Salinity value at point
 * @param temperature_value - Temperature value at point (optional)
 * @return true if likely land area
 */
bool detectLandArea(double salinity_value, double temperature_value = 0.0);

/**
 * Add oceanographic gradient background
 * @param ren - VTK renderer to configure
 */
void addOceanBackground(vtkRenderer* ren);

/**
 * Enhance streamline visual properties
 * @param streamlineActor - VTK actor for streamlines
 */
void enhanceStreamlineVisibility(vtkActor* streamlineActor);

/**
 * Print visualization information and oceanographic context
 * @param scalar_range - Range of scalar values
 * @param velocity_range - Range of velocity values (optional)
 */
void printVisualizationInfo(double* scalar_range, double* velocity_range = nullptr);

// =============================================================================
// IMPROVED CORE FUNCTIONS
// =============================================================================

/**
 * Load transfer function with improved settings (version 2)
 */
void loadTransferFunction2(json &j, vtkVolumeProperty *volumeProperty);

/**
 * Load camera settings (version 2)
 */
void loadCamera2(AnimatorKF &keyframe, vtkRenderer *ren1);

/**
 * Load keyframe with improved visualization (version 2)
 */
void loadKF2(Animator &animator, uint32_t idx, vtkImageData *img, vtkRenderer *ren1);

/**
 * Write image with improved feedback (version 2)
 */
void writeImage2(std::string const& fileName, vtkRenderWindow* renWin, bool rgba);

/**
 * Main rendering function with improved oceanographic visualization (version 2)
 * @param jsonStr - JSON configuration string
 * @param output_dir - Output directory path
 * @param header_sel - Rendering selection mode
 */
void run2(std::string jsonStr, std::string output_dir, int header_sel);

#endif // VTK_FUNS2_H
