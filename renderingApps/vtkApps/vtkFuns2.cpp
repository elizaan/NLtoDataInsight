#include <filesystem>  // Add this line
#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFixedPointVolumeRayCastMapper.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkImageData.h>
#include <vtkThreshold.h>
#include <vtkGeometryFilter.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

#include <vtkSmartPointer.h>
#include <vtkImageWriter.h>
#include <vtkPNGWriter.h>
#include <vtkFloatArray.h>
#include <vtkPostScriptWriter.h>
#include <vtkWindowToImageFilter.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkStreamTracer.h>
#include <vtkStructuredGridOutlineFilter.h>
#include <vtkPointData.h>
#include <vtkStreamTracer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>
#include <vtkPolyDataWriter.h>
#include <array>
#include <vtkPointSource.h>
#include <vtkPlaneSource.h>
#include <vtkLookupTable.h>
#include <vtkMarchingCubes.h>
#include <vtkGlyph3D.h>
#include <vtkArrowSource.h>
#include <vtkMaskPoints.h>

#include <fstream>
#include <filesystem>
#include "../../loader.h"
#include "vtkFuns.h"
#include <vtkPNGReader.h>
#include <vtkTexture.h>


using namespace visuser;
using namespace nlohmann_loader;

const float MAX_LAND_SALINITY = 0.005;

std::string getOutName2(std::string str, uint32_t idx)
{
    std::string base_filename = str.substr(str.find_last_of("/\\") + 1);
    std::string outname = base_filename.substr(0, base_filename.find_last_of("."));
    return "img_"+outname+"_kf"+std::to_string(idx)+".png";
}

// =============================================================================
// NEW IMPROVED VISUALIZATION HELPER FUNCTIONS
// =============================================================================
void createLandIsosurface(vtkImageData* img, vtkRenderer* ren1, double* range) {
    // // Create isosurface for land areas (salinity ≈ 0

    // // Create land actor
    // const std::string textureFilePath = "/home/eliza89/PhD/codes/vis_user_tool/renderingApps/vtkApps/mediterranean_mask_land.png";
    const std::string textureFilePath = "/home/eliza89/PhD/codes/vis_user_tool/renderingApps/vtkApps/agulhaas_mask_land.png";
    vtkNew<vtkThreshold> threshold;
    threshold->SetInputData(img);
    threshold->SetLowerThreshold(0.0f);
    threshold->SetUpperThreshold(MAX_LAND_SALINITY);
    threshold->Update();
    // Convert the thresholded data to polydata for visualization
    vtkNew<vtkGeometryFilter> geometryFilter;
    geometryFilter->SetInputConnection(threshold->GetOutputPort());
    geometryFilter->Update();

    // Print XYZ coordinates of sample points on the land isovolume
    vtkPolyData* landPolyData = geometryFilter->GetOutput();
    vtkPoints* points = landPolyData->GetPoints();
    if (points) {
        std::cout << "Land isovolume found with " << points->GetNumberOfPoints() << " points" << std::endl;

        double bounds[6];
        landPolyData->GetBounds(bounds);
        double x_min = bounds[0], x_max = bounds[1];
        double y_min = bounds[2], y_max = bounds[3];
        double z_min = bounds[4], z_max = bounds[5];

        // std::cout << "Land isovolume bounds: " << std::endl;
        // std::cout << "  X: [" << x_min << ", " << x_max << "]" << std::endl;
        // std::cout << "  Y: [" << y_min << ", " << y_max << "]" << std::endl;
        // std::cout << "  Z: [" << z_min << ", " << z_max << "]" << std::endl;

        // Create texture coordinates array
        vtkNew<vtkFloatArray> texCoords;
        texCoords->SetName("TextureCoordinates");
        texCoords->SetNumberOfComponents(2);
        texCoords->SetNumberOfTuples(points->GetNumberOfPoints());
        
        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
            double pt[3];
            points->GetPoint(i, pt);
            
            // Map 3D coordinates to 2D texture coordinates (0-1 range)
            double u = (z_max > z_min) ? (pt[2] - z_min) / (z_max - z_min) : 0.0;
            double v = (y_max > y_min) ? (pt[1] - y_min) / (y_max - y_min) : 0.0;

            // std::cout << "Point " << i << ": (" << pt[0] << ", " << pt[1] << ", " << pt[2] 
            //           << ") -> Texture Coords: (" << u << ", " << v << ")" << std::endl;
            
            // Clamp to [0,1] range to avoid texture wrapping issues
            u = std::max(0.0, std::min(1.0, u));
            v = std::max(0.0, std::min(1.0, v));
            
            texCoords->SetTuple2(i, u, v);
        }

        landPolyData->GetPointData()->SetTCoords(texCoords);
        
        // Check if texture file exists before trying to load it
        if (std::filesystem::exists(textureFilePath)) {
            // Load texture
            vtkNew<vtkPNGReader> textureReader;
            textureReader->SetFileName(textureFilePath.c_str());
            textureReader->Update();
            
            vtkNew<vtkTexture> texture;
            texture->SetInputConnection(textureReader->GetOutputPort());
            texture->InterpolateOn();
            texture->RepeatOff(); // Prevent texture wrapping
            
            // Create mapper with texture
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputData(landPolyData);
            mapper->ScalarVisibilityOff(); // Turn off scalar coloring to use texture instead
            
            vtkNew<vtkActor> actor;
            actor->SetMapper(mapper);
            actor->SetTexture(texture);
            actor->GetProperty()->SetOpacity(1.0);
            
            // std::cout << "Applied Mediterranean land texture from: " << textureFilePath << std::endl;
            
            // Add to renderer
            ren1->AddActor(actor);
            return;
        } else {
            std::cout << "Warning: Texture file not found at " << textureFilePath << std::endl;
            std::cout << "Falling back to color-based land rendering" << std::endl;
        }

        // Fallback: Create color-based land rendering if texture fails
        // Create a normalized y coordinate array for elevation-based coloring
        vtkNew<vtkFloatArray> yNormArray;
        yNormArray->SetName("y_norm");
        yNormArray->SetNumberOfComponents(1);
        yNormArray->SetNumberOfTuples(points->GetNumberOfPoints());
        
        for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
            double pt[3];
            points->GetPoint(i, pt);
            double y_norm = (y_max > y_min) ? (pt[1] - y_min) / (y_max - y_min) : 0.0;
            yNormArray->SetTuple1(i, y_norm);
        }
        landPolyData->GetPointData()->SetScalars(yNormArray);

        // Create a black-to-red lookup table for elevation coloring
        vtkNew<vtkLookupTable> lut;
        lut->SetNumberOfTableValues(256);
        lut->SetRange(0.0, 1.0);
        for (int i = 0; i < 256; ++i) {
            double t = double(i) / 255.0;
            lut->SetTableValue(i, t, 0.0, 0.0, 1.0); // RGBA: black to red
        }
        lut->Build();

        // Create a mapper for the isovolume
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(landPolyData);
        mapper->SetLookupTable(lut);
        mapper->SetScalarRange(0.0, 1.0);
        mapper->ScalarVisibilityOn();

        // Create an actor for the isovolume
        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetOpacity(1.0);

        std::cout << "Applied fallback elevation-based land coloring" << std::endl;
        
        // Add to renderer
        ren1->AddActor(actor);
    } else {
        std::cout << "No points found in land isovolume." << std::endl;
    }
    // Create an actor for the isovolume
    // vtkNew<vtkActor> actor;
    // actor->SetMapper(mapper);
    // actor->GetProperty()->SetColor(0.518, 0.408, 0.216);
    // actor->GetProperty()->SetOpacity(1.0);

    // Add to renderer
    // ren1->AddActor(actor);
}

void setupOceanicColormap(vtkColorTransferFunction* colorTF, double* range, const std::string& field_type, const std::vector<float>& json_colors) {
    colorTF->RemoveAllPoints();
    
    if (field_type == "salinity") {
        // Use JSON colors for ocean colormap if provided, otherwise use hardcoded
        if (!json_colors.empty() && json_colors.size() >= 27) { // 9 colors * 3 components
            // Map JSON colors to the data range
            for (int i = 0; i < 9; i++) {
                float value_ratio = float(i) / 8.0f; // 0 to 1
                double current_val = range[0] + value_ratio * (range[1] - range[0]);
                colorTF->AddRGBPoint(current_val, json_colors[i*3], json_colors[i*3+1], json_colors[i*3+2]);
            }
            std::cout << "Applied JSON ocean colormap for range [" 
                      << range[0] << ", " << range[1] << "]" << std::endl;
        } else {
            // Fallback to hardcoded ocean colormap
            colorTF->AddRGBPoint(0.0, 1.0, 1.0, 1.0); // White for land (will be invisible due to isosurface)
            colorTF->AddRGBPoint(6.3, 0.933, 0.957, 0.980);
            colorTF->AddRGBPoint(8.6, 0.839, 0.886, 0.949);
            colorTF->AddRGBPoint(10.8, 0.722, 0.820, 0.898);
            colorTF->AddRGBPoint(13.1, 0.553, 0.718, 0.843);
            colorTF->AddRGBPoint(15.3, 0.392, 0.600, 0.780);
            colorTF->AddRGBPoint(17.6, 0.259, 0.463, 0.706);
            colorTF->AddRGBPoint(19.8, 0.157, 0.333, 0.712);
            colorTF->AddRGBPoint(range[1], 0.086, 0.192, 0.620);
            
            std::cout << "Applied hardcoded 'Ocean' colormap (clipped): Land=transparent, Ocean=Ocean (0.5-" 
                      << range[1] << ")" << std::endl;
        }
                  
    } else if (field_type == "temperature") {
        double mid_temp = (range[0] + range[1]) / 2.0;
        colorTF->AddRGBPoint(range[0], 0.0, 0.0, 1.0);
        colorTF->AddRGBPoint(mid_temp, 0.0, 1.0, 0.0);
        colorTF->AddRGBPoint(range[1], 1.0, 0.0, 0.0);
        
        std::cout << "Applied temperature colormap for range [" 
                  << range[0] << ", " << range[1] << "]" << std::endl;
                  
    } else {
        colorTF->AddRGBPoint(range[0], 0.2, 0.4, 0.7);
        colorTF->AddRGBPoint(range[1], 0.8, 0.9, 1.0);
        
        std::cout << "Applied default colormap for range [" 
                  << range[0] << ", " << range[1] << "]" << std::endl;
    }
}

void setupDepthBasedOpacity(vtkPiecewiseFunction* opacityTF, double* range, const std::vector<float>& json_opacities) {
    opacityTF->RemoveAllPoints();
    
    if (!json_opacities.empty() && json_opacities.size() >= 9) {
        // Use JSON opacities mapped to the data range
        for (int i = 0; i < json_opacities.size(); i++) {
            float value_ratio = float(i) / float(json_opacities.size() - 1); // 0 to 1
            double current_val = range[0] + value_ratio * (range[1] - range[0]);
            opacityTF->AddPoint(current_val, json_opacities[i]);
        }
        std::cout << "Applied JSON ocean opacity for range [" 
                  << range[0] << ", " << range[1] << "]" << std::endl;
    } else {
        // Fallback to hardcoded opacity values
        // Make land areas completely transparent (isosurface will handle land)
        opacityTF->AddPoint(0.0, 0.0);      // Transparent land
        opacityTF->AddPoint(MAX_LAND_SALINITY, 0.0);      // Transparent land boundary
        
        // OCEAN AREAS - Much more transparent for semi-transparent effect
        opacityTF->AddPoint(1.0, 0.01);     // Very transparent coastal water
        opacityTF->AddPoint(5.0, 0.05);     // Very transparent low salinity ocean
        opacityTF->AddPoint(10.0, 0.08);    // Still very transparent medium salinity
        opacityTF->AddPoint(15.0, 0.2);    // Slightly more visible high salinity
        opacityTF->AddPoint(20.0, 0.24);    // Semi-transparent very high salinity
        opacityTF->AddPoint(range[1], 0.33); // Maximum transparency at max salinity
        
        std::cout << "Applied hardcoded very transparent ocean opacity: Land=transparent, Ocean=semi-transparent (0.05-0.20) for range [" 
                  << range[0] << ", " << range[1] << "]" << std::endl;
    }
}

void setupStreamlineColormap(vtkLookupTable* lut, const std::string& /*scalar_type*/) {
    lut->SetNumberOfColors(256);
    
    // Make all streamlines white for clear visibility against land and ocean
    lut->SetHueRange(0.0, 0.0);        // No hue variation
    lut->SetSaturationRange(0.0, 0.0); // No saturation = white
    lut->SetValueRange(1.0, 1.0);      // Full brightness = white
    
    lut->Build();
    
    std::cout << "Applied white streamline colormap for clear visibility" << std::endl;
}

bool detectLandArea(double salinity_value, double temperature_value = 0.0) {
    // Updated for your specific 0-22 PSU data range
    // Land areas have zero or very low salinity
    bool is_land = (salinity_value <= MAX_LAND_SALINITY);  // Land is salinity ≈ 0
    
    if (is_land) {
        std::cout << "Detected land area: salinity=" << salinity_value << std::endl;
    }
    
    return is_land;
}

void addOceanBackground(vtkRenderer* ren) {
    //  black background - 
    vtkNew<vtkNamedColors> colors;
    ren->SetBackground(colors->GetColor3d("black").GetData());
    
    std::cout << "Applied familiar black background" << std::endl;
}



void printVisualizationInfo(double* scalar_range, double* velocity_range = nullptr) {
    std::cout << "\n=== VISUALIZATION INFO ===" << std::endl;
    std::cout << "Scalar field range: [" << scalar_range[0] << ", " << scalar_range[1] << "]" << std::endl;
    
    if (velocity_range) {
        std::cout << "Velocity magnitude range: [" << velocity_range[0] << ", " << velocity_range[1] << "]" << std::endl;
    }
    
    // Provide context about typical oceanographic values
    std::cout << "\nOceanographic context:" << std::endl;
    std::cout << "- Your data range: 0-" << scalar_range[1] << " PSU" << std::endl;
    std::cout << "- Land areas: salinity ≈ 0 PSU" << std::endl;
    std::cout << "- Ocean areas: salinity > 0.5 PSU" << std::endl;
    std::cout << "- Typical ocean salinity: 34-37 PSU" << std::endl;
    std::cout << "========================\n" << std::endl;
}

// =============================================================================
// ORIGINAL FUNCTIONS WITH IMPROVEMENTS
// =============================================================================
void loadTransferFunction2(json &j, vtkVolumeProperty *volumeProperty)
{
    // Create transfer mapping scalar value to opacity
    vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
    // Create transfer mapping scalar value to color
    vtkNew<vtkColorTransferFunction> colorTransferFunction;
    
    std::vector<float> colors = j["transferFunc"]["colors"].get<std::vector<float>>();
    std::vector<float> opacities = j["transferFunc"]["opacities"].get<std::vector<float>>();
    glm::vec2 tfRange = get_vec2f(j["transferFunc"]["range"]);
    

    for (uint32_t i=0; i<colors.size()/3; i++){
        float current_val = tfRange[0] + float(i)/(colors.size()/3) * (tfRange[1] - tfRange[0]);
        opacityTransferFunction->AddPoint(current_val, opacities[i]);
        colorTransferFunction->AddRGBPoint(current_val, colors[3*i], colors[i*3+1], colors[i*3+2]);
    }
    std::cout << "load tf sz= "<< colors.size()/3<<" \n";

    volumeProperty->SetColor(colorTransferFunction);
    volumeProperty->SetScalarOpacity(opacityTransferFunction);
    //volumeProperty->ShadeOn();
    volumeProperty->SetInterpolationTypeToLinear();
}

// void loadCamera2(AnimatorKF &keyframe, vtkRenderer *ren1)
// {
//     // load camera
//     double world_scale_xy = 1;
//     for (size_t i=0; i<2; i++)
//         world_scale_xy = std::min(world_scale_xy, double(keyframe.get_bbox()[i]/keyframe.get_max_dims()[i]));
//     double cam_scale =  1/world_scale_xy;
    
//     visuser::Camera c;
//     keyframe.get_current_cam(c);
//     auto camera = ren1->GetActiveCamera();
//     double eyePos[3] = {c.pos[0]*cam_scale, c.pos[1]*cam_scale, c.pos[2]*cam_scale };
//     double focalPoint[3] = {eyePos[0] + c.dir[0]*cam_scale*60,
//         eyePos[1] + c.dir[1]*cam_scale*60,
//         eyePos[2] + c.dir[2]*cam_scale*60};

//     std::cout <<"load pos:"<< c.pos[0] <<" "<<c.pos[1] <<" "<< c.pos[2]<<"\n";
//     std::cout <<"load dir:"<< c.dir[0] <<" "<<c.dir[1] <<" "<< c.dir[2]<<"\n";
//     std::cout <<"load up:"<< c.up[0] <<" "<<c.up[1] <<" "<< c.up[2]<<"\n";
//     std::cout << "load scale:"<< cam_scale << "\n";
//     camera->SetPosition(eyePos[0], eyePos[1], eyePos[2]);
//     camera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
//     camera->SetViewUp(c.up[0], c.up[1], c.up[2]);
//     ren1->ResetCameraClippingRange();
// }

void loadCamera2(AnimatorKF &keyframe, vtkRenderer *ren1)
{
    // load camera
    double world_scale_xy = 1;
    for (size_t i=0; i<2; i++)
        world_scale_xy = std::min(world_scale_xy, double(keyframe.get_bbox()[i]/keyframe.get_max_dims()[i]));
    double cam_scale =  1/world_scale_xy;
    
    visuser::Camera c;
    keyframe.get_current_cam(c);
    auto camera = ren1->GetActiveCamera();
    // double eyePos[3] = {-300, -100, 126};        // Directly above data center, high Z
    // double focalPoint[3] = {11, 90, 126};
    double eyePos[3] = {-300, -150, 145};        // Directly above data center, high Z
    double focalPoint[3] = {11, 80, 145};
    camera->SetPosition(eyePos[0], eyePos[1], eyePos[2]);
    camera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
    camera->SetViewUp(0, 1, 0);
    // ren1->ResetCameraClippingRange();
}

void loadKF2(Animator &animator,
        uint32_t idx,
        vtkImageData *img,
        vtkRenderer *ren1)
{
    bool loadVol = false;
    bool loadDerived = false;
    ren1->RemoveAllViewProps();
    
    for (int j = 0; j < animator.kfs[idx].get_data_list_size(); ++j) {
        int dim[3];
        std::string file_name;
        double spc[3] = {1.0, 1.0, 1.0};
        
        // The property describes how the data will look
        vtkNew<vtkVolumeProperty> volumeProperty;
        json data = animator.get_scene_data(idx, j);
        json info = animator.get_scene_info(idx, j);
        
        if (data["type"] == "streamline"){
            std::cout << "\n=== PROCESSING STREAMLINE DATA ===" << std::endl;
            
            //
            // Read data
            //
            file_name = data["src"]["name"];
            vtkNew<vtkStructuredGridReader> reader;
            reader->SetFileName(file_name.c_str());
            reader->Update(); // force a read to occur
            reader->GetOutput()->GetLength();
            vtkNew<vtkNamedColors> colors;
            double range[2];

            reader->GetOutput()->GetPointData()->GetScalars()->GetRange(range);
            std::cout << "Scalar range: " << range[0] << " to " << range[1] << std::endl;

            // Print active scalar field name
            vtkDataArray* activeScalars = reader->GetOutput()->GetPointData()->GetScalars();
            std::cout << "Active scalar field name: " << 
                (activeScalars ? activeScalars->GetName() : "unnamed") << std::endl;

            // DEBUG: Print all available data arrays
            std::cout << "\n=== VTK FILE DATA ARRAYS ===" << std::endl;
            vtkPointData* pointData = reader->GetOutput()->GetPointData();
            for (int i = 0; i < pointData->GetNumberOfArrays(); i++) {
                vtkDataArray* array = pointData->GetArray(i);
                std::cout << "Array " << i << ": " << (array->GetName() ? array->GetName() : "unnamed")
                          << " (components: " << array->GetNumberOfComponents() << ")" << std::endl;
            }
            std::cout << "========================\n" << std::endl;

            // Check for velocity vectors and calculate velocity range
            double velocityRange[2] = {0.0, 0.0};
            vtkDataArray* velocityVectors = reader->GetOutput()->GetPointData()->GetVectors();
            if (velocityVectors) {
                std::cout << "✓ FOUND velocity vectors with name: " << velocityVectors->GetName() << std::endl;
                
                // Get the velocity magnitude range
                velocityRange[0] = DBL_MAX;
                velocityRange[1] = -DBL_MAX;
                int pointCount = reader->GetOutput()->GetNumberOfPoints();
                
                for (int i = 0; i < pointCount; i++) {
                    double velocity[3];
                    velocityVectors->GetTuple(i, velocity);
                    double magnitude = sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
                    
                    velocityRange[0] = std::min(velocityRange[0], magnitude);
                    velocityRange[1] = std::max(velocityRange[1], magnitude);
                }
                
                std::cout << "✓ Velocity magnitude range: " << velocityRange[0] << " to " << velocityRange[1] << std::endl;
            } else {
                std::cout << "⚠️  WARNING: NO VELOCITY VECTORS FOUND!" << std::endl;
            }

            // Print visualization info
            // printVisualizationInfo(range, velocityVectors ? velocityRange : nullptr);

            //
            // Outline around data.
            //
            vtkNew<vtkStructuredGridOutlineFilter> outlineF;
            outlineF->SetInputConnection(reader->GetOutputPort());
            vtkNew<vtkPolyDataMapper> outlineMapper;
            outlineMapper->SetInputConnection(outlineF->GetOutputPort());
            vtkNew<vtkActor> outline;
            outline->SetMapper(outlineMapper);
            outline->GetProperty()->SetColor(colors->GetColor3d("DarkSlateGray").GetData());
            outline->GetProperty()->SetLineWidth(0.3);

            // ===== VOLUME RENDERING =====
            
            // Read scalar field type and transfer function data from JSON
            std::string scalar_field_type = animator.kfs[idx].get_scalar_field();
            std::vector<float> json_colors, json_opacities;
            
            // Get colors and opacities from JSON if available
            if (info.contains("transferFunc")) {
                if (info["transferFunc"].contains("colors")) {
                    json_colors = info["transferFunc"]["colors"].get<std::vector<float>>();
                }
                if (info["transferFunc"].contains("opacities")) {
                    json_opacities = info["transferFunc"]["opacities"].get<std::vector<float>>();
                }
            }
            
            std::cout << "Using scalar field type: " << scalar_field_type << std::endl;
            std::cout << "JSON colors size: " << json_colors.size() << ", opacities size: " << json_opacities.size() << std::endl;
            
            // Create improved opacity transfer function
            vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
            setupDepthBasedOpacity(opacityTransferFunction, range, json_opacities);

            // Create improved color transfer function  
            vtkNew<vtkColorTransferFunction> colorTransferFunction;
            setupOceanicColormap(colorTransferFunction, range, scalar_field_type, json_colors);

            // The property describes how the data will look
            vtkNew<vtkVolumeProperty> volumeProperty;
            volumeProperty->SetColor(colorTransferFunction);
            volumeProperty->SetScalarOpacity(opacityTransferFunction);
            volumeProperty->ShadeOn();  // Enable shading for better 3D effect
            volumeProperty->SetInterpolationTypeToLinear();
            volumeProperty->SetAmbient(0.2);    // Subtle ambient lighting
            volumeProperty->SetDiffuse(0.7);    // Main diffuse lighting
            volumeProperty->SetSpecular(0.1);   // Slight specular highlights
        
            vtkNew<vtkImageData> imageData;
            imageData->GetPointData()->SetScalars(reader->GetOutput()->GetPointData()->GetArray(0));
            int dims[3];
            reader->GetOutput()->GetDimensions(dims);
            std::cout << "Data dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
            imageData->SetDimensions(dims);
            imageData->SetSpacing(1.0, 1.0, 1.0); 
            imageData->SetOrigin(0.0, 0.0, 0.0);

            // The mapper / ray cast function know how to render the data
            vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
            volumeMapper->SetInputData(imageData);

            // The volume holds the mapper and the property and
            // can be used to position/orient the volume
            vtkNew<vtkVolume> volume;
            volume->SetMapper(volumeMapper);
            volume->SetProperty(volumeProperty);

            // ===== STREAMLINES WITH BETTER DENSITY AND VISIBILITY =====
            
            // Create a plane for streamline seeding with higher density
            auto ext = outlineMapper->GetBounds();
            std::cout << "Data bounds: "
                      << "X[" << ext[0] << ", " << ext[1] << "], "
                      << "Y[" << ext[2] << ", " << ext[3] << "], "
                      << "Z[" << ext[4] << ", " << ext[5] << "]" << std::endl;
            double bblow[3] = {ext[0], ext[2], ext[4]};
            double bblen[3] = {(ext[1] - ext[0]), (ext[3] - ext[2]), (ext[5] - ext[4])};
            std::cout << "BB Low: " << bblow[0] << ", " << bblow[1] << ", " << bblow[2] << std::endl;
            std::cout << "BB Len: " << bblen[0] << ", " << bblen[1] << ", " << bblen[2] << std::endl;
        
            vtkNew<vtkPlaneSource> source;
            source->SetOrigin(bblow[0]+ bblen[0]/4, bblow[1], bblow[2]);
            source->SetPoint1(bblow[0]+ bblen[0]/4, bblow[1] + bblen[1], bblow[2]);
            source->SetPoint2(bblow[0]+ bblen[0]/4, bblow[1], bblow[2]+ bblen[2]);
            source->SetXResolution(20);
            source->SetYResolution(20); 
            source->Update();
        
            vtkPolyData* poly = source->GetOutput();
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputData(poly);

            vtkNew<vtkActor> actor;
            actor->SetMapper(mapper);
            actor->GetProperty()->SetColor(colors->GetColor3d("LightGray").GetData());
            actor->GetProperty()->SetOpacity(1.0);  // Make seed plane fully opaque

            vtkNew<vtkStreamTracer> streamers;
            streamers->SetInputConnection(reader->GetOutputPort());
            streamers->SetSourceConnection(source->GetOutputPort());
            streamers->SetMaximumPropagation(200);   // Slightly longer propagation (was 200)
            streamers->SetInitialIntegrationStep(.3); // Slightly smaller step for smoother lines (was .3)
            streamers->SetMinimumIntegrationStep(.05); // Smaller minimum step (was .05)
            streamers->SetIntegratorType(2);
            streamers->SetIntegrationDirectionToBoth();
            
            // CRITICAL: Check if we have velocity vectors for proper streamlines
            if (velocityVectors) {
                std::cout << "✓ Using velocity vectors for streamlines - will show eddy flow correctly" << std::endl;
            } else {
                std::cout << "⚠️  WARNING: No velocity vectors - streamlines will be based on scalar gradient only!" << std::endl;
                std::cout << "⚠️  Run: python test-vtk.py -save  to generate VTK files with velocity data!" << std::endl;
            }
            
            streamers->Update();

            // Create white color lookup table for streamlines
            vtkNew<vtkLookupTable> whiteLut;
            whiteLut->SetNumberOfColors(256);
            whiteLut->SetHueRange(0.0, 0.0);        // No hue variation
            whiteLut->SetSaturationRange(0.0, 0.0); // No saturation = white
            whiteLut->SetValueRange(1.0, 1.0);      // Full brightness = white
            whiteLut->Build();
            
            // ===== STREAMLINE LINES =====
            vtkNew<vtkPolyDataMapper> streamLineMapper;
            streamLineMapper->SetInputConnection(streamers->GetOutputPort());
            streamLineMapper->SetLookupTable(whiteLut);
            streamLineMapper->ScalarVisibilityOff();  // Use solid white color, not scalar coloring

            vtkNew<vtkActor> streamLineActor;
            streamLineActor->SetMapper(streamLineMapper);
            streamLineActor->VisibilityOn();
            
            // Set white color properties for streamlines
            streamLineActor->GetProperty()->SetColor(1.0, 1.0, 1.0);  // White
            streamLineActor->GetProperty()->SetLineWidth(1.5);         // Thicker lines
            streamLineActor->GetProperty()->SetOpacity(1.0);           // Start fully opaque at seed points
            streamLineActor->GetProperty()->SetSpecular(0.1);          // Slight shininess
            streamLineActor->GetProperty()->SetSpecularPower(10);      // Focused specular highlight
            streamLineActor->GetProperty()->SetRenderLinesAsTubes(true);
            
            // Add all actors to the scene
            ren1->AddVolume(volume);
            
            // ADD LAND ISOSURFACE HERE
            createLandIsosurface(imageData, ren1, range);
            
            ren1->AddActor(outline);
            ren1->AddActor(streamLineActor);  // White streamlines
            // Optionally add seed plane for debugging: ren1->AddActor(actor);

            std::cout << "=== STREAMLINE RENDERING COMPLETE ===\n" << std::endl;
            break;
        }
        
        if (data["type"] == "structured"){
            // load only one volume for now
            if (!loadVol) loadVol = true;
            else break;
            // load data info
            glm::vec3 data_dims = get_vec3f(data["src"]["dims"]);
            dim[0] = data_dims[0];
            dim[1] = data_dims[1];
            dim[2] = data_dims[2];
            std::cout << "Data dimensions: "
                      << "X[" << dim[0] << "], "
                      << "Y[" << dim[1] << "], "
                      << "Z[" << dim[2] << "]" << std::endl;

            file_name = data["src"]["name"];
            // load scene info  
            loadTransferFunction2(info, volumeProperty);
        }
        
        //vtkNew<vtkImageData> img;
        img->SetDimensions(dim);
        img->AllocateScalars(VTK_INT, 1);
      
        std::cout <<"Loading volume "<<file_name <<" with dim "<<dim[0]<<" "<<dim[1]<<" "<<dim[2]<<"\n";
        std::fstream file;
        file.open(file_name, std::fstream::in | std::fstream::binary);
        for (int z = 0; z < dim[2]; ++z)
            for (int y = 0; y < dim[1]; ++y)
                for (int x = 0; x < dim[0]; ++x)
                    {
                    float buff;
                    file.read((char*)(&buff), sizeof(buff));
                    img->SetScalarComponentFromDouble(x, y, z, 0, buff);
                    }

        file.close();
        std::cout<< "End load\n";

        img->SetSpacing(spc);

        // The mapper / ray cast function know how to render the data
        vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
        volumeMapper->SetInputData(img);

        // The volume holds the mapper and the property and
        // can be used to position/orient the volume
        vtkNew<vtkVolume> volume;
        volume->SetMapper(volumeMapper);
        volume->SetProperty(volumeProperty);
        volume->SetPosition(0, 0, 0);

        // Create the standard renderer, render window and interactor
        ren1->AddVolume(volume);
        
        // ADD LAND ISOSURFACE FOR STRUCTURED DATA TOO
        createLandIsosurface(img, ren1, img->GetScalarRange());
    }
    
    // Apply improved ocean background
    addOceanBackground(ren1);
    
    loadCamera2(animator.kfs[idx], ren1);
}

void writeImage2(std::string const& fileName, vtkRenderWindow* renWin, bool rgba)
{
    if (!fileName.empty())
    {
        std::string fn = fileName;
        auto writer = vtkSmartPointer<vtkPNGWriter>::New();
        
        vtkNew<vtkWindowToImageFilter> window_to_image_filter;
        window_to_image_filter->SetInput(renWin);
        window_to_image_filter->SetScale(1); // image quality
        if (rgba) window_to_image_filter->SetInputBufferTypeToRGBA();	
        else window_to_image_filter->SetInputBufferTypeToRGB();
        
        // Read from the front buffer.
        window_to_image_filter->ReadFrontBufferOff();
        window_to_image_filter->Update();

        writer->SetFileName(fn.c_str());
        writer->SetInputConnection(window_to_image_filter->GetOutputPort());
        writer->Write();
        
        std::cout << "Saved improved visualization to: " << fn << std::endl;
    }
    else std::cerr << "No filename provided." << std::endl;

    return;
}

void run2(std::string jsonStr, std::string output_dir, int header_sel){

    std::cout << "\n=== STARTING VTK RENDERING ===" << std::endl;

    // Create output directory if it doesn't exist
    if (!output_dir.empty()) {
        if (!std::filesystem::exists(output_dir)) {
            std::cout << "Creating output directory: " << output_dir << std::endl;
            std::filesystem::create_directories(output_dir);
        }
    }
    
    // load json
    std::cout << "\n\nStart json loading ... \n";
    
    Animator animator;
    animator.init(jsonStr.c_str());
    
    std::cout << "\nEnd json loading ... \n\n";
    
    vtkNew<vtkRenderer> ren1;
    
    vtkNew<vtkRenderWindow> renWin;
    renWin->AddRenderer(ren1);
    
    vtkNew<vtkRenderWindowInteractor> iren;
    iren->SetRenderWindow(renWin);
    
    // Create the reader for the data
    vtkNew<vtkImageData> img;
    loadKF2(animator, 0, img, ren1);
    
    // Set higher resolution for better quality
    renWin->SetSize(1200, 900);
    renWin->SetWindowName("Oceanographic Visualization");
    renWin->Render();
    
    if (header_sel >= 0){ // render selected keyframe
        // save file with output directory
        std::string outname;
        if (!output_dir.empty()) {
            outname = output_dir + "/" + getOutName2("", header_sel);
        } else {
            outname = getOutName2("", header_sel);
        }
        writeImage2(outname, renWin, false);
    } else {
        // reload widget for each key frame
        for (int kf_idx=0; kf_idx<animator.kfs.size(); kf_idx++){
            loadKF2(animator, kf_idx, img, ren1);
            if (header_sel == -1){ // render key frames
                renWin->Render();
                // save file with output directory
                std::string outname;
                if (!output_dir.empty()) {
                    outname = output_dir + "/" + getOutName2("", kf_idx);
                } else {
                    outname = getOutName2("", kf_idx);
                }
                writeImage2(outname, renWin, false);
            } else if (header_sel == -2){ // renderAllFrames
                glm::vec2 frameRange = animator.kfs[kf_idx].get_fRange();
                for (int f = frameRange[0]; f <= frameRange[1]; f++){
                    renWin->Render();
                    std::string outname;
                    if (!output_dir.empty()) {
                        outname = output_dir + "/img_" + std::to_string(kf_idx) + "_f" + std::to_string(f) + ".png";
                    } else {
                        outname = "img_" + std::to_string(kf_idx) + "_f" + std::to_string(f) + ".png";
                    }
                    writeImage2(outname, renWin, false);
                    if (f < frameRange[1]){
                        // advance frame 
                        animator.kfs[kf_idx].advanceFrame();
                        // load camera
                        loadCamera2(animator.kfs[kf_idx], ren1);
                    }
                }
            }
        }
    }
    
    std::cout << "=== VTK VISUALIZATION COMPLETE ===" << std::endl;
}
