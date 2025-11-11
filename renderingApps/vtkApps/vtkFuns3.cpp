#include <filesystem>
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
#include <vtkStructuredGrid.h>
#include <vtkStructuredGridReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkProbeFilter.h>
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
#include "vtkFuns3.h"
#include <vtkPNGReader.h>
#include <vtkTexture.h>

using namespace visuser;
using namespace nlohmann_loader;

std::string getOutName2(std::string str, uint32_t idx)
{
    std::string base_filename = str.substr(str.find_last_of("/\\") + 1);
    std::string outname = base_filename.substr(0, base_filename.find_last_of("."));
    return "img_"+outname+"_kf"+std::to_string(idx)+".png";
}

// =============================================================================
// CONFIGURATION-DRIVEN HELPER FUNCTIONS
// =============================================================================

void setupVolumeFromJSON(vtkImageData* imageData, vtkRenderer* ren1, double* range, const json& volume_config, 
                        const std::vector<float>& tf_colors, const std::vector<float>& tf_opacities) {
    if (!volume_config.contains("enabled") || !volume_config["enabled"].get<bool>()) {
        std::cout << "Volume representation disabled, skipping..." << std::endl;
        return;
    }

    std::cout << "\n=== CREATING VOLUME RENDERING ===" << std::endl;

  
    double tf_range[2] = { range[0], range[1] };
    
    if (volume_config.contains("scene_info") && volume_config["scene_info"].contains("transferFunc")) {
        auto tf = volume_config["scene_info"]["transferFunc"];
        if (tf.contains("range")) {
            // use explicit transfer function range if present in JSON
            glm::vec2 r = get_vec2f(tf["range"]);
            tf_range[0] = r[0];
            tf_range[1] = r[1];
        }

    }

    // Create opacity transfer function
    vtkNew<vtkPiecewiseFunction> opacityTransferFunction;
    opacityTransferFunction->RemoveAllPoints();
    
   
    for (size_t i = 0; i < tf_opacities.size(); i++) {
        // float value_ratio = float(i) / float(tf_opacities.size() - 1);
        double current_val = tf_range[0] + tf_colors[i*4+0] * (tf_range[1] - tf_range[0]);
        opacityTransferFunction->AddPoint(current_val, tf_opacities[i]);
    }
    

    // Create color transfer function
    vtkNew<vtkColorTransferFunction> colorTransferFunction;
    colorTransferFunction->RemoveAllPoints();

    size_t nColors = tf_colors.size() / 4; // assuming tf_colors is flat RGBA array
    
    for (size_t i = 0; i < nColors; i++) {
        // float value_ratio = float(i) / float(nColors - 1);
        double current_val = tf_range[0] + tf_colors[i*4+0] * (tf_range[1] - tf_range[0]);
        colorTransferFunction->AddRGBPoint(current_val, tf_colors[i*4+1], tf_colors[i*4+2], tf_colors[i*4+3]);
    }

    // Debug: print transfer function summary
    std::cout << "[DEBUG] Volume TF range: [" << tf_range[0] << ", " << tf_range[1] << "]" << std::endl;
    std::cout << "[DEBUG] Color TF points: " << colorTransferFunction->GetSize() << ", Opacity TF points: " << opacityTransferFunction->GetSize() << std::endl;

    // Configure volume property
    vtkNew<vtkVolumeProperty> volumeProperty;
    volumeProperty->SetColor(colorTransferFunction);
    volumeProperty->SetScalarOpacity(opacityTransferFunction);
    volumeProperty->SetInterpolationTypeToLinear();
    
    // Apply volume properties from config
    if (volume_config.contains("scene_info") && volume_config["scene_info"].contains("volumeProperties")) {
        auto props = volume_config["scene_info"]["volumeProperties"];
        if (props.contains("shadeOn") && props["shadeOn"].get<bool>()) {
            volumeProperty->ShadeOn();
        }
        if (props.contains("ambient")) {
            volumeProperty->SetAmbient(props["ambient"].get<double>());
        }
        if (props.contains("diffuse")) {
            volumeProperty->SetDiffuse(props["diffuse"].get<double>());
        }
        if (props.contains("specular")) {
            volumeProperty->SetSpecular(props["specular"].get<double>());
        }
        if (props.contains("specularPower")) {
            volumeProperty->SetSpecularPower(props["specularPower"].get<double>());
        }
        if (props.contains("scalarOpacityUnitDistance")) {
            volumeProperty->SetScalarOpacityUnitDistance(props["scalarOpacityUnitDistance"].get<double>());
        }
        if (props.contains("independentComponents")) {
            volumeProperty->SetIndependentComponents(props["independentComponents"].get<bool>());
        }
    }

    // Create volume mapper
    vtkNew<vtkFixedPointVolumeRayCastMapper> volumeMapper;
    volumeMapper->SetInputData(imageData);
    // // CRITICAL: This tells the mapper to map your data's range 
    // // (e.g., -0.4 to 29) onto the color map's range (0 to 1).
    // volumeMapper->SetScalarRange(range[0], range[1]);
    
    // Apply mapper properties from config (look under scene_info if present)
    if (volume_config.contains("scene_info") && volume_config["scene_info"].contains("mapperProperties")) {
        auto mapper_props = volume_config["scene_info"]["mapperProperties"];
        if (mapper_props.contains("sampleDistance")) {
            volumeMapper->SetSampleDistance(mapper_props["sampleDistance"].get<double>());
        }
        if (mapper_props.contains("imageSampleDistance")) {
            volumeMapper->SetImageSampleDistance(mapper_props["imageSampleDistance"].get<double>());
        }
        if (mapper_props.contains("autoAdjustSampleDistances")) {
            volumeMapper->SetAutoAdjustSampleDistances(mapper_props["autoAdjustSampleDistances"].get<bool>());
        }
        if (mapper_props.contains("maximumImageSampleDistance")) {
            volumeMapper->SetMaximumImageSampleDistance(mapper_props["maximumImageSampleDistance"].get<double>());
        }
        if (mapper_props.contains("blendMode")) {
            std::string bm = mapper_props["blendMode"].get<std::string>();
            if (bm == "composite") {
                volumeMapper->SetBlendModeToComposite();
            } else if (bm == "maximum") {
                volumeMapper->SetBlendModeToMaximumIntensity();
            }
        }
    }

    // Create and add volume
    vtkNew<vtkVolume> volume;
    volume->SetMapper(volumeMapper);
    volume->SetProperty(volumeProperty);
    
    ren1->AddVolume(volume);
    std::cout << "[DEBUG] Added vtkVolume to renderer" << std::endl;
    // If mapper props were present, also print a short summary
    if (volume_config.contains("scene_info") && volume_config["scene_info"].contains("mapperProperties")) {
        auto mapper_props = volume_config["scene_info"]["mapperProperties"];
        std::cout << "[DEBUG] mapperProperties present:";
        if (mapper_props.contains("sampleDistance")) std::cout << " sampleDistance=" << mapper_props["sampleDistance"].get<double>();
        if (mapper_props.contains("imageSampleDistance")) std::cout << " imageSampleDistance=" << mapper_props["imageSampleDistance"].get<double>();
        if (mapper_props.contains("autoAdjustSampleDistances")) std::cout << " autoAdjust=" << mapper_props["autoAdjustSampleDistances"].get<bool>();
        if (mapper_props.contains("maximumImageSampleDistance")) std::cout << " maxImageSampleDistance=" << mapper_props["maximumImageSampleDistance"].get<double>();
        if (mapper_props.contains("blendMode")) std::cout << " blendMode=" << mapper_props["blendMode"].get<std::string>();
        std::cout << std::endl;
    }
    std::cout << "=== VOLUME RENDERING COMPLETE ===\n" << std::endl;
}

void setupStreamlinesFromJSON(vtkStructuredGridReader* reader, vtkRenderer* ren1, const json& rep) {
    if (!rep.contains("enabled") || !rep["enabled"].get<bool>()) {
        std::cout << "Streamline representation disabled, skipping..." << std::endl;
        return;
    }
    
    if (!rep.contains("scene_info")) {
        std::cout << "No scene_info found for streamline representation" << std::endl;
        return;
    }

    const json& streamline_config = rep["scene_info"];
    std::cout << "\n=== CREATING STREAMLINES ===" << std::endl;

    vtkNew<vtkNamedColors> colors;

    // Create outline
    vtkNew<vtkStructuredGridOutlineFilter> outlineF;
    outlineF->SetInputConnection(reader->GetOutputPort());
    vtkNew<vtkPolyDataMapper> outlineMapper;
    outlineMapper->SetInputConnection(outlineF->GetOutputPort());
    
    auto ext = outlineMapper->GetBounds();
    double bblow[3] = {ext[0], ext[2], ext[4]};
    double bblen[3] = {(ext[1] - ext[0]), (ext[3] - ext[2]), (ext[5] - ext[4])};

    // Create seed plane from config
    vtkNew<vtkPlaneSource> source;
    
    if (streamline_config.contains("seedPlane") && streamline_config["seedPlane"].contains("enabled") && 
        streamline_config["seedPlane"]["enabled"].get<bool>()) {
        auto seed_config = streamline_config["seedPlane"];
        
        double pos_fraction = seed_config.contains("positionFraction") ? 
                             seed_config["positionFraction"].get<double>() : 0.25;
        
        source->SetOrigin(bblow[0] + bblen[0] * pos_fraction, bblow[1], bblow[2]);
        source->SetPoint1(bblow[0] + bblen[0] * pos_fraction, bblow[1] + bblen[1], bblow[2]);
        source->SetPoint2(bblow[0] + bblen[0] * pos_fraction, bblow[1], bblow[2] + bblen[2]);
        
        int xRes = seed_config.contains("xResolution") ? seed_config["xResolution"].get<int>() : 20;
        int yRes = seed_config.contains("yResolution") ? seed_config["yResolution"].get<int>() : 20;
        source->SetXResolution(xRes);
        source->SetYResolution(yRes);
    }
    source->Update();

    // Create streamline tracer
    vtkNew<vtkStreamTracer> streamers;
    streamers->SetInputConnection(reader->GetOutputPort());
    streamers->SetSourceConnection(source->GetOutputPort());
    
    // Apply integration properties from config
    if (streamline_config.contains("integrationProperties")) {
        auto int_props = streamline_config["integrationProperties"];
        if (int_props.contains("maxPropagation")) {
            streamers->SetMaximumPropagation(int_props["maxPropagation"].get<double>());
        }
        if (int_props.contains("initialIntegrationStep")) {
            streamers->SetInitialIntegrationStep(int_props["initialIntegrationStep"].get<double>());
        }
        if (int_props.contains("minimumIntegrationStep")) {
            streamers->SetMinimumIntegrationStep(int_props["minimumIntegrationStep"].get<double>());
        }
        if (int_props.contains("integratorType")) {
            streamers->SetIntegratorType(int_props["integratorType"].get<int>());
        }
        if (int_props.contains("integrationDirection")) {
            std::string dir = int_props["integrationDirection"].get<std::string>();
            if (dir == "both") streamers->SetIntegrationDirectionToBoth();
            else if (dir == "forward") streamers->SetIntegrationDirectionToForward();
            else if (dir == "backward") streamers->SetIntegrationDirectionToBackward();
        }
    }
    
    streamers->Update();

    // Create streamline mapper and actor
    vtkNew<vtkPolyDataMapper> streamLineMapper;
    streamLineMapper->SetInputConnection(streamers->GetOutputPort());
    streamLineMapper->ScalarVisibilityOn();

    // Synchronize streamline coloring with volume scalar field
    vtkPolyData* streamPoly = streamers->GetOutput();
    // Probe the volume scalars onto the streamlines
    vtkImageData* volumeImage = nullptr;
    if (reader->GetOutput() && reader->GetOutput()->GetPointData() && reader->GetOutput()->GetPointData()->GetScalars()) {
        // Try to get the image data from the reader (if available)
        volumeImage = vtkImageData::SafeDownCast(reader->GetOutput());
    }
    if (volumeImage) {
        vtkNew<vtkProbeFilter> probe;
        probe->SetSourceData(volumeImage);
        probe->SetInputData(streamPoly);
        probe->Update();
        vtkPolyData* probedPoly = vtkPolyData::SafeDownCast(probe->GetOutput());
        if (probedPoly && probedPoly->GetPointData()->GetScalars()) {
            streamPoly->GetPointData()->SetScalars(probedPoly->GetPointData()->GetScalars());
        }
    }
    double scalarRange[2] = {0.0, 1.0};
    if (streamPoly && streamPoly->GetPointData() && streamPoly->GetPointData()->GetScalars()) {
        streamPoly->GetPointData()->GetScalars()->GetRange(scalarRange);
        std::cout << "[DEBUG] Streamline scalar range for coloring: [" << scalarRange[0] << ", " << scalarRange[1] << "]" << std::endl;
    } else {
        std::cout << "[DEBUG] No scalars found on streamlines for coloring." << std::endl;
    }
    vtkNew<vtkLookupTable> lut;
    lut->SetNumberOfTableValues(256);
    // Read LUT range from config if present
    double lut_min = 34.0, lut_max = 35.71;
    double actual_min = scalarRange[0];
    double actual_max = scalarRange[1];
    cout << "[DEBUG] Streamline scalar range for coloring: [" << actual_min << ", " << actual_max << "]" << std::endl;
    if (rep.contains("scene_info") && rep["scene_info"].contains("colorMapping") && rep["scene_info"]["colorMapping"].contains("scalarRange")) {
        auto sr = rep["scene_info"]["colorMapping"]["scalarRange"];
        if (sr.is_array() && sr.size() == 2) {
            // Use min(config_min, actual_min) and max(config_max, actual_max)
            lut_min = std::min(sr[0].get<double>(), actual_min);
            lut_max = std::min(sr[1].get<double>(), actual_max);
        } else {
            lut_min = actual_min;
            lut_max = actual_max;
        }
    } else {
        lut_min = actual_min;
        lut_max = actual_max;
    }
    lut->SetRange(lut_min, lut_max);
    // LUT: low values white, high values red
    for (int i = 0; i < 256; ++i) {
        double t = static_cast<double>(i) / 255.0;
        // t=0: white (1,1,1), t=1: red (1,0,0)
        lut->SetTableValue(i, 1.0, 1.0 - t, 1.0 - t, 1.0);
    }
    lut->Build();
    streamLineMapper->SetLookupTable(lut);
    streamLineMapper->SetColorModeToMapScalars();
    // Use LUT range from config for scalar coloring
    streamLineMapper->SetScalarRange(lut_min, lut_max);

    vtkNew<vtkActor> streamLineActor;
    streamLineActor->SetMapper(streamLineMapper);
    streamLineActor->VisibilityOn();

    // Apply streamline properties from config
    if (streamline_config.contains("streamlineProperties")) {
        auto stream_props = streamline_config["streamlineProperties"];
        if (stream_props.contains("lineWidth")) {
            streamLineActor->GetProperty()->SetLineWidth(stream_props["lineWidth"].get<double>());
        }
        if (stream_props.contains("opacity")) {
            streamLineActor->GetProperty()->SetOpacity(stream_props["opacity"].get<double>());
        }
        if (stream_props.contains("renderAsTubes") && stream_props["renderAsTubes"].get<bool>()) {
            streamLineActor->GetProperty()->SetRenderLinesAsTubes(true);
        }
        if (stream_props.contains("specular")) {
            streamLineActor->GetProperty()->SetSpecular(stream_props["specular"].get<double>());
        }
        if (stream_props.contains("specularPower")) {
            streamLineActor->GetProperty()->SetSpecularPower(stream_props["specularPower"].get<double>());
        }
    }

    ren1->AddActor(streamLineActor);

    // Add outline if enabled
    if (streamline_config.contains("outline") && streamline_config["outline"].contains("enabled") && 
        streamline_config["outline"]["enabled"].get<bool>()) {
        vtkNew<vtkActor> outline;
        outline->SetMapper(outlineMapper);
        
        auto outline_config = streamline_config["outline"];
        if (outline_config.contains("color")) {
            auto color = outline_config["color"];
            outline->GetProperty()->SetColor(color[0].get<double>(), 
                                            color[1].get<double>(), 
                                            color[2].get<double>());
        }
        if (outline_config.contains("lineWidth")) {
            outline->GetProperty()->SetLineWidth(outline_config["lineWidth"].get<double>());
        }
        
        ren1->AddActor(outline);
    }
    
    std::cout << "=== STREAMLINE RENDERING COMPLETE ===\n" << std::endl;
}

// Helper: setup streamlines directly from a vtkDataSet (structured grid or other)
void setupStreamlinesFromData(vtkDataSet* ds, vtkRenderer* ren1, const json& rep) {
    if (!rep.contains("enabled") || !rep["enabled"].get<bool>()) {
        std::cout << "Streamline representation disabled, skipping..." << std::endl;
        return;
    }

    const json& streamline_config = rep["scene_info"];
    std::cout << "\n=== CREATING STREAMLINES (from dataset) ===" << std::endl;

    vtkNew<vtkNamedColors> colors;

    // Create outline using dataset bounds
    double bounds[6]; ds->GetBounds(bounds);
    double bblow[3] = {bounds[0], bounds[2], bounds[4]};
    double bblen[3] = {(bounds[1]-bounds[0]), (bounds[3]-bounds[2]), (bounds[5]-bounds[4])};

    // Create seed plane
    vtkNew<vtkPlaneSource> source;
    if (streamline_config.contains("seedPlane") && streamline_config["seedPlane"].contains("enabled") &&
        streamline_config["seedPlane"]["enabled"].get<bool>()) {
        auto seed_config = streamline_config["seedPlane"];
        double pos_fraction = seed_config.contains("positionFraction") ? seed_config["positionFraction"].get<double>() : 0.25;
        source->SetOrigin(bblow[0] + bblen[0] * pos_fraction, bblow[1], bblow[2]);
        source->SetPoint1(bblow[0] + bblen[0] * pos_fraction, bblow[1] + bblen[1], bblow[2]);
        source->SetPoint2(bblow[0] + bblen[0] * pos_fraction, bblow[1], bblow[2] + bblen[2]);
        int xRes = seed_config.contains("xResolution") ? seed_config["xResolution"].get<int>() : 20;
        int yRes = seed_config.contains("yResolution") ? seed_config["yResolution"].get<int>() : 20;
        source->SetXResolution(xRes);
        source->SetYResolution(yRes);
    }
    source->Update();

    // Select vector array
    vtkPointData* pd = ds->GetPointData();
    vtkDataArray* vecArr = nullptr;
    if (rep.contains("field") && !rep["field"].is_null()) {
        std::string fname = rep["field"].get<std::string>();
        vecArr = pd->GetArray(fname.c_str());
        if (!vecArr) {
            std::cout << "[WARN] vector field '" << fname << "' not found; falling back to active vectors" << std::endl;
        }
    }
    if (!vecArr) vecArr = pd->GetVectors();
    if (!vecArr) {
        std::cout << "[WARN] No vector array found; streamlines will be skipped." << std::endl;
        return;
    }
    pd->SetActiveVectors(vecArr->GetName());

    // Build tracer
    vtkNew<vtkStreamTracer> streamers;
    // We need a vtkAlgorithmOutput; many datasets are not pipeline sources, so use a temporary producer
    vtkNew<vtkPolyData> seedPD;
    streamers->SetInputData(ds);
    streamers->SetSourceConnection(source->GetOutputPort());

    if (streamline_config.contains("integrationProperties")) {
        auto int_props = streamline_config["integrationProperties"];
        if (int_props.contains("maxPropagation")) streamers->SetMaximumPropagation(int_props["maxPropagation"].get<double>());
        if (int_props.contains("initialIntegrationStep")) streamers->SetInitialIntegrationStep(int_props["initialIntegrationStep"].get<double>());
        if (int_props.contains("minimumIntegrationStep")) streamers->SetMinimumIntegrationStep(int_props["minimumIntegrationStep"].get<double>());
        if (int_props.contains("integratorType")) streamers->SetIntegratorType(int_props["integratorType"].get<int>());
        if (int_props.contains("integrationDirection")) {
            std::string dir = int_props["integrationDirection"].get<std::string>();
            if (dir == "both") streamers->SetIntegrationDirectionToBoth();
            else if (dir == "forward") streamers->SetIntegrationDirectionToForward();
            else if (dir == "backward") streamers->SetIntegrationDirectionToBackward();
        }
    }
    streamers->Update();

    vtkNew<vtkPolyDataMapper> streamLineMapper;
    streamLineMapper->SetInputConnection(streamers->GetOutputPort());
    streamLineMapper->ScalarVisibilityOn();

    // Synchronize streamline coloring with volume scalar field
    vtkPolyData* streamPoly = streamers->GetOutput();
    // Probe the volume scalars onto the streamlines
    vtkImageData* volumeImage = vtkImageData::SafeDownCast(ds);
    if (volumeImage && volumeImage->GetPointData() && volumeImage->GetPointData()->GetScalars()) {
        vtkNew<vtkProbeFilter> probe;
        probe->SetSourceData(volumeImage);
        probe->SetInputData(streamPoly);
        probe->Update();
        vtkPolyData* probedPoly = vtkPolyData::SafeDownCast(probe->GetOutput());
        if (probedPoly && probedPoly->GetPointData()->GetScalars()) {
            streamPoly->GetPointData()->SetScalars(probedPoly->GetPointData()->GetScalars());
        }
    }
    double scalarRange[2] = {0.0, 1.0};
    if (streamPoly && streamPoly->GetPointData() && streamPoly->GetPointData()->GetScalars()) {
        streamPoly->GetPointData()->GetScalars()->GetRange(scalarRange);
        std::cout << "[DEBUG] Streamline scalar range for coloring: [" << scalarRange[0] << ", " << scalarRange[1] << "]" << std::endl;
    } else {
        std::cout << "[DEBUG] No scalars found on streamlines for coloring." << std::endl;
    }
    vtkNew<vtkLookupTable> lut;
    lut->SetNumberOfTableValues(256);
    // Read LUT range from config if present
    double lut_min = 34.0, lut_max = 35.71;
    double actual_min = scalarRange[0];
    double actual_max = scalarRange[1];
    std::cout << "[DEBUG] Streamline scalar range for coloring: [" << actual_min << ", " << actual_max << "]" << std::endl;
    if (rep.contains("scene_info") && rep["scene_info"].contains("colorMapping") && rep["scene_info"]["colorMapping"].contains("scalarRange")) {
        auto sr = rep["scene_info"]["colorMapping"]["scalarRange"];
        if (sr.is_array() && sr.size() == 2) {
            // Use min(config_min, actual_min) and max(config_max, actual_max)
            lut_min = std::min(sr[0].get<double>(), actual_min);
            lut_max = std::min(sr[1].get<double>(), actual_max);
        } else {
            lut_min = actual_min;
            lut_max = actual_max;
        }
    } else {
        lut_min = actual_min;
        lut_max = actual_max;
    }
    lut->SetRange(lut_min, lut_max);
    // LUT: low values white, high values red
    for (int i = 0; i < 256; ++i) {
        double t = static_cast<double>(i) / 255.0;
        // t=0: white (1,1,1), t=1: red (1,0,0)
        lut->SetTableValue(i, 1.0, 1.0 - t, 1.0 - t, 1.0);
    }
    lut->Build();
    streamLineMapper->SetLookupTable(lut);
    streamLineMapper->SetColorModeToMapScalars();
    streamLineMapper->SetScalarRange(lut_min, lut_max);

    vtkNew<vtkActor> streamLineActor;
    streamLineActor->SetMapper(streamLineMapper);
    streamLineActor->VisibilityOn();
    if (streamline_config.contains("streamlineProperties")) {
        auto stream_props = streamline_config["streamlineProperties"];
        if (stream_props.contains("lineWidth")) streamLineActor->GetProperty()->SetLineWidth(stream_props["lineWidth"].get<double>());
        if (stream_props.contains("opacity")) streamLineActor->GetProperty()->SetOpacity(stream_props["opacity"].get<double>());
        if (stream_props.contains("renderAsTubes") && stream_props["renderAsTubes"].get<bool>()) streamLineActor->GetProperty()->SetRenderLinesAsTubes(true);
    }

    ren1->AddActor(streamLineActor);
    std::cout << "=== STREAMLINE FROM DATA COMPLETE ===\n" << std::endl;
}

void createIsosurfaceFromJSON(vtkImageData* img, vtkRenderer* ren1, double* range, const json& rep) {
    if (!rep.contains("enabled") || !rep["enabled"].get<bool>()) {
        std::cout << "Isosurface representation disabled, skipping..." << std::endl;
        return;
    }
    
    if (!rep.contains("scene_info")) {
        std::cout << "No scene_info found for isosurface representation" << std::endl;
        return;
    }

    const json& isosurface_config = rep["scene_info"];
    std::cout << "\n=== CREATING ISOSURFACE (LAND MASK) ===" << std::endl;

    // Get threshold range from config
    double lower_threshold = 0.0;
    double upper_threshold = 0.005; // default MAX_LAND_SALINITY
    if (isosurface_config.contains("thresholdRange")) {
        auto threshold_range = isosurface_config["thresholdRange"];
        lower_threshold = threshold_range[0].get<double>();
        upper_threshold = threshold_range[1].get<double>();
    }

    vtkNew<vtkThreshold> threshold;
    threshold->SetInputData(img);
    threshold->SetLowerThreshold(lower_threshold);
    threshold->SetUpperThreshold(upper_threshold);
    threshold->Update();

    vtkNew<vtkGeometryFilter> geometryFilter;
    geometryFilter->SetInputConnection(threshold->GetOutputPort());
    geometryFilter->Update();

    vtkPolyData* landPolyData = geometryFilter->GetOutput();
    vtkPoints* points = landPolyData->GetPoints();
    
    if (!points || points->GetNumberOfPoints() == 0) {
        std::cout << "No points found in land isovolume." << std::endl;
        return;
    }

    std::cout << "Land isovolume found with " << points->GetNumberOfPoints() << " points" << std::endl;

    double bounds[6];
    landPolyData->GetBounds(bounds);
    double x_min = bounds[0], x_max = bounds[1];
    double y_min = bounds[2], y_max = bounds[3];
    double z_min = bounds[4], z_max = bounds[5];
    (void)x_min; (void)x_max; // unused here but kept for clarity

    std::cout << "[DEBUG] Isosurface bounds: x=[" << x_min << "," << x_max << "], y=[" << y_min << "," << y_max << "], z=[" << z_min << "," << z_max << "]" << std::endl;
    std::cout << "[DEBUG] Isosurface point count: " << points->GetNumberOfPoints() << std::endl;
    // Print first few points
    vtkIdType maxDump = std::min<vtkIdType>(points->GetNumberOfPoints(), 5);
    for (vtkIdType pi = 0; pi < maxDump; ++pi) {
        double p[3]; points->GetPoint(pi, p);
        std::cout << "[DEBUG] pt[" << pi << "] = (" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    }

    // Create texture coordinates
    vtkNew<vtkFloatArray> texCoords;
    texCoords->SetName("TextureCoordinates");
    texCoords->SetNumberOfComponents(2);
    texCoords->SetNumberOfTuples(points->GetNumberOfPoints());

    // Create texture coordinates using the same mapping as v2:
    // u <- z axis, v <- y axis (map into 0..1 using bounds)
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        double pt[3];
        points->GetPoint(i, pt);
        double u = (z_max > z_min) ? (pt[2] - z_min) / (z_max - z_min) : 0.0;
        double v = (y_max > y_min) ? (pt[1] - y_min) / (y_max - y_min) : 0.0;

        // Clamp to [0,1]
        u = std::max(0.0, std::min(1.0, u));
        v = std::max(0.0, std::min(1.0, v));

        texCoords->SetTuple2(i, u, v);
    }
    landPolyData->GetPointData()->SetTCoords(texCoords);

    // Check for texture in config
    bool use_texture = false;
    std::string textureFilePath;
    if (isosurface_config.contains("texture") && isosurface_config["texture"].contains("enabled") && 
        isosurface_config["texture"]["enabled"].get<bool>()) {
        textureFilePath = isosurface_config["texture"]["textureFile"].get<std::string>();
        use_texture = std::filesystem::exists(textureFilePath);
    }

    if (use_texture) {
        std::cout << "[DEBUG] Texture file exists: " << textureFilePath << std::endl;
        std::cout << "[DEBUG] Renderer props before adding isosurface: " << ren1->GetViewProps()->GetNumberOfItems() << std::endl;
        vtkNew<vtkPNGReader> textureReader;
        textureReader->SetFileName(textureFilePath.c_str());
        textureReader->Update();
        
        vtkNew<vtkTexture> texture;
        texture->SetInputConnection(textureReader->GetOutputPort());
        texture->InterpolateOn();
        texture->RepeatOff();
        
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(landPolyData);
        mapper->ScalarVisibilityOff();
        
        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->SetTexture(texture);
        
        // Apply surface properties from config
        if (isosurface_config.contains("surfaceProperties")) {
            auto props = isosurface_config["surfaceProperties"];
            if (props.contains("opacity")) {
                actor->GetProperty()->SetOpacity(props["opacity"].get<double>());
            }
        }
        
        ren1->AddActor(actor);
        std::cout << "Applied land texture from: " << textureFilePath << std::endl;
        std::cout << "[DEBUG] Added textured isosurface actor (points=" << points->GetNumberOfPoints() << ")" << std::endl;
        std::cout << "[DEBUG] Renderer props after adding isosurface: " << ren1->GetViewProps()->GetNumberOfItems() << std::endl;
    } else {
        // Fallback to color-based rendering
        std::cout << "Using fallback color-based land rendering" << std::endl;
        
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

        // Get fallback color from config
        std::vector<double> fallback_color = {0.518, 0.408, 0.216};
        if (isosurface_config.contains("surfaceProperties") && 
            isosurface_config["surfaceProperties"].contains("color")) {
            auto color = isosurface_config["surfaceProperties"]["color"];
            fallback_color = {color[0].get<double>(), color[1].get<double>(), color[2].get<double>()};
        }

        vtkNew<vtkLookupTable> lut;
        lut->SetNumberOfTableValues(256);
        lut->SetRange(0.0, 1.0);
        for (int i = 0; i < 256; ++i) {
            double t = double(i) / 255.0;
            lut->SetTableValue(i, fallback_color[0] * t, fallback_color[1] * t, fallback_color[2] * t, 1.0);
        }
        lut->Build();

        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(landPolyData);
        mapper->SetLookupTable(lut);
        mapper->SetScalarRange(0.0, 1.0);
        mapper->ScalarVisibilityOn();

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetOpacity(1.0);
        
        ren1->AddActor(actor);
        std::cout << "[DEBUG] Added color fallback isosurface actor (points=" << points->GetNumberOfPoints() << ")" << std::endl;
        std::cout << "[DEBUG] Renderer props after adding isosurface: " << ren1->GetViewProps()->GetNumberOfItems() << std::endl;
    }
    
    std::cout << "=== ISOSURFACE COMPLETE ===\n" << std::endl;
}

void loadCameraFromJSON(AnimatorKF &keyframe, vtkRenderer *ren1)
{
    visuser::Camera c;
    keyframe.get_current_cam(c);
    auto camera = ren1->GetActiveCamera();

    // Use the interpolated Camera returned by AnimatorKF::get_current_cam exclusively.
    // `c.pos` contains the interpolated eye position. When the animator detected
    // JSON that provided position+focalPoint it stored an interpolated focal point
    // in `c.dir` (so dir acts as focalPoint in that mode). When the JSON used
    // pos+dir the animator stored a normalized direction vector in `c.dir`.
    // Renderer uses these values directly so the animator is responsible for
    // ensuring `c` contains the desired semantics.
    double eyePos[3] = {c.pos[0], c.pos[1], c.pos[2]};
    double focalPoint[3] = {c.dir[0], c.dir[1], c.dir[2]};
    camera->SetPosition(eyePos[0], eyePos[1], eyePos[2]);
    camera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);


    camera->SetViewUp(c.up[0], c.up[1], c.up[2]);
    ren1->ResetCameraClippingRange();

    std::cout << "Camera loaded - Position: [" << camera->GetPosition()[0] << ", " << camera->GetPosition()[1] << ", " << camera->GetPosition()[2] << "]" << std::endl;
    std::cout << "                FocalPoint: [" << camera->GetFocalPoint()[0] << ", " << camera->GetFocalPoint()[1] << ", " << camera->GetFocalPoint()[2] << "]" << std::endl;
    std::cout << "                Up: [" << c.up[0] << ", " << c.up[1] << ", " << c.up[2] << "]" << std::endl;
}

void loadKF2(Animator &animator, uint32_t idx, vtkImageData *img, vtkRenderer *ren1)
{
    std::cout << "\n=== LOADING KEYFRAME " << idx << " ===" << std::endl;
    
    ren1->RemoveAllViewProps();
    
    if (animator.kfs[idx].get_data_list_size() == 0) {
        std::cout << "No data in keyframe " << idx << std::endl;
        return;
    }
    
    // Get data source info
    json data = animator.get_scene_data(idx, 0);
    std::string file_name = data["src"]["name"];
    glm::vec3 data_dims = get_vec3f(data["src"]["dims"]);
    int json_dims[3] = {static_cast<int>(data_dims[0]), static_cast<int>(data_dims[1]), static_cast<int>(data_dims[2])};
    
    std::cout << "Loading data from: " << file_name << std::endl;
    std::cout << "JSON data dims: " << json_dims[0] << " x " << json_dims[1] << " x " << json_dims[2] << std::endl;
    // Read VTK file (streamline format with velocity data)
    vtkNew<vtkStructuredGridReader> reader;
    reader->SetFileName(file_name.c_str());
    reader->Update();

    vtkDataSet* ds = reader->GetOutput();

    double range[2] = {0.0, 1.0};
    if (ds->GetPointData() && ds->GetPointData()->GetScalars()) {
        ds->GetPointData()->GetScalars()->GetRange(range);
    }
    std::cout << "Scalar range: [" << range[0] << ", " << range[1] << "]" << std::endl;

    // Create ImageData for volume/isosurface (we may replace its content by probing)
    vtkNew<vtkImageData> imageData;
    imageData->Initialize();

    // Prefer reader dimensions (VTK output) to ensure consistent coordinate ordering
    int reader_dims[3] = {0,0,0};
    vtkStructuredGrid* sgrid_out = vtkStructuredGrid::SafeDownCast(ds);
    if (sgrid_out) {
        sgrid_out->GetDimensions(reader_dims);
    } else {
        vtkImageData* img_out = vtkImageData::SafeDownCast(ds);
        if (img_out) img_out->GetDimensions(reader_dims);
    }
    if (reader_dims[0] > 0 && reader_dims[1] > 0 && reader_dims[2] > 0) {
        imageData->SetDimensions(reader_dims);
        std::cout << "Reader data dims: " << reader_dims[0] << " x " << reader_dims[1] << " x " << reader_dims[2] << std::endl;
    } else {
        imageData->SetDimensions(json_dims);
        std::cout << "Reader dims unavailable, using JSON dims: " << json_dims[0] << " x " << json_dims[1] << " x " << json_dims[2] << std::endl;
    }

    // Get spacing/origin from animator global metadata if available
    auto global_meta = animator.get_global_metadata();
    if (!global_meta.is_null() && global_meta.contains("spacing") && global_meta.contains("origin")) {
        glm::vec3 sp = get_vec3f(global_meta["spacing"]);
        glm::vec3 org = get_vec3f(global_meta["origin"]);
        imageData->SetSpacing(sp[0], sp[1], sp[2]);
        imageData->SetOrigin(org[0], org[1], org[2]);
        std::cout << "Using spacing from global_metadata: [" << sp[0] << ", " << sp[1] << ", " << sp[2] << "]" << std::endl;
    } else {
        imageData->SetSpacing(1.0, 1.0, 1.0);
        imageData->SetOrigin(0.0, 0.0, 0.0);
    }
    
    // Get transfer function data
    std::vector<float> tf_colors, tf_opacities;
    json scene_data_list = animator.kfs[idx].get_scene_data_list();
    
    // Process representations in the order they appear in the JSON (single pass)
    for (size_t rep_idx = 0; rep_idx < scene_data_list.size(); rep_idx++) {
        json rep = scene_data_list[rep_idx];
    if (!rep.contains("enabled") || !rep["enabled"].get<bool>()) continue;
    if (!rep.contains("scene_info")) continue;
    // If the scene_info explicitly disables this representation, skip it
    if (rep["scene_info"].contains("enabled") && !rep["scene_info"]["enabled"].get<bool>()) continue;
        std::string rep_type = rep["representation"].get<std::string>();

        std::cout << "\nProcessing representation " << rep_idx << ": " << rep_type << std::endl;

        if (rep_type == "volume") {
            json scene_info = rep["scene_info"];
            if (scene_info.contains("transferFunc")) {
                tf_colors = scene_info["transferFunc"]["colors"].get<std::vector<float>>();
                tf_opacities = scene_info["transferFunc"]["opacities"].get<std::vector<float>>();
            }
            // If the global metadata indicates a structured grid, resample the structured dataset
            auto global_meta = animator.get_global_metadata();
            bool is_structured = (!global_meta.is_null() && global_meta.contains("grid_type") && global_meta["grid_type"].get<std::string>() == "structured");

            if (is_structured) {
                // Build a target image with JSON dims and spacing/origin from global metadata
                // Use reader dimensions as the sampling grid to preserve VTK axis ordering
                int target_dims[3] = {reader_dims[0], reader_dims[1], reader_dims[2]};
                glm::vec3 sp_json(1.0f,1.0f,1.0f), org_json(0.0f,0.0f,0.0f);
                if (!global_meta.is_null() && global_meta.contains("spacing")) sp_json = get_vec3f(global_meta["spacing"]);
                if (!global_meta.is_null() && global_meta.contains("origin")) org_json = get_vec3f(global_meta["origin"]);

                // Detect if reader dims are a permutation of json dims (common case: axes swapped)
                bool permute = false;
                if (reader_dims[0] == json_dims[2] && reader_dims[1] == json_dims[1] && reader_dims[2] == json_dims[0]) {
                    permute = true; // reader is (z,y,x) while json is (x,y,z)
                }

                glm::vec3 sp_target = sp_json;
                glm::vec3 org_target = org_json;
                if (permute) {
                    sp_target = glm::vec3(sp_json[2], sp_json[1], sp_json[0]);
                    org_target = glm::vec3(org_json[2], org_json[1], org_json[0]);
                    std::cout << "[INFO] Detected axis permutation between reader dims and JSON dims; permuting spacing/origin accordingly." << std::endl;
                }

                vtkNew<vtkImageData> targetImage;
                targetImage->SetDimensions(target_dims);
                targetImage->SetSpacing(sp_target[0], sp_target[1], sp_target[2]);
                targetImage->SetOrigin(org_target[0], org_target[1], org_target[2]);

                // Use vtkProbeFilter to sample dataset onto the target image
                vtkNew<vtkProbeFilter> probe;
                probe->SetSourceData(ds);
                probe->SetInputData(targetImage);
                probe->Update();

                vtkImageData* sampled = vtkImageData::SafeDownCast(probe->GetOutput());
                if (sampled) {
                    // Set sampled image as the one used for volume and isosurface
                    imageData->ShallowCopy(sampled);
                    sampled->GetPointData()->GetScalars()->GetRange(range);
                    std::cout << "[INFO] Resampled structured grid to image grid for volume rendering." << std::endl;
                } else {
                    std::cout << "[WARN] Resampling failed, falling back to original imageData." << std::endl;
                }

            }
            setupVolumeFromJSON(imageData, ren1, range, rep, tf_colors, tf_opacities);
        } else if (rep_type == "isosurface") {
            createIsosurfaceFromJSON(imageData, ren1, range, rep);
        } else if (rep_type == "streamline") {
            // Prefer dataset-based streamlines which can use structured grids directly
            vtkDataSet* dataset = ds;
            if (dataset) {
                setupStreamlinesFromData(dataset, ren1, rep);
            } else {
                // Fallback to reader-based helper if dataset unavailable
                setupStreamlinesFromJSON(reader, ren1, rep);
            }
        } else {
            std::cout << "Unknown representation type: " << rep_type << "; skipping." << std::endl;
        }
    }
    
    // Set background color (hardcoded black as requested)
    vtkNew<vtkNamedColors> colors;
    ren1->SetBackground(colors->GetColor3d("black").GetData());
    
    // Load camera
    loadCameraFromJSON(animator.kfs[idx], ren1);

    // Comment out the line above and uncomment the line below.
    // ren1->ResetCamera();
    // std::cout << "=== WARNING: USING AUTOMATIC ResetCamera() ===" << std::endl;
    
    
    // Debug: print renderer props summary (do NOT delete the collection returned by GetViewProps)
    vtkPropCollection* props = ren1->GetViewProps();
    if (props) {
        std::cout << "[DEBUG] Renderer has " << props->GetNumberOfItems() << " view props:" << std::endl;
        for (int i = 0; i < props->GetNumberOfItems(); ++i) {
            vtkProp* p = vtkProp::SafeDownCast(props->GetItemAsObject(i));
            if (!p) continue;
            const char* cname = p->GetClassName();
            std::string type = cname ? cname : "unknown";
            std::cout << "  prop[" << i << "] type=" << type;
            if (vtkActor::SafeDownCast(p)) {
                vtkActor* a = vtkActor::SafeDownCast(p);
                std::cout << " (actor) opacity=" << a->GetProperty()->GetOpacity();
            }
            if (vtkVolume::SafeDownCast(p)) {
                std::cout << " (volume)";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "=== KEYFRAME " << idx << " LOADED ===\n" << std::endl;
}

void writeImage2(std::string const& fileName, vtkRenderWindow* renWin, bool rgba)
{
    if (!fileName.empty())
    {
        std::string fn = fileName;
        auto writer = vtkSmartPointer<vtkPNGWriter>::New();
        
        vtkNew<vtkWindowToImageFilter> window_to_image_filter;
        window_to_image_filter->SetInput(renWin);
        window_to_image_filter->SetScale(1);
        if (rgba) window_to_image_filter->SetInputBufferTypeToRGBA();	
        else window_to_image_filter->SetInputBufferTypeToRGB();
        
        window_to_image_filter->ReadFrontBufferOff();
        window_to_image_filter->Update();

        writer->SetFileName(fn.c_str());
        writer->SetInputConnection(window_to_image_filter->GetOutputPort());
        writer->Write();
        
        std::cout << "Saved visualization to: " << fn << std::endl;
    }
    else std::cerr << "No filename provided." << std::endl;

    return;
}

void run3(std::string jsonStr, std::string output_dir, int header_sel){
    std::cout << "\n=== STARTING VTK RENDERING ===" << std::endl;

    if (!output_dir.empty()) {
        if (!std::filesystem::exists(output_dir)) {
            std::cout << "Creating output directory: " << output_dir << std::endl;
            std::filesystem::create_directories(output_dir);
        }
    }
    
    std::cout << "\n\nStart json loading ... \n";
    Animator animator;
    animator.init(jsonStr.c_str());
    std::cout << "\nEnd json loading ... \n\n";
    
    vtkNew<vtkRenderer> ren1;
    vtkNew<vtkRenderWindow> renWin;
    renWin->AddRenderer(ren1);
    
    vtkNew<vtkRenderWindowInteractor> iren;
    iren->SetRenderWindow(renWin);
    
    vtkNew<vtkImageData> img;
    loadKF2(animator, 0, img, ren1);
    
    // Hardcoded window settings as requested
    renWin->SetSize(1200, 900);
    renWin->SetWindowName("Oceanographic Visualization");
    renWin->Render();
    
    if (header_sel >= 0){ 
        std::string outname;
        if (!output_dir.empty()) {
            outname = output_dir + "/" + getOutName2("", header_sel);
        } else {
            outname = getOutName2("", header_sel);
        }
        writeImage2(outname, renWin, false);
    } else {
        for (size_t kf_idx=0; kf_idx<animator.kfs.size(); kf_idx++){
            loadKF2(animator, static_cast<uint32_t>(kf_idx), img, ren1);
            if (header_sel == -1){ 
                renWin->Render();
                std::string outname;
                if (!output_dir.empty()) {
                    outname = output_dir + "/" + getOutName2("", kf_idx);
                } else {
                    outname = getOutName2("", kf_idx);
                }
                writeImage2(outname, renWin, false);
            } else if (header_sel == -2){ 
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
                        animator.kfs[kf_idx].advanceFrame();
                        loadCameraFromJSON(animator.kfs[kf_idx], ren1);
                        //    ren1->ResetCamera();
                        //    std::cout << "=== WARNING: USING AUTOMATIC ResetCamera() ===" << std::endl;
                        
                    }
                }
            }
        }
    }
    
    std::cout << "=== VTK VISUALIZATION COMPLETE ===" << std::endl;
}