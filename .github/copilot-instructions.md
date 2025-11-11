# Copilot Instructions for vis_user_tool

## Project Overview
This is a **scientific visualization tool** for large-scale oceanographic data animation, featuring JSON-based scripting and multiple rendering backends (OSPRay v2.12, VTK 9.3.1). The system enables automated animation production from remote scientific datasets with AI-assisted interfaces.

## Architecture Components

### Core Data Flow
1. **Data Sources**: OpenVisus `.idx` files from remote repositories (NSDF, NASA datasets)
2. **Python Interface**: `python/renderInterface.py` - main data handling and animation scripting  
3. **Rendering Backends**: 
   - OSPRay: `renderingApps/osprayApps/` (volume rendering)
   - VTK: `renderingApps/vtkApps/vtkFuns2.cpp` (current active backend for improved land-ocean rendering)
4. **Output**: PNG frame sequences → FFmpeg animations

### Key File Patterns
- **Animation Scripts**: `python/case_study_*-*/script_cs*.py` - generate JSON configurations
- **JSON Configs**: `*/GAD_text/*.json` - define camera paths, transfer functions, data regions
- **Rendered Output**: `*/rendered_frames/` directories - PNG sequences organized by case study

## Essential Development Workflows

### Building the System
```bash
# Full dependency build (OSPRay + VTK from source)
./superbuild.sh

# Incremental builds (after code changes)
cd build && make -j
```

### Running Visualizations
```bash
# Generate JSON scripts for a case study
python python/case_study_1-1/script_cs1-1.py -save

# Render with VTK backend (current preference)
python python/renderVTK.py python/case_study_1-1/GAD_text/case1_script.json

# Alternative OSPRay rendering
python python/render.py path_to_script.json
```

### VTK Development Cycle
Critical: After modifying `renderingApps/vtkApps/vtkFuns2.cpp`:
1. `cd build && make -j` 
2. Verify `python/renderInterface.py` line 288 references `vistool_py_vtk2`
3. Test with: `python python/test/test-vtk.py -save` then render

## Project-Specific Conventions

### Data Region Definitions
```python
# Agulhas Current (common study region)
x_range = [int(a.x_max*0.119), int(a.x_max*0.253)]
y_range = [int(a.y_max*0.378667), int(a.y_max*0.501333)]
t_list = np.arange(0, 96, 24, dtype=int).tolist()
quality = -6  # LOD for remote data
```

### File Naming Patterns
- Raw data: `ocean_{dimsx}_{dimsy}_{dimsz}_t{time}_float32.raw`
- VTK files: `ocean_{dimsx}_{dimsy}_{dimsz}_t{time}.vtk`  
- Rendered output: `img_{frame}_f{time}.png`

### JSON Schema Structure
Animation scripts must include:
- `isheader`: boolean for keyframe vs data definition
- `data`: structured data with dims, world_bbox, frameRange
- `camera`: array with frame, pos, dir, up vectors
- `transferFunc`: frame-based color mapping and opacity functions

## Integration Points

### Python Environment Setup
Always activate: `source venv_new/bin/activate`
Install deps: `pip install -r requirements.txt` 

### OpenVisus Integration  
- Remote data access via `OpenVisus` and `openvisuspy` modules
- URL patterns: `https://nsdf-climate3-origin.nationalresearchplatform.org:50098/...`
- Common datasets: temperature (theta), salinity (salt), velocity components (u,v,w)

### AI Interface
- Entry point: `python AIExample/script/openai_run.py`
- Requires: `openai_api_key.txt` in AIExample directory
- Generates visualization scripts from natural language prompts

## Debugging Notes

### Common Issues
- **Library version conflicts**: Use `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` prefix for conda/C++ mismatch
- **VTK backend switching**: Ensure `renderInterface.py` imports correct `vistool_py_vtk2`
- **Data download timeouts**: Check quality parameter and region size

### Development Targets
- Land-ocean boundary improvements in VTK renderer
- Performance optimization for large dataset streaming
- Enhanced AI-driven animation scripting