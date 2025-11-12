# NLtoAnimation - Web-based Oceanographic Animation System

## Overview
NLtoAnimation is a web application that transforms the Agent.py command-line tool into an interactive, ChatGPT-like interface for generating animations. The system enables users to create complex animations through natural language conversations, combining AI-powered parameter selection.

## How to Run

Follow these steps from the repository root. Commands below assume a Unix-like shell (Linux/macOS).

1) Create and activate a virtual environment in the project root:

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install Python dependencies (from the repository root):

```bash
pip install -r requirements.txt
```

3) Install rendering backend compatible tools

```
./superbuild.sh
```

4) Change into the web app folder and run the development server:

```bash
cd agents
python src/app.py
```

Alternative: use the Flask CLI and explicitly bind to all interfaces so other machines on your network can connect:

```bash
export FLASK_APP=src/app.py
flask run --host=0.0.0.0 --port=5000
# or
python -m flask run --host=0.0.0.0 --port=5000
```

Notes:
- The built-in Flask server is fine for development and demos. For production expose the app via a WSGI server (gunicorn/uvicorn) behind nginx and enable TLS.
- If you want the app reachable from other machines, use `--host=0.0.0.0` and make sure the server's firewall allows inbound connections on the chosen port (default 5000).


## Architecture Overview

### System Components
1. **Frontend Interface**: Responsive web UI with chat-like interaction
2. **Backend API**: Flask-based server handling conversation flow and animation generation
3. **Animation Engine**: Frame-based animation system with basic playback
4. **Data Pipeline**: Integration with DYAMOND LLC2160 high-resolution ocean simulation data

### Key Features
- **Conversational Interface**: Natural language interaction for animation parameter selection
- **Real-time Frame Discovery**: Dynamic loading of animation frames as they're generated
- **Interactive Playback Controls**: Full animation control with play/pause, seeking, and speed adjustment

## Project Structure
```
agents/
├── src/
│   ├── app.py                     # Flask application entry point
│   ├── api/
│   │   ├── routes.py              # API endpoints (/api/chat, /api/animations)
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css          # Dark theme UI styling
│   │   └── js/
│   │       └── chat.js            # Animation player and chat interface
│   ├── templates/
│   │   └── index.html             # Main application interface
│   └── models/
│       └── schemas.py             # Data validation schemas
├── animation_*/                   # Generated animation folders
│   └── Rendered_frames/           # Individual PNG frame files
│       ├── img_kf0f0.png         # Frame naming pattern: img_kfXfY.png
│       ├── img_kf0f1.png         # X = base index, Y = X or X+1
│       └── ...                    # 120 total frames (60 base indices × 2)
├── requirements.txt               # Python dependencies
└── README.md                     # This documentation
```

## Animation Pipeline Architecture

### 1. User Interaction Flow
```
User Input → Conversation Management → Parameter Selection → Animation Generation → Frame Discovery → Playback
```

#### Phase 1: Conversation Initiation
- User selects dataset (DYAMOND LLC2160)
- Chooses phenomenon type:
  - Agulhas Ring Current (Temperature/Salinity)
  - Mediterranean Sea Current (Temperature/Salinity)
  - Custom description (natural language)

#### Phase 2: AI-Powered Parameter Negotiation
- **Conversation Loop**: Interactive refinement of animation parameters
- **Smart Guidance**: AI suggests optimal parameter ranges and quality settings
- **Parameter Validation**: Real-time validation of oceanographic coordinates
- **Geographic Translation**: Automatic conversion between grid coordinates and lat/lon

#### Phase 3: Animation Generation Strategy
The system employs intelligent animation type detection:

**Existing Animation Detection:**
```javascript
// Detects: "Found existing animation. Reusing: /path/to/animation"
const isExistingAnimation = message.includes('Found existing animation. Reusing:');
```

**Strategy Selection:**
- **Pre-generated (Bulk Loading)**: Loads all 120 frames rapidly for existing animations
- **Real-time Generation**: Monitors frame creation with progressive discovery
- **Auto-detection**: Falls back to real-time if no existing animation found


#### Discovery Algorithms

**Bulk Loading (Existing Animations):**
```javascript
// Aggressive discovery with minimal delays
maxConsecutiveFails = 3;  // Quick termination
frameCheckDelay = 10ms;   // Rapid checking
strategy = "LOAD_ALL_EXISTING";
```

**Real-time Monitoring (New Animations):**
```javascript
// Patient discovery for generating frames
discoveryInterval = 3000ms;     // Check every 3 seconds
maxConsecutiveFails = 3;        // Conservative stopping
batchSize = 5;                  # Check 5 frames ahead per batch
```

### 2. Animation Playback Engine

#### Frame Management
- **Intelligent Caching**: Frames loaded once and cached for smooth playback
- **Progressive Loading**: New frames integrated seamlessly during generation
- **Memory Optimization**: Efficient frame storage and retrieval

#### Interactive Controls
```javascript
// Full animation control suite
playAnimation()     // Start/resume playback
pauseAnimation()    // Pause at current frame
stopAnimation()     // Reset to frame 0
nextFrame()         // Step forward one frame
previousFrame()     // Step backward one frame
seekToFrame(index)  // Jump to specific frame
setSpeed(multiplier) // Adjust playback speed (0.5x - 2.0x)
```

#### Real-time UI Updates
- **Progress Bar**: Visual progress indicator with click-to-seek
- **Frame Counter**: "Frame X / 120" display
- **Speed Control**: Dynamic speed adjustment with visual feedback
- **Play/Pause Toggle**: Contextual button state management

### 4. Technical Implementation Details

#### Frame Existence Validation
```javascript
async checkFrameExists(framePath) {
    const response = await fetch(framePath, { method: 'HEAD' });
    return response.ok;  // 200 = exists, 404 = missing
}
```

#### Smart Boundary Detection
```javascript
// Stops at actual frame boundaries (not hardcoded limits)
while (consecutiveFails < maxConsecutiveFails) {
    // Try both patterns for each base index
    patterns = [`img_kf${index}f${index}.png`, `img_kf${index}f${index + 1}.png`];
    // Increment consecutiveFails only when NO frames found for base index
}
```

#### Animation Type Strategy Selection
```javascript
if (isExistingAnimation === true) {
    this.discoverAllExistingFrames();  // Bulk load strategy
} else if (isExistingAnimation === false) {
    this.startFrameDiscovery();        // Real-time monitoring
} else {
    this.discoverFrames();             // Auto-detect strategy
}
```

## Data Integration

### DYAMOND LLC2160 Dataset
- **Resolution**: High-resolution global ocean simulation
- **Variables**: Temperature, Salinity, Velocity fields
- **Coverage**: Global ocean with focus on major current systems
- **Temporal Resolution**: Hourly data for detailed temporal analysis

### Coordinate System
- **Grid Coordinates**: Internal simulation grid (x, y, z, t)
- **Geographic Translation**: Automatic conversion to lat/lon coordinates
- **Depth Levels**: Configurable depth ranges for 3D analysis
- **Time Series**: Flexible temporal window selection

## User Interface Design

### Chat-based Interaction
- **Conversational Flow**: Natural language parameter selection
- **Visual Feedback**: Real-time status indicators and progress updates
- **Error Handling**: Graceful error recovery with user guidance
- **Responsive Design**: Optimized for desktop and mobile devices

### Animation Display
- **Adaptive Sizing**: Automatic image scaling with aspect ratio preservation
- **Control Integration**: Embedded animation controls below visualization
- **Visual Polish**: Dark theme with professional oceanographic presentation
- **Performance Optimization**: Smooth playback with minimal resource usage

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd agents
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in the required values.
   - Add your OpenAI API key to the `openai_api_key.txt` file in the root directory.
   
   **⚠️ Security Note:** Never commit `.env` files or API keys to version control. These files are automatically ignored by `.gitignore`.

4. **Run the application:**
   ```
   python src/app.py
   ```

5. **Access the application:**
   Open your web browser and navigate to `http://localhost:5000`.

## Usage Guide

### Getting Started
1. **Launch Application**: Navigate to `http://localhost:5000`
2. **Select Dataset**: Choose "Use Available Dataset" (DYAMOND LLC2160)
3. **Choose Phenomenon**: Select from predefined options or use custom description
4. **Interactive Conversation**: Engage in natural language parameter refinement
5. **Animation Generation**: System automatically generates or reuses existing animations
6. **Playback Control**: Use integrated controls for animation exploration

### Example Conversation Flow
```
System: Welcome! Let's create an oceanographic animation.
User: I want to visualize Agulhas Ring Current temperature
System: Great! I'll set up temperature visualization for the Agulhas Ring...

[Parameter negotiation through conversation]

System: Animation generated! Here's your visualization with 120 frames.
[Animation appears with full playback controls]

User: Can you make it cover a larger area?
System: I'll expand the spatial coverage and regenerate...
```

### Animation Controls
- **Play/Pause**: ▶️ Toggle animation playback
- **Stop**: ⏹️ Reset to first frame
- **Step Controls**: ⏮️ ⏭️ Navigate frame by frame
- **Progress Bar**: Click anywhere to jump to specific frame
- **Speed Control**: Adjust playback speed (0.5x to 2.0x)
- **Frame Counter**: Real-time frame position display

### Supported Phenomena
1. **Agulhas Ring Current**
   - Temperature visualization
   - Salinity with streamlines
2. **Mediterranean Sea Current**
   - Temperature dynamics
   - Salinity circulation patterns
3. **Custom Descriptions**
   - Natural language phenomenon specification
   - AI-powered parameter interpretation

## API Reference and Agent.py Integration

### API-to-Agent Function Mapping

The web interface communicates with Agent.py through a carefully orchestrated sequence of function calls. Each API endpoint triggers specific Agent.py functionality to create a seamless conversation flow.

#### Core Agent.py Functions

-##### 1. **get_region_from_description(description)**
- **Triggered by**: Phenomenon selection API call
- **Purpose**: Converts a free-text natural language description into structured region parameters
- **AI Integration**: Uses OpenAI GPT to parse custom descriptions into oceanographic parameters
- **Input**: Natural language description or predefined phenomenon ID
- **Output**: Complete region_params dictionary with x_range, y_range, z_range, t_list, quality settings

```python
# Example output structure
region_params = {
    "x_range": [1000, 2500],          # Grid coordinates in dataset
    "y_range": [2000, 3500], 
    "z_range": [0, 90],               # Depth range
    "t_list": [0, 4, 8, 12, 16],      # Time steps
    "quality": -6,                     # Compression quality
    "flip_axis": 2,                   # Data orientation
    "transpose": False,
    "render_mode": 0,                 # 0=flat, 2=spherical
    "needs_velocity": False,          # Streamlines enabled
    "field": "temperature"            # Data field type
}
```

##### 2. **find_existing_animation(region_params)**
- **Triggered by**: Before every animation generation
- **Purpose**: Intelligent animation caching system
- **Process**: 
  - Generates standardized folder name from parameters
  - Searches AIdemo directory for matching animations
  - Verifies GAD files and frame directories exist
- **Output**: Animation metadata or `{"exists": False}`

```python
# Returned when animation exists
{
    "output_base": "/path/to/animation_folder",
    "gad_file_path": "/path/to/script.json", 
    "frames_dir": "/path/to/Rendered_frames",
    "animation_path": "/path/to/animation.gif",
    "exists": True
}
```

##### 3. **generate_animation(region_params, phenomenon_description, output_id)**
- **Triggered by**: Initial generation and refinement calls
- **Purpose**: Complete animation pipeline execution
- **Sub-processes**:

**Data Reading Phase**:
```python
# Read sample data for script generation
data = self.animation.readData(
    t=region_params['t_list'][0],
    x_range=region_params['x_range'], 
    y_range=region_params['y_range'],
    z_range=region_params['z_range'],
    q=region_params['quality'],
    flip_axis=region_params['flip_axis'],
    transpose=region_params['transpose']
)
```

**Script Generation Phase**:
```python
# Generate visualization script based on render mode
if render_mode == 0 and not needs_velocity:  # Flat temperature/salinity
    self.animation.generateScript(input_names, kf_interval, dims, mesh_type, 
                                world_bbx_len, cam, tf_range, template, outfile)
elif render_mode == 0 and needs_velocity:    # Flat with streamlines
    self.animation.generateScriptStreamline(input_names, kf_interval, dims, 
                                          mesh_type, world_bbx_len, cam, tf_range, outfile)
elif render_mode == 2:                       # Spherical projection
    self.animation.generateScript(input_names, kf_interval, dims, "structuredSpherical",
                                world_bbx_len, cam, tf_range, "rotate", outfile)
```

**Raw Data Processing**:
```python
# Save processed data files for rendering
if not needs_velocity:
    self.animation.saveRawFilesByVisusRead(t_list, x_range, y_range, z_range, 
                                         q, flip_axis, transpose, output_dir)
else:
    self.animation.saveVTKFilesByVisusRead(velocity_u_url, velocity_v_url, 
                                         velocity_w_url, field, t_list, x_range, 
                                         y_range, z_range, q, flip_axis, transpose, output_dir)
```

**Rendering Phase**:
```python
# Execute rendering based on mode
self.render_animation(render_mode, needs_velocity, render_file_path)
# This internally calls:
# - renderTaskOffline() for standard visualization
# - renderTaskOfflineVTK() for streamline visualization
```

**Animation Creation**:
```python
# Convert frames to GIF animation
create_animation_from_frames(output_frames_dir, animation_name, format="gif")
```

##### 4. **evaluate_animation(animation_info, phenomenon_description, region_params)**
- **Triggered by**: After successful animation generation
- **Purpose**: AI-powered animation quality assessment
- **Process**:
  - Samples key frames from animation
  - Uses OpenAI Vision API to analyze oceanographic accuracy
  - Provides scientific evaluation and suggestions
- **AI Integration**: GPT-4 Vision analyzes animation frames for scientific validity

##### 5. **get_custom_animation_guidance(user_input=None)**
- **Triggered by**: User refinement requests ("y" or "g" responses)
- **Purpose**: Generate intelligent parameter modifications
- **AI Integration**: 
  - Analyzes conversation history and current animation
  - Suggests specific parameter adjustments
  - Can recommend completely new approaches
- **Output**: Modification strategy with new parameters

```python
# Example guidance output
{
    "action": "modify",
    "modification_type": "parameters", 
    "parameters": {
        "x_range": [800, 2800],    # Expanded spatial coverage
        "quality": -4,             # Higher resolution
        "t_list": [0, 2, 4, 6, 8, 10]  # More time steps
    },
    "description": "Expanding spatial coverage for better context"
}
```

##### 6. **validate_region_params(region_params)**
- **Triggered by**: Before every animation generation
- **Purpose**: Ensure parameters are within valid dataset bounds
- **Process**: Caps coordinates to maximum dataset dimensions
- **Safety**: Prevents invalid data requests that would crash rendering

### API Endpoints

#### POST /api/chat
**Purpose**: Main conversation endpoint mirroring Agent.py's run_conversation() flow

**Request Body**:
```json
{
    "message": "User input text",
    "action": "start|continue_conversation|provide_guidance"
}
```

**API Call Flow**:

1. **action: "start"** → Phenomenon selection
   ```python
   # No Agent function called, returns predefined phenomena list
   response = {
       "type": "dataset_selection",
       "phenomena": [
           {"id": "1", "name": "Agulhas Ring Current - Temperature"},
           {"id": "2", "name": "Agulhas Ring Current - Salinity with Streamlines"},
           {"id": "3", "name": "Mediterranean Sea Current - Temperature"},
           {"id": "4", "name": "Mediterranean Sea Current - Salinity with Streamlines"},
           {"id": "0", "name": "Custom Description"}
       ]
   }
   ```

2. **action: "continue_conversation"** → Animation generation pipeline
   ```python
   # Step 1: Convert free-text description to parameters
   # The API expects the phenomenon or custom description as free-text.
   region_params = agent.get_region_from_description(phenomenon_text)

   # Step 2: Check for existing animation
   existing_animation = agent.find_existing_animation(region_params)

   # Step 3A: Reuse existing animation
   if existing_animation["exists"]:
       print(f"Found existing animation. Reusing: {existing_animation['output_base']}")
       animation_info = existing_animation

   # Step 3B: Generate new animation
   else:
       animation_info = agent.generate_animation(region_params, phenomenon_text)
   ```
   # Step 4: Evaluate animation quality
   evaluation = agent.evaluate_animation(animation_info, phenomenon, region_params)
   ```

3. **action: "continue_conversation"** → Refinement handling
   ```python
   user_response = user_message.lower().strip()
   
   if user_response == "y":  # LLM-driven improvement
       guidance = agent.get_custom_animation_guidance()
       adjusted_params = apply_guidance(guidance, current_params)
       new_animation = agent.generate_animation(adjusted_params, phenomenon)
       
   elif user_response == "g":  # User guidance request
       # Prompt user for specific guidance
       
   elif user_response == "n":  # User satisfied
       # End conversation with final LLM comment
   ```

4. **action: "provide_guidance"** → User-directed refinement
   ```python
   # Process user's specific guidance
   guidance = agent.get_custom_animation_guidance(user_message)
   adjusted_params = apply_guidance(guidance, current_params)
   refined_animation = agent.generate_animation(adjusted_params, phenomenon)
   evaluation = agent.evaluate_animation(refined_animation, phenomenon, adjusted_params)
   ```

**Response Types**:
```json
// Dataset selection
{
    "type": "dataset_selection",
    "message": "Welcome message",
    "phenomena": [{"id": "1", "name": "Agulhas Ring Current - Temperature"}]
}

// Animation generated  
{
    "type": "animation_generated",
    "message": "Found existing animation. Reusing: /path/to/animation OR Animation generated successfully!",
    "animation_path": "/api/animations/animation_xxx/Animation/animation_xxx.gif",
    "evaluation": "AI-generated scientific analysis of animation quality",
    "continue_prompt": "Would you like me to generate modified animation (y), or provide specific guidance (g), or finish (n)?"
}

// Animation refined
{
    "type": "animation_refined", 
    "message": "Refined animation generated successfully!",
    "animation_path": "Updated animation path",
    "evaluation": "Updated AI analysis"
}

// Conversation end
{
    "type": "conversation_end",
    "message": "Final animation available with LLM farewell",
    "animation_path": "Final animation path"
}
```

#### GET /api/animations/<path:animation_path>
**Purpose**: Serves animation files and individual frames from Agent.py output

**Agent.py Integration**: 
- Serves files from animation directories created by `generate_animation()`
- Handles both GIF animations and individual PNG frames
- Provides access to complete animation folder structure

**Examples**:
```
/api/animations/animation_1226-3923-0_2394-4726-90_0-216-24_-6_salinity_True/Rendered_frames/img_kf0f0.png
/api/animations/animation_1226-3923-0_2394-4726-90_0-216-24_-6_salinity_True/Animation/animation.gif
```

### Data Processing Pipeline

The Agent.py backend implements a sophisticated oceanographic data processing pipeline that transforms raw simulation data into compelling visualizations.

#### 1. **Data Source Configuration**
**Function**: `setup_data_source(field_name)`
**Purpose**: Configure data access URLs based on phenomenon type

```python
# Temperature/Salinity data sources
temperature_url = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/theta_llc2160_x_y_depth.idx"
salinity_url = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx"

# Velocity data sources (for streamlines)  
velocity_u_url = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
velocity_v_url = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
velocity_w_url = "pelican://osg-htc.org/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
```

#### 2. **Coordinate System Transformation**
**Function**: `geographic_to_dataset_coords(lon, lat)`
**Purpose**: Convert geographic coordinates to dataset grid coordinates

```python
# Example transformation for Agulhas Ring Current
# Geographic: (-35°S to -28°S, 15°E to 25°E)
# Dataset coords: x_range=[1226, 3923], y_range=[2394, 4726]

def geographic_to_dataset_coords(self, lon, lat):
    # Convert longitude (-180 to 180) to x coordinate (0 to x_max)
    x = int((lon + 180) / 360 * self.animation.x_max)
    # Convert latitude (-90 to 90) to y coordinate (0 to y_max)  
    y = int((lat + 90) / 180 * self.animation.y_max)
    return x, y
```

#### 3. **Data Reading and Preprocessing**
**Function**: `self.animation.readData(t, x_range, y_range, z_range, q, flip_axis, transpose)`
**Purpose**: Extract and preprocess oceanographic data from massive datasets

```python
# Read 4D oceanographic data (x, y, z, t)
data = self.animation.readData(
    t=0,                          # Time step index
    x_range=[1226, 3923],         # Longitude range in grid coordinates
    y_range=[2394, 4726],         # Latitude range in grid coordinates  
    z_range=[0, 90],              # Depth range (surface to 90m)
    q=-6,                         # Quality/compression level
    flip_axis=2,                  # Coordinate system orientation
    transpose=False               # Data layout optimization
)

# Data shape: [depth_levels, latitude_points, longitude_points]
# Example: [91, 2333, 2698] = 574M data points per timestep
```

#### 4. **Visualization Script Generation**
**Agent.py generates different visualization scripts based on phenomenon requirements:**

**Standard Temperature/Salinity Visualization**:
```python
self.animation.generateScript(
    input_names=["frame_000.raw", "frame_001.raw", ...],    # Raw data files
    kf_interval=1,                                          # Keyframe interval
    dims=[2698, 2333, 91],                                 # Data dimensions [x,y,z]
    mesh_type="structured",                                 # Grid type
    world_bbx_len=10,                                      # Bounding box size
    cam=[4.84619, -3.10767, -8.16708, 0.0138851, 0.607355, 0.794303, -0.00317818, 0.794402, -0.607375],  # Camera position
    tf_range=[min_value, max_value],                       # Transfer function range
    template="fixedCam",                                   # Camera movement
    outfile="animation_script"                             # Output script name
)
```

**Streamline Visualization** (for velocity fields):
```python
self.animation.generateScriptStreamline(
    input_names=["velocity_000.vtk", "velocity_001.vtk", ...],  # VTK files with vector data
    kf_interval=1,
    dims=[2698, 2333, 91], 
    mesh_type="streamline",                                      # Streamline rendering
    world_bbx_len=10,
    cam=[-10, -5, 20, 0.56, 0.42, -0.71, 0, 0.43, -0.46],     # Optimized for flow visualization
    tf_range=[min_velocity, max_velocity],
    template="fixedCam"
)
```

**Spherical Projection** (global ocean view):
```python
self.animation.generateScript(
    input_names=["frame_000.raw", "frame_001.raw", ...],
    kf_interval=1,
    dims=[2698, 2333, 91],
    mesh_type="structuredSpherical",                            # Spherical mapping
    world_bbx_len=10, 
    cam=[-30, 0, 0, 1, 0, 0, 0, 0, -1],                       # Spherical camera
    tf_range=[min_value, max_value],
    template="rotate",                                          # Rotating animation
    s=45, e=135, dist=25,                                      # Rotation parameters
    bgImg="land.png"                                           # Earth background
)
```

#### 5. **Raw Data File Generation**
**Purpose**: Convert processed data into formats suitable for high-performance rendering

**For Standard Visualization**:
```python
self.animation.saveRawFilesByVisusRead(
    t_list=[0, 4, 8, 12, 16, 20, 24],                         # Time steps to process
    x_range=[1226, 3923],                                      # Spatial bounds
    y_range=[2394, 4726], 
    z_range=[0, 90],
    q=-6,                                                      # Compression quality
    flip_axis=2,                                               # Data orientation
    transpose=False,
    output_dir="/path/to/Out_text"                             # Raw file directory
)
# Generates: frame_000.raw, frame_001.raw, frame_002.raw, ...
```

**For Streamline Visualization**:
```python
self.animation.saveVTKFilesByVisusRead(
    eastwest_velocity_url,                                     # U component URL
    northsouth_velocity_url,                                   # V component URL  
    vertical_velocity_url,                                     # W component URL
    scalar_field_url,                                          # Temperature/Salinity for coloring
    t_list=[0, 4, 8, 12, 16, 20, 24],
    x_range=[1226, 3923], 
    y_range=[2394, 4726],
    z_range=[0, 90],
    q=-6,
    flip_axis=2,
    transpose=False,
    output_dir="/path/to/Out_text"
)
# Generates: velocity_000.vtk, velocity_001.vtk, velocity_002.vtk, ...
```

#### 6. **High-Performance Rendering**
**Function**: `render_animation(render_mode, needs_velocity, render_file_path)`
**Purpose**: Execute GPU-accelerated visualization rendering

```python
def render_animation(self, render_mode, needs_velocity, render_file_path):
    if render_mode == 0 and not needs_velocity:
        # Standard scalar field rendering
        self.animation.renderTaskOffline(render_file_path)
    elif render_mode == 0 and needs_velocity:  
        # Vector field streamline rendering
        self.animation.renderTaskOfflineVTK(render_file_path)
    elif render_mode == 2:
        # Spherical projection rendering
        self.animation.renderTaskOffline(render_file_path)
        
# Generates: img_kf0f0.png, img_kf0f1.png, img_kf1f1.png, img_kf1f2.png, ...
# 120 total frames (60 keyframes × 2 interpolated frames each)
```

#### 7. **Animation Assembly**
**Function**: `create_animation_from_frames(frames_dir, animation_name, format="gif")`
**Purpose**: Combine individual frames into smooth animation

```python
def create_animation_from_frames(frames_dir, animation_name, format="gif"):
    # Collect all PNG frames in sequence
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "img_kf*.png")))
    
    # Create optimized GIF animation
    # - Frame rate: 8 FPS for smooth oceanographic motion
    # - Optimization: Palette-based compression for web delivery
    # - Loop: Infinite for continuous playback
    
    # Output: animation.gif (typically 5-15MB for web streaming)
```

### Rendering Engine Architecture

The visualization system uses a multi-stage rendering pipeline optimized for scientific accuracy and performance:

#### Stage 1: **Data Preprocessing**
- **Coordinate transformation**: Geographic → Dataset grid coordinates
- **Temporal sampling**: Extract specific time steps from 4D datasets
- **Spatial cropping**: Focus on region of interest
- **Quality scaling**: Balance detail vs. performance

#### Stage 2: **Visualization Configuration** 
- **Transfer functions**: Map data values to colors/opacity
- **Camera positioning**: Optimal viewpoints for each phenomenon
- **Lighting setup**: Enhance 3D structure visibility
- **Mesh generation**: Create renderable geometry from data

#### Stage 3: **GPU Rendering**
- **Volume rendering**: Direct volumetric visualization of scalar fields
- **Streamline integration**: Particle tracing through velocity fields
- **Texture mapping**: Apply colormaps and transparency
- **Anti-aliasing**: Smooth visual quality

#### Stage 4: **Animation Generation**
- **Keyframe interpolation**: Smooth temporal transitions
- **Frame optimization**: Compress for web delivery
- **Format conversion**: PNG sequences → GIF animations

### Performance Optimizations

#### Data Access Optimization
- **Pelican storage**: High-performance distributed data access
- **Compressed formats**: IDX/raw files for efficient I/O
- **Spatial indexing**: Quick region-of-interest extraction
- **Temporal caching**: Reuse processed time steps

#### Rendering Optimization  
- **GPU acceleration**: CUDA/OpenCL parallel processing
- **LOD systems**: Level-of-detail for interactive performance
- **Memory management**: Efficient handling of large datasets
- **Pipeline parallelism**: Overlap I/O and computation

#### Web Delivery Optimization
- **Progressive loading**: Stream frames as available
- **Compression**: Optimized GIF encoding
- **Caching**: Browser and server-side frame caching
- **CDN ready**: Optimized for content distribution networks

This pipeline enables the system to transform terabytes of raw oceanographic simulation data into compelling, scientifically accurate visualizations that can be delivered efficiently through a web browser interface.

### Conversation State Management

The API maintains conversation state that mirrors Agent.py's internal conversation flow:

```python
conversation_state = {
    'step': 'start|phenomenon_selection|conversation_loop|awaiting_guidance',
    # 'phenomenon' holds the most recent free-text description or selected phenomenon name
    'phenomenon': 'Phenomenon description string',
    'region_params': 'Current animation parameters',
    'animation_info': 'Current animation metadata'
}
```

This state enables the web interface to seamlessly continue conversations across multiple API calls, maintaining the same interactive experience as the original CLI Agent.py tool.

## Performance Optimization

### Frame Discovery Optimization
- **Parallel Checking**: Multiple frame existence checks in parallel
- **Smart Caching**: Frame metadata cached to avoid redundant checks
- **Adaptive Timing**: Discovery intervals adjust based on animation type
- **Memory Management**: Efficient frame loading and garbage collection

### Network Optimization
- **HEAD Requests**: Lightweight frame existence validation
- **Progressive Loading**: Frames loaded as needed for smooth UX
- **Compression**: Optimized PNG compression for faster loading
- **CDN Ready**: Static file serving optimized for distribution

### Browser Compatibility
- **Modern Browsers**: Chrome, Firefox, Safari, Edge (latest versions)
- **Progressive Enhancement**: Graceful degradation for older browsers
- **Mobile Optimization**: Touch-friendly controls and responsive design
- **Performance Monitoring**: Client-side performance tracking

## Troubleshooting

### Common Issues

#### Animation Controls Not Visible
- **Check Console**: Look for "Animation controls shown" message
- **Browser DevTools**: Verify element exists with `document.getElementById('animationControls')`
- **CSS Loading**: Ensure style.css loads properly
- **Cache Clear**: Hard refresh (Ctrl+F5) to clear browser cache

#### Frame Discovery Issues
- **Path Verification**: Check animation folder structure matches expected pattern
- **Network Errors**: Verify server can serve files from animation directories
- **Naming Convention**: Ensure frames follow `img_kfXfY.png` pattern
- **Permissions**: Check file system permissions for animation directories

#### Performance Issues
- **Frame Loading**: Reduce animation quality settings for faster loading
- **Browser Memory**: Close other tabs to free up browser memory
- **Network Speed**: Consider local deployment for faster frame loading
- **Hardware Acceleration**: Enable browser hardware acceleration

#### API Connection Issues
- **Server Status**: Verify Flask server is running on correct port
- **CORS Settings**: Check cross-origin resource sharing configuration
- **API Keys**: Ensure OpenAI API key is properly configured
- **Network Firewall**: Verify no firewall blocking local connections

### Debug Information

#### Enable Verbose Logging
```javascript
// In browser console
localStorage.setItem('debug', 'true');
location.reload();
```

#### Check Animation State
```javascript
// In browser console
console.log('Frame Player State:', chatInterface.framePlayer);
console.log('Animation Controls:', document.getElementById('animationControls'));
```

#### Verify Frame Discovery
```javascript
// Check frame loading progress
console.log('Discovered Frames:', chatInterface.framePlayer.frames.length);
console.log('Current Frame:', chatInterface.framePlayer.currentFrame);
```

## Development

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run in development mode
export FLASK_ENV=development
python src/app.py
```

### Code Structure

#### Frontend Architecture
- **chat.js**: Main application logic and animation player
- **style.css**: Dark theme styling and responsive design
- **index.html**: Single-page application structure

#### Backend Architecture
- **app.py**: Flask application and route definitions
- **routes.py**: API endpoint implementations

### Adding New Features

#### New Animation Type
1. Update phenomenon list in API response
2. Add handling in `selectPhenomenon()` function
3. Implement parameter validation in backend
4. Test with various parameter combinations

#### Custom Frame Formats
1. Add format detection in `discoverFrames()`
2. Update pattern matching in frame discovery
3. Test with new naming conventions
4. Document format specifications

## Security Considerations

### API Security
- **Input Validation**: All user inputs sanitized and validated
- **Rate Limiting**: API endpoints protected against abuse
- **CORS Configuration**: Proper cross-origin resource sharing setup
- **Error Handling**: Sensitive information not exposed in error messages

### File Security
- **Path Traversal**: Animation paths validated to prevent directory traversal
- **File Type Validation**: Only PNG files served from animation directories
- **Access Control**: Restricted access to animation generation directories
- **Upload Security**: No user file uploads in current implementation

### Environment Security
- **API Key Protection**: OpenAI API keys stored securely outside repository
- **Environment Variables**: Sensitive configuration in environment files
- **Production Deployment**: Security headers and HTTPS recommended
- **Logging Security**: No sensitive data logged in production