"""
Parameter schemas for the multi-agent pipeline.
These define the structure of data flowing between agents.
"""
from pydantic import BaseModel, Field
from typing import Union, List, Literal, Optional


class RegionDict(BaseModel):
    """Spatial region definition"""
    x_range: List[int] = Field(..., description="[min, max] as absolute indices")
    y_range: List[int] = Field(..., description="[min, max] as absolute indices")
    z_range: List[int] = Field(..., description="[min, max] as absolute indices")
    
    # Optional geographic region specification
    geographic_region: Optional[str] = Field(
        None, 
        description="Named region (e.g., 'Gulf Stream') or lat/lon specification (e.g., 'lat:[30,45], lon:[-80,-50]')"
    )


class TimeRangeDict(BaseModel):
    """Temporal range definition"""
    start_timestep: int = Field(..., description="Starting timestep")
    end_timestep: int = Field(..., description="Ending timestep")
    num_frames: int = Field(24, description="Number of frames (default 24 for ~1 sec at 30fps)")


class RepresentationsDict(BaseModel):
    """Which visualization types to enable"""
    volume: bool = Field(False, description="Enable if visualizing scalar field")
    streamline: bool = Field(False, description="Enable if showing flow/currents")
    isosurface: bool = Field(False, description="Enable if showing land/boundaries")


class CameraDict(BaseModel):
    """Camera positioning"""
    position: Union[List[float], Literal["auto"]] = Field("auto", description="Camera position [x,y,z] or 'auto'")
    focal_point: Union[List[float], Literal["auto"]] = Field("auto", description="Focal point [x,y,z] or 'auto'")
    up: List[float] = Field([0.0, 1.0, 0.0], description="Up vector")


class TransferFunctionDict(BaseModel):
    """
    Transfer function for volume rendering.
    Note: colormap and opacity_profile are symbolic names that will be converted
    to actual color/opacity arrays by the animation agent before calling backend.
    """
    colormap: str = Field(..., description="Colormap name matching variable")
    opacity_profile: Literal["high", "medium", "low"] = Field("high", description="Opacity profile")
    # Optional explicit RGBPoints for VTK-style transfer functions: a flat list
    # of numbers [value, r, g, b, value, r, g, b, ...]. When provided the
    # animation generator should use these exact points for color mapping.
    RGBPoints: Optional[List[float]] = Field(None, description="Optional VTK-style RGBPoints list")

    # Optional explicit opacity control points; if present, these override
    # the opacity_profile symbolic name.
    opacity_values: Optional[List[float]] = Field(None, description="Optional opacity control points")

class IntegrationProperties(BaseModel):
    maxPropagation: float = Field(200.0)
    initialIntegrationStep: float = Field(0.3)
    minimumIntegrationStep: float = Field(0.05)
    maximumIntegrationStep: float = Field(1.0)
    integratorType: int = Field(2)
    integrationDirection: Literal["forward", "backward", "both"] = Field("both")
    maximumNumberOfSteps: int = Field(2000)
    terminalSpeed: float = Field(1.0e-12)
    computeVorticity: bool = Field(False)
    rotationScale: float = Field(1.0)
    surfaceStreamlines: bool = Field(False)


class SeedPlane(BaseModel):
    type: Literal["plane"] = Field("plane")
    enabled: bool = Field(True)
    position: str = Field("quarter_x")
    positionFraction: float = Field(0.25)
    origin: Optional[List[float]] = None
    point1: Optional[List[float]] = None
    point2: Optional[List[float]] = None
    xResolution: int = Field(20)
    yResolution: int = Field(20)
    center: Optional[List[float]] = None
    normal: List[float] = Field([1.0, 0.0, 0.0])


class SeedPoints(BaseModel):
    type: Literal["points"] = Field("points")
    enabled: bool = Field(False)
    numberOfPoints: int = Field(100)
    center: List[float] = Field([0.0, 0.0, 0.0])
    radius: float = Field(5.0)
    distribution: Literal["uniform", "random"] = Field("uniform")
    points: List[List[float]] = Field(default_factory=list)


class SeedLine(BaseModel):
    type: Literal["line"] = Field("line")
    enabled: bool = Field(False)
    point1: List[float] = Field([0.0, 0.0, 0.0])
    point2: List[float] = Field([10.0, 10.0, 10.0])
    resolution: int = Field(50)


class SeedRake(BaseModel):
    type: Literal["rake"] = Field("rake")
    enabled: bool = Field(False)
    startPoint: List[float] = Field([0.0, 0.0, 0.0])
    endPoint: List[float] = Field([10.0, 0.0, 0.0])
    numberOfLines: int = Field(10)
    perpendicularDirection: List[float] = Field([0.0, 1.0, 0.0])
    lineLength: float = Field(5.0)


class StreamlineProperties(BaseModel):
    color: List[float] = Field([1.0, 1.0, 1.0])
    lineWidth: float = Field(1.5)
    opacity: float = Field(1.0)
    renderAsTubes: bool = Field(True)
    tubeRadius: float = Field(0.1)
    tubeNumberOfSides: int = Field(6)
    tubeVaryRadius: Literal["off", "byScalar"] = Field("off")
    tubeRadiusFactor: float = Field(10.0)
    ambient: float = Field(0.3)
    diffuse: float = Field(0.7)
    specular: float = Field(0.1)
    specularPower: float = Field(10.0)
    specularColor: List[float] = Field([1.0, 1.0, 1.0])
    edgeVisibility: bool = Field(False)
    edgeColor: List[float] = Field([0.0, 0.0, 0.0])
    lighting: bool = Field(True)
    representation: Literal["surface", "wireframe", "points"] = Field("surface")
    backfaceCulling: bool = Field(False)
    frontfaceCulling: bool = Field(False)


class LookupTable(BaseModel):
    type: Literal["preset", "custom"] = Field("preset")
    presetName: Optional[str] = Field("Rainbow")
    customColors: List[float] = Field(default_factory=list)
    numberOfTableValues: int = Field(256)
    hueRange: List[float] = Field([0.667, 0.0])
    saturationRange: List[float] = Field([1.0, 1.0])
    valueRange: List[float] = Field([1.0, 1.0])
    alphaRange: List[float] = Field([1.0, 1.0])
    scale: Literal["linear", "log"] = Field("linear")
    ramp: Literal["linear", "s-curve"] = Field("linear")


class ColorMapping(BaseModel):
    colorByScalar: bool = Field(True)
    scalarField: str = Field("salinity")
    scalarMode: str = Field("usePointFieldData")
    scalarRange: List[float] = Field([34, 35.71])
    autoRange: bool = Field(True)
    colorSpace: Literal["RGB", "HSV"] = Field("RGB")
    nanColor: List[float] = Field([0.5, 0.0, 0.0])
    lookupTable: LookupTable = Field(default_factory=LookupTable)


class TransferFunc(BaseModel):
    enabled: bool = Field(False)
    range: List[float] = Field([0.0, 1.5])
    colors: List[float] = Field(default_factory=lambda: [0.0,0.0,1.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 1.0,1.0,0.0, 1.0,0.0,0.0])
    opacities: List[float] = Field(default_factory=lambda: [0.5,0.7,0.9,1.0,1.0])


class OutlineConfig(BaseModel):
    enabled: bool = Field(True)
    color: List[float] = Field([0.18431372549019609, 0.30980392156862744, 0.30980392156862744])
    lineWidth: float = Field(0.3)
    opacity: float = Field(1.0)


class StreamlineConfigDict(BaseModel):
    """Streamline visualization parameters (expanded)"""
    enabled: bool = Field(False, description="Enable streamlines")
    integrationProperties: IntegrationProperties = Field(default_factory=IntegrationProperties)
    seedPlane: SeedPlane = Field(default_factory=SeedPlane)
    seedPoints: SeedPoints = Field(default_factory=SeedPoints)
    seedLine: SeedLine = Field(default_factory=SeedLine)
    seedRake: SeedRake = Field(default_factory=SeedRake)
    streamlineProperties: StreamlineProperties = Field(default_factory=StreamlineProperties)
    colorMapping: ColorMapping = Field(default_factory=ColorMapping)
    transferFunc: TransferFunc = Field(default_factory=TransferFunc)
    outline: OutlineConfig = Field(default_factory=OutlineConfig)


class SurfaceProperties(BaseModel):
    """Surface rendering properties for isosurfaces"""
    color: List[float] = Field([0.518, 0.408, 0.216])
    opacity: float = Field(1.0)
    ambient: float = Field(0.3)
    diffuse: float = Field(0.7)
    specular: float = Field(0.2)
    specularPower: float = Field(20.0)
    specularColor: List[float] = Field([1.0, 1.0, 1.0])
    metallic: float = Field(0.0)
    roughness: float = Field(0.5)
    edgeVisibility: bool = Field(False)
    edgeColor: List[float] = Field([0.0, 0.0, 0.0])
    lighting: bool = Field(True)
    interpolation: Literal["flat", "gouraud", "phong"] = Field("gouraud")
    representation: Literal["surface", "wireframe", "points"] = Field("surface")
    backfaceCulling: bool = Field(False)
    frontfaceCulling: bool = Field(False)


class TextureTransform(BaseModel):
    """Texture transformation properties"""
    position: List[float] = Field([0.0, 0.0])
    scale: List[float] = Field([1.0, 1.0])
    rotation: float = Field(0.0)


class TextureConfig(BaseModel):
    """Texture mapping configuration"""
    enabled: bool = Field(True)
    textureFile: str = Field("/home/eliza89/PhD/codes/vis_user_tool/renderingApps/vtkApps/agulhaas_mask_land.png")
    mapMode: Literal["plane", "sphere", "cylinder"] = Field("plane")
    repeat: bool = Field(False)
    interpolate: bool = Field(True)
    edgeClamp: bool = Field(False)
    quality: Literal["default", "low", "medium", "high"] = Field("default")
    blendMode: Literal["replace", "modulate", "decal"] = Field("replace")
    transform: TextureTransform = Field(default_factory=TextureTransform)


class IsosurfaceColorMapping(BaseModel):
    """Color mapping for isosurfaces"""
    colorByScalar: bool = Field(False)
    scalarField: str = Field("salinity")
    scalarMode: str = Field("usePointFieldData")
    scalarRange: List[float] = Field([0.0, 0.005])
    autoRange: bool = Field(False)
    colorSpace: Literal["RGB", "HSV"] = Field("RGB")
    interpolateScalarsBeforeMapping: bool = Field(True)


class IsosurfaceTransferFunc(BaseModel):
    """Transfer function for isosurfaces"""
    enabled: bool = Field(False)
    range: List[float] = Field([0.0, 0.005])
    colors: List[float] = Field(default_factory=lambda: [0.518, 0.408, 0.216, 0.618, 0.508, 0.316])
    opacities: List[float] = Field(default_factory=lambda: [1.0, 1.0])


class IsosurfaceConfigDict(BaseModel):
    """Isosurface visualization parameters (full config matching test-vtk3.py)"""
    enabled: bool = Field(True)
    isoMethod: Literal["threshold", "marching_cubes"] = Field("threshold")
    thresholdRange: List[float] = Field([0.0, 0.005])
    isoValues: List[float] = Field([0.0025])
    numberOfContours: int = Field(1)
    computeNormals: bool = Field(True)
    computeGradients: bool = Field(False)
    computeScalars: bool = Field(True)
    arrayComponent: int = Field(0)
    surfaceProperties: SurfaceProperties = Field(default_factory=SurfaceProperties)
    texture: TextureConfig = Field(default_factory=TextureConfig)
    colorMapping: IsosurfaceColorMapping = Field(default_factory=IsosurfaceColorMapping)
    transferFunc: IsosurfaceTransferFunc = Field(default_factory=IsosurfaceTransferFunc)


class StreamlineHints(BaseModel):
    """High-level hints for customizing streamline visualization (LLM fills this)"""
    seed_density: Optional[Literal["sparse", "normal", "dense", "very_dense"]] = Field(
        None, description="How many streamlines to show (sparse=5x5, normal=20x20, dense=40x40, very_dense=80x80)"
    )
    integration_length: Optional[Literal["short", "medium", "long", "very_long"]] = Field(
        None, description="How far streamlines propagate (short=100, medium=200, long=400, very_long=800)"
    )
    color_by: Optional[Literal["velocity_magnitude", "solid_color", "temperature", "salinity"]] = Field(
        None, description="What to color streamlines by"
    )
    solid_color: Optional[List[float]] = Field(
        None, description="If color_by='solid_color', RGB values [r,g,b] in range [0,1]"
    )
    tube_thickness: Optional[Literal["thin", "normal", "thick"]] = Field(
        None, description="Visual thickness of streamline tubes (thin=0.05, normal=0.1, thick=0.2)"
    )
    show_outline: Optional[bool] = Field(
        None, description="Whether to show bounding box outline around streamlines"
    )


class AnimationParameters(BaseModel):
    """Complete parameter set for animation generation"""
    variable: str = Field(..., description="Variable to visualize (must match dataset variables)")
    region: RegionDict
    time_range: TimeRangeDict
    representations: RepresentationsDict
    camera: Union[CameraDict, Literal["auto"]] = Field("auto")
    transfer_function: TransferFunctionDict
    # OpenVisus quality / downsampling control. Integer <= 0. q=0 => full res.
    quality: int = Field(-6, description="OpenVisus quality (q) integer <= 0; lower values mean coarser LOD")
    # URL mapping for dataset variables. Keys may include:
    #  - active_scalar_url: url of the scalar used by default for volume rendering
    #  - scalar_1_url, scalar_2_url, ... : urls for each scalar variable in dataset order
    #  - url_u, url_v, url_w : vector component urls (only present if dataset contains a vector variable)
    url: Optional[dict] = Field(None, description="Mapping of dataset variable urls (active_scalar_url, scalar_N_url, url_u/v/w)")
    
    # Make these OPTIONAL - only required if corresponding representation is enabled
    streamline_config: Optional[StreamlineConfigDict] = Field(None, description="Only required if streamline representation is enabled")
    isosurface_config: Optional[IsosurfaceConfigDict] = Field(None, description="Only required if isosurface representation is enabled")
    
    # High-level hints for customizing representations (LLM extracts these from natural language)
    streamline_hints: Optional[StreamlineHints] = Field(None, description="Natural language hints for streamline customization")
   
    
    # Confidence from LLM
    confidence: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in parameter extraction (0.0-1.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "variable": "salinity",
                "region": {
                    "x_range": [0.119, 0.253],
                    "y_range": [0.378, 0.501],
                    "z_range": [0.0, 1.0]
                },
                "time_range": {
                    "start_timestep": 2184,
                    "end_timestep": 2207,
                    "num_frames": 24
                },
                "representations": {
                    "volume": True,
                    "streamline": False,
                    "isosurface": False
                },
                "confidence": 0.9
            }
        }