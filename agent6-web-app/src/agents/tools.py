"""src/agents/tools.py
----------------------
LangChain tool wrappers for the legacy `PGAAgent` implementation.

These small adapter functions expose existing `PGAAgent` methods as
callable tools that can be used by LangChain agents or by the local
multi-agent orchestrator. Each tool is intentionally conservative and
defensive: if the underlying `PGAAgent` does not implement a specific
helper, the wrapper either falls back to a permissive default or
returns an error-like dict so callers can handle the fallback.

Typical usage:
    - `set_agent(pga)` to register the global agent instance used by tools
    - call the wrapped functions directly, or use them as LangChain tools

Notes:
    - Keep these wrappers thin — they should not duplicate the agent's
        domain logic, only adapt interfaces and provide small fallbacks.
"""
import OpenVisus as ov
from PIL import Image
from langchain.tools import tool
import os
import sys
import rasterio
from rasterio.transform import rowcol
import xarray as xr
import numpy as np
from typing import Optional, List, Dict, Union


# Global agent instance (reuse for all tool calls)
_agent_instance = None

def set_agent(agent_instance):
    """Register a PGAAgent instance for subsequent tool calls.

    This sets a module-global pointer that the tool wrappers will use to
    delegate work to the real implementation in `Agent.PGAAgent`.

    Args:
        agent_instance: An initialized `PGAAgent` object.

    Side effects:
        Subsequent calls to the tool wrappers will use this agent. Call
        this once during application initialization (for example from
        `AnimationAgent.__init__`).
    """
    global _agent_instance
    _agent_instance = agent_instance

def get_agent():
    """Return the registered `PGAAgent` instance.

    Raises:
        RuntimeError: if no PGAAgent has been registered via
            `set_agent()`.

    Returns:
        The `PGAAgent` instance previously passed to `set_agent()`.
    """
    global _agent_instance
    if _agent_instance is None:
        raise RuntimeError("PGAAgent not initialized. Call set_agent() first.")
    return _agent_instance

@tool
def generate_animation_from_params(params: dict, description: str) -> dict:
    """Generate an animation using the given parameters.

    This thin wrapper calls `PGAAgent.generate_animation(params, description)`
    and returns whatever the agent implementation produces (typically a
    dict with an `animation_path` or `path` key and status information).

    Args:
        params: Visualization parameter dict (x/y ranges, t_list, variable, etc.).
        description: Original user query or explanation string used for logging.

    Returns:
        Result dict from the underlying `PGAAgent.generate_animation` call.
    """
    agent = get_agent()
    return agent.generate_animation(params, description)

@tool
def evaluate_animation_quality(animation_info: dict, description: str, params: dict) -> str:
    """Evaluate the quality of a generated animation.

    Delegates to `PGAAgent.evaluate_animation` which may use internal
    heuristics or an ML/vision model to score or describe the animation.

    Args:
        animation_info: Metadata about the animation (paths, frames, etc.).
        description: Human description of what the animation should show.
        params: Parameters used to generate the animation.

    Returns:
        A human-readable evaluation string or structured score information.
    """
    agent = get_agent()
    return agent.evaluate_animation(animation_info, description, params)

@tool
def get_dataset_summary(dataset: dict) -> dict:
    """Produce a concise summary of the provided dataset.

    Calls `PGAAgent.summarize_dataset` to build a description including
    available variables, spatial/temporal extents, and recommended
    visualization suggestions.

    Args:
        dataset: Dataset metadata dictionary expected by the PGAAgent.

    Returns:
        A dict summarizing the dataset (variables, extents, suggested views, ...).
    """
    agent = get_agent()
    return agent.summarize_dataset(dataset)

@tool
def find_existing_animation(region_params: dict) -> dict:
    """Check for an existing animation matching given parameters.

    This wrapper calls `PGAAgent.find_existing_animation` and returns a
    dict containing at minimum an `'exists'` boolean and, when true,
    path information under keys such as `'path'` or `'animation_path'`.

    Args:
        region_params: Parameter dict describing the animation region/time/variable.

    Returns:
        A dict with keys like `'exists'` (bool) and path fields when found.
    """
    agent = get_agent()
    return agent.find_existing_animation(region_params)


# ============================================================================
# GEOGRAPHIC CONVERSION TOOLS
# ============================================================================

# Named geographic regions database (approximate lat/lon bounds)
NAMED_REGIONS = {
    "gulf stream": {"lat": [30, 45], "lon": [-80, -50]},
    "agulhas current": {"lat": [-40, -30], "lon": [15, 40]},
    "agulhas": {"lat": [-50, -30], "lon": [5, 40]},
    "kuroshio current": {"lat": [20, 40], "lon": [120, 150]},
    "mediterranean sea": {"lat": [30, 46], "lon": [-6, 36]},
    "red sea": {"lat": [12, 30], "lon": [32, 44]},
    "persian gulf": {"lat": [23, 31], "lon": [48, 57]},
    "gulf of mexico": {"lat": [18, 31], "lon": [-98, -80]},
    "caribbean sea": {"lat": [9, 22], "lon": [-88, -60]},
    "north atlantic": {"lat": [20, 60], "lon": [-70, 0]},
    "south atlantic": {"lat": [-60, 0], "lon": [-60, 20]},
    "north pacific": {"lat": [0, 60], "lon": [120, -70]},
    "south pacific": {"lat": [-60, 0], "lon": [120, -70]},
    "indian ocean": {"lat": [-60, 30], "lon": [20, 120]},
    "southern ocean": {"lat": [-90, -50], "lon": [-180, 180]},
    "arctic ocean": {"lat": [66, 90], "lon": [-180, 180]},
    "bering sea": {"lat": [51, 66], "lon": [-180, -160]},
    "sea of japan": {"lat": [33, 52], "lon": [127, 142]},
    "south china sea": {"lat": [0, 23], "lon": [99, 121]},
    "tasman sea": {"lat": [-50, -25], "lon": [145, 165]},
    "coral sea": {"lat": [-30, -10], "lon": [145, 160]},
    "bay of bengal": {"lat": [5, 22], "lon": [80, 95]},
    "arabian sea": {"lat": [0, 30], "lon": [50, 77]},
    "labrador sea": {"lat": [53, 65], "lon": [-60, -45]},
    "hudson bay": {"lat": [51, 65], "lon": [-95, -77]},
    "baltic sea": {"lat": [53, 66], "lon": [10, 30]},
    "black sea": {"lat": [41, 47], "lon": [27, 42]},
    "caspian sea": {"lat": [36, 47], "lon": [46, 55]},
    # Add more regions as needed
}


class GeographicConverter:
    """Handles conversion between geographic coordinates and grid indices"""
    
    def __init__(self, geographic_info_file: str):
        """
        Initialize converter with lat/lon data from NetCDF file.
        
        Args:
            geographic_info_file: Path to .nc file with lat/lon data
        """
        try:
            # Resolve relative paths from src/datasets/ directory
            if not os.path.isabs(geographic_info_file):
                tools_dir = os.path.dirname(os.path.abspath(__file__))  # src/agents/
                src_dir = os.path.dirname(tools_dir)                    # src/
                datasets_dir = os.path.join(src_dir, 'datasets')        # src/datasets/
                geographic_info_file = os.path.join(datasets_dir, geographic_info_file)
                geotiff_file = os.path.join(datasets_dir, 'BlueMarbleNG-TB_2004-12-01_rgb_3600x1800.TIFF')
                landmask_folder = os.path.join(datasets_dir, 'land_masks')
            if not os.path.exists(geographic_info_file):
                raise FileNotFoundError(f"Geographic info file not found: {geographic_info_file}")
            
            self.ds = xr.open_dataset(geographic_info_file)
            self.lat_center = self.ds["latitude"].values
            self.lon_center = self.ds["longitude"].values
            self.geotiff_file = geotiff_file
            self.landmask_folder = landmask_folder
            
            # Create land_masks directory if it doesn't exist
            os.makedirs(self.landmask_folder, exist_ok=True)

        except Exception as e:
            raise ValueError(f"Failed to load geographic info file {geographic_info_file}: {str(e)}")
        
    def latlon_to_indices(self, lat_range: List[float], lon_range: List[float]) -> Dict:
        """
        Convert lat/lon range to x/y indices.
        
        Args:
            lat_range: [min_lat, max_lat] in degrees
            lon_range: [min_lon, max_lon] in degrees
            
        Returns:
            {
                'x_range': [x_min, x_max],
                'y_range': [y_min, y_max],
                'actual_lat_range': [actual_min_lat, actual_max_lat],
                'actual_lon_range': [actual_min_lon, actual_max_lon]
            }
        """
        # Create mask for points within the lat/lon range
        mask = (
            (self.lat_center >= lat_range[0]) & (self.lat_center <= lat_range[1]) &
            (self.lon_center >= lon_range[0]) & (self.lon_center <= lon_range[1])
        )
        
        y_indices, x_indices = np.where(mask)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError(f"No data found in lat range {lat_range}, lon range {lon_range}")
        
        x_min = int(x_indices.min())
        x_max = int(x_indices.max()) + 1
        y_min = int(y_indices.min())
        y_max = int(y_indices.max()) + 1
        
        # Get actual lat/lon bounds of the selected region
        lat_sub = self.lat_center[y_min:y_max, x_min:x_max]
        lon_sub = self.lon_center[y_min:y_max, x_min:x_max]
        
        return {
            'x_range': [x_min, x_max],
            'y_range': [y_min, y_max],
            'actual_lat_range': [float(lat_sub.min()), float(lat_sub.max())],
            'actual_lon_range': [float(lon_sub.min()), float(lon_sub.max())]
        }
    
    def indices_to_latlon(self, x_range: List[int], y_range: List[int]) -> Dict:
        """
        Convert x/y indices to lat/lon range.
        
        Args:
            x_range: [x_min, x_max]
            y_range: [y_min, y_max]
            
        Returns:
            {
                'lat_range': [min_lat, max_lat],
                'lon_range': [min_lon, max_lon]
            }
        """
        lat_sub = self.lat_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        lon_sub = self.lon_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        
        return lat_sub, lon_sub

    def create_land_mask_from_sea_coordinates(self,x_range, y_range, field_url="https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
                            z=[0,1], 
                            sea_threshold=0.01): # Values below this are land areas (will be kept visible)
        """
        Create a land mask from sea coordinates by identifying and masking sea areas in the provided GeoTIFF.
        Steps:
        1. Extract oceanographic data and get lat/lon coordinates  
        2. Identify land pixels where field values ≈ 0 (for salinity data)
        3. Convert sea lat/lon coordinates to GeoTIFF pixel coordinates
        4. Make sea areas transparent, keep land areas opaque
        
        Args:
            x_range, y_range: Data coordinate ranges
            field_url: Oceanographic data URL (salinity, temperature, etc.)
            lat_center, lon_center: LLC2160 coordinate arrays
            geotiff_path: Path to GeoTIFF file
            output_dir: Directory to save mask files
            region_name: Name prefix for saved files
            z, quality: OpenVisus parameters
            sea_threshold: Values below this threshold are LAND areas (kept visible)
            figsize: Figure size for visualization
        
        Returns:
            Dictionary with mask data and saved file paths
        """
        
        # Create output directory
        # os.makedirs(output_dir, exist_ok=True)
        output_dir = self.landmask_folder
        
        # print(f"STEP 1: Extract Oceanographic Data and Coordinates")
        # print(f"Field URL: {field_url}")
        # print(f"X, Y range: x=[{x_range[0]}, {x_range[1]}], y=[{y_range[0]}, {y_range[1]}]")
        
        # Extract oceanographic data AND coordinate arrays
        lat_sub, lon_sub = self.indices_to_latlon(x_range, y_range)
        db = ov.LoadDataset(field_url)
        data_sub = db.read(x=x_range, y=y_range, z=z, quality=0)[0, :, :]
        
        # print(f"Data shape: {data_sub.shape}")
        # print(f"Data range: [{data_sub.min():.6f}, {data_sub.max():.6f}]")
        # print(f"Coordinate arrays shape: lat={lat_sub.shape}, lon={lon_sub.shape}")
        
        # CORRECTED: For salinity data - Land has ~0 values, Sea has high values (35-40 PSU)
        land_mask = np.abs(data_sub) <= sea_threshold  # True where land (≈0 salinity)
        sea_mask = ~land_mask  # True where sea (high salinity values)
        
        # print(f"Land threshold: {sea_threshold} (values below this are land)")
        # print(f"Land pixels: {land_mask.sum()} / {land_mask.size} ({100*land_mask.sum()/land_mask.size:.1f}%)")
        # print(f"Sea pixels: {sea_mask.sum()} / {sea_mask.size} ({100*sea_mask.sum()/sea_mask.size:.1f}%)")
        
        # Get lat/lon coordinates of SEA areas (these will be made transparent)
        sea_lat_coords = lat_sub[sea_mask]
        sea_lon_coords = lon_sub[sea_mask]
        
        # print(f"Sea coordinate ranges (will be made transparent):")
        # print(f"  Lat: [{sea_lat_coords.min():.3f}°, {sea_lat_coords.max():.3f}°]")
        # print(f"  Lon: [{sea_lon_coords.min():.3f}°, {sea_lon_coords.max():.3f}°]")
        
        # print(f"\nSTEP 2: Get GeoTIFF Crop for the Region")
        
        # Get overall region bounds for GeoTIFF cropping
        lat_min, lat_max = lat_sub.min(), lat_sub.max()
        lon_min, lon_max = lon_sub.min(), lon_sub.max()
        
        with rasterio.open(self.geotiff_file) as src:
            # print(f"GeoTIFF: {src.width} x {src.height} pixels")
            
            # Convert region bounds to pixel coordinates for cropping
            corners = [
                (lat_min, lon_min, "Bottom-Left"),
                (lat_max, lon_min, "Top-Left"), 
                (lat_min, lon_max, "Bottom-Right"),
                (lat_max, lon_max, "Top-Right")
            ]
            
            pixel_coords = []
            for lat, lon, corner_name in corners:
                row, col = rowcol(src.transform, lon, lat)
                pixel_coords.append((row, col))
                in_bounds = 0 <= row < src.height and 0 <= col < src.width
                print(f"  {corner_name:12} ({lat:7.3f}°, {lon:8.3f}°) → Row: {row:4}, Col: {col:4}")
            
            # Get pixel bounds for cropping
            rows = [coord[0] for coord in pixel_coords]
            cols = [coord[1] for coord in pixel_coords] 
            
            row_min = max(0, min(rows))
            row_max = min(src.height, max(rows))
            col_min = max(0, min(cols))
            col_max = min(src.width, max(cols))
            
            # print(f"GeoTIFF crop bounds: Rows [{row_min}, {row_max}], Cols [{col_min}, {col_max}]")
            
            # Read RGB crop
            crop_window = rasterio.windows.Window(col_min, row_min, 
                                                col_max - col_min, row_max - row_min)
            
            red = src.read(1, window=crop_window)
            green = src.read(2, window=crop_window) 
            blue = src.read(3, window=crop_window)
            
            rgb_crop = np.stack([red, green, blue], axis=2)
            crop_bounds = rasterio.windows.bounds(crop_window, src.transform)
            
            # print(f"RGB crop shape: {rgb_crop.shape}")
            # print(f"Crop geographic bounds: {crop_bounds}")
            
            # print(f"\nSTEP 3: Convert Sea Lat/Lon to GeoTIFF Pixels and Make Transparent")
            
            # Convert sea coordinates to pixel coordinates within the cropped region
            sea_pixel_rows = []
            sea_pixel_cols = []
            
            for lat, lon in zip(sea_lat_coords, sea_lon_coords):
                # Get global pixel coordinates
                global_row, global_col = rowcol(src.transform, lon, lat)
                
                # Convert to coordinates within the crop
                crop_row = global_row - row_min
                crop_col = global_col - col_min
                
                # Check if within crop bounds
                if 0 <= crop_row < rgb_crop.shape[0] and 0 <= crop_col < rgb_crop.shape[1]:
                    sea_pixel_rows.append(crop_row)
                    sea_pixel_cols.append(crop_col)
            
            # print(f"Found {len(sea_pixel_rows)} sea pixels within GeoTIFF crop (these will be made transparent)")
            
            # Create RGBA image
            rgba_result = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1], 4), dtype=np.uint8)
            
            # Normalize RGB to 0-255 range
            if rgb_crop.max() <= 1.0:
                rgb_uint8 = (rgb_crop * 255).astype(np.uint8)
            else:
                rgb_uint8 = rgb_crop.astype(np.uint8)
            
            # Start with all pixels opaque (showing full GeoTIFF)
            rgba_result[:, :, :3] = rgb_uint8
            rgba_result[:, :, 3] = 255  # All opaque initially
            
            # Make ONLY sea pixels transparent
            if len(sea_pixel_rows) > 0:
                sea_rows_array = np.array(sea_pixel_rows)
                sea_cols_array = np.array(sea_pixel_cols)
                
                # Make sea pixels transparent (removes ocean areas, keeps land)
                rgba_result[sea_rows_array, sea_cols_array, 3] = 0
            
            # # Count final results
            # transparent_count = (rgba_result[:, :, 3] == 0).sum()
            # opaque_count = (rgba_result[:, :, 3] == 255).sum()
            
        #     print(f"Final result:")
        #     print(f"  Transparent pixels (sea): {transparent_count:,} ({100*transparent_count/rgba_result[:,:,3].size:.1f}%)")
        #     print(f"  Opaque pixels (land): {opaque_count:,} ({100*opaque_count/rgba_result[:,:,3].size:.1f}%)")
        
        # print(f"\nSTEP 4: Save Results")
        
        # saved_files = []
        
        # Save land-only PNG
        land_png_filename = f"{x_range}_{y_range}_land.png"
        land_png_filepath = os.path.join(output_dir, land_png_filename)
        
        land_img = Image.fromarray(rgba_result, 'RGBA')
        land_img.save(land_png_filepath)

        return land_png_filepath
        # saved_files.append(land_png_filepath)
        
        # print(f"✓ Saved CORRECTED land mask: {land_png_filepath}")
        
        # # Save comparison visualization
        # comparison_filename = f"{region_name}.png"
        # comparison_filepath = os.path.join(output_dir, comparison_filename)
        
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # fig.suptitle(f'{region_name}: CORRECTED Land Mask (Keep Land, Remove Sea)', fontsize=14, fontweight='bold')
        
        # # Panel 1: Original oceanographic data
        # ax1 = axes[0,0]
        # im1 = ax1.imshow(data_sub, cmap="turbo", origin="lower") 
        # plt.colorbar(im1, ax=ax1, label='Salinity')
        # ax1.set_title('1. Salinity Data\n(Low=Land, High=Sea)')
        
        # # Panel 2: Land mask (white=land areas that will be kept)
        # ax2 = axes[0,1]
        # ax2.imshow(land_mask, cmap="gray", origin="lower")  # white=land (keep)
        # ax2.set_title('2. Land Areas (White=Will Be Kept)')
        
        # # Panel 3: Original GeoTIFF crop
        # ax3 = axes[1,0]
        # if rgb_crop.max() > 1:
        #     rgb_display = rgb_crop / 255.0
        # else:
        #     rgb_display = rgb_crop
        # ax3.imshow(rgb_display)
        # ax3.set_title('3. Original GeoTIFF')
        
        # # Panel 4: CORRECTED result - only land visible!
        # ax4 = axes[1,1]
        # ax4.imshow(rgba_result)
        # ax4.set_title('4. CORRECTED: Only Land Visible!')
        
        # for ax in axes.flat:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        
        # plt.tight_layout()
        # plt.savefig(comparison_filepath, dpi=300, bbox_inches='tight')
        # plt.close()
        # saved_files.append(comparison_filepath)
        
        # print(f"✓ Saved comparison: {comparison_filepath}")
        
        # Return results
        # result = {
        #     'approach': 'corrected_sea_coordinates_to_geotiff_pixels',
        #     'voxel_range': {'x': x_range, 'y': y_range},
        #     'coordinate_bounds': {
        #         'lat_min': lat_min, 'lat_max': lat_max,
        #         'lon_min': lon_min, 'lon_max': lon_max
        #     },
        #     'ocean_data': data_sub,
        #     'land_mask': land_mask,  # Fixed: land_mask is where salinity ≈ 0
        #     'sea_mask': sea_mask,    # Fixed: sea_mask is where salinity > threshold
        #     'sea_coordinates': {'lat': sea_lat_coords, 'lon': sea_lon_coords},
        #     'geotiff_crop': rgb_crop,
        #     'final_rgba': rgba_result,
        #     'sea_threshold': sea_threshold,
        #     'transparent_percentage': 100 * transparent_count / rgba_result[:,:,3].size,
        #     'saved_files': saved_files
        # }
        
        
        # print(f"{'='*70}")
        # print(f"Region: {region_name}")
        # print(f"Land threshold: {sea_threshold} (salinity below this = land)")
        # print(f"Sea areas made transparent: {result['transparent_percentage']:.1f}%")
        # print(f"Logic: Low salinity (≈0) = Land (kept), High salinity = Sea (removed)")
        # print(f"Output files: {len(saved_files)}")
        # for file_path in saved_files:
        #     file_size_kb = os.path.getsize(file_path) / 1024
        #     print(f"  - {os.path.basename(file_path)} ({file_size_kb:.1f} KB)")
        # print(f"Picture 4 should now show ONLY land areas! 🎯")
        # print(f"{'='*70}")
        
        # return result

def get_region_latlon(region_name: str) -> Optional[Dict[str, List[float]]]:
    """
    Get lat/lon bounds for a named region.
    
    Args:
        region_name: Name of region (case-insensitive)
        
    Returns:
        {'lat': [min, max], 'lon': [min, max]} or None if not found
    """
    region_name_lower = region_name.lower().strip()
    return NAMED_REGIONS.get(region_name_lower)


@tool
def convert_geographic_region(
    dataset: dict,
    query_region: Optional[str] = None,
    lat_range: Optional[List[float]] = None,
    lon_range: Optional[List[float]] = None
) -> Dict:
    """
    Convert geographic region specification to x/y indices.
    
    This tool is used when a dataset has geographic information and the user
    specifies a region by name (e.g., "Gulf Stream") or by lat/lon coordinates.
    
    Args:
        dataset: Dataset metadata dict
        query_region: Named region (e.g., "Gulf Stream", "Mediterranean Sea")
        lat_range: [min_lat, max_lat] in degrees
        lon_range: [min_lon, max_lon] in degrees
        
    Returns:
        {
            'status': 'success' | 'error',
            'x_range': [x_min_fraction, x_max_fraction],  # As fractions 0-1
            'y_range': [y_min_fraction, y_max_fraction],  # As fractions 0-1
            'x_range_absolute': [x_min, x_max],  # Absolute indices
            'y_range_absolute': [y_min, y_max],  # Absolute indices
            'lat_range': [actual_min_lat, actual_max_lat],
            'lon_range': [actual_min_lon, actual_max_lon],
            'message': str
        }
    """
    try:
        # Check if dataset has geographic info
        spatial_info = dataset.get('spatial_info', {})
        geo_info = spatial_info.get('geographic_info', {})
        
        if geo_info.get('has_geographic_info') != 'yes':
            return {
                'status': 'error',
                'message': 'Dataset does not have geographic information'
            }
        
        geo_file = geo_info.get('geographic_info_file')
        if not geo_file:
            return {
                'status': 'error',
                'message': 'Geographic info file not specified in dataset'
            }
        
        # Initialize converter
        converter = GeographicConverter(geo_file)
        
        # Determine lat/lon range
        if query_region:
            # Look up named region
            region_bounds = get_region_latlon(query_region)
            if not region_bounds:
                available_regions = list(NAMED_REGIONS.keys())[:10]
                return {
                    'status': 'error',
                    'message': f"Unknown region '{query_region}'. Available regions include: {', '.join(available_regions)}..."
                }
            lat_range = region_bounds['lat']
            lon_range = region_bounds['lon']
        
        if not lat_range or not lon_range:
            return {
                'status': 'error',
                'message': 'Must provide either query_region or both lat_range and lon_range'
            }
        
        # Convert to indices
        result = converter.latlon_to_indices(lat_range, lon_range)
        
        # Convert to fractions (0-1) for consistency with parameter schema
        dims = geo_info.get('dimensions', {})
        x_max_dim = dims.get('x', 8640)
        y_max_dim = dims.get('y', 6480)
        
        return {
            'status': 'success',
            'x_range': [result['x_range'][0] , result['x_range'][1]],
            'y_range': [result['y_range'][0], result['y_range'][1]],
            'x_range_absolute': result['x_range'],
            'y_range_absolute': result['y_range'],
            'lat_range': result['actual_lat_range'],
            'lon_range': result['actual_lon_range'],
            'message': f"Converted {query_region or 'lat/lon range'} to grid indices"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Geographic conversion failed: {str(e)}'
        }


@tool
def list_available_geographic_regions() -> Dict:
    """
    List all predefined geographic regions that can be used in queries.
    
    Returns:
        {
            'regions': List of region names,
            'count': Number of regions
        }
    """
    return {
        'regions': sorted(list(NAMED_REGIONS.keys())),
        'count': len(NAMED_REGIONS)
    }


@tool
def create_animation_dirs(params: dict) -> Dict:
    """
    Create a hierarchical animation directory structure under the PGAAgent ai_dir
    using the parameter values extracted by the ParameterExtractorAgent.

    Directory format (example):
    ai_dir/animation_{x0}_{y0}_{z0}-{x1}_{y1}_{z1}-{q}-{variable}-{active_scalar_name}/
        Out_text/
        GAD_text/
        Rendered_frames/

    Returns a dict with absolute paths for each created folder.
    """
    return create_animation_dirs_impl(params)


def create_animation_dirs_impl(params: dict) -> Dict:
    """Implementation of create_animation_dirs used internally by the codebase.

    The LangChain `@tool` wrapper above delegates to this implementation so
    internal callers can import and call the plain function without invoking
    the LangChain Tool wrapper object.
    """
    try:
        agent = get_agent()
    except Exception:
        agent = None

    # Determine base ai directory
    if agent and hasattr(agent, 'ai_dir') and agent.ai_dir:
        base_ai = os.path.abspath(agent.ai_dir)
    else:
        # Fallback to repo-relative ai_data
        base_ai = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_data'))

    region = params.get('region', {}) or {}
    x_range = region.get('x_range', [0, 0])
    y_range = region.get('y_range', [0, 0])
    z_range = region.get('z_range', [0, 0])
    q = params.get('quality', -6)
    variable = params.get('variable', 'unknown')

    # Determine active scalar name from params['url'] or dataset if available
    active_scalar_name = 'active_scalar'
    url_map = params.get('url', {}) or {}
    active_url = url_map.get('active_scalar_url')
    # Try to find a matching dataset variable id (if agent dataset available)
    if agent and hasattr(agent, 'dataset') and agent.dataset:
        for v in agent.dataset.get('variables', []):
            if active_url and v.get('url') == active_url:
                active_scalar_name = v.get('id') or v.get('name')
                break

    # Fallback: try to infer name from active_url
    if (not active_scalar_name or active_scalar_name == 'active_scalar') and active_url:
        active_scalar_name = os.path.splitext(os.path.basename(active_url))[0]

    # check which representations are being used #"representations": {
    #   "volume": true,
    #   "streamline": false,
    #   "isosurface": false
    # },
    representations = params.get('representations', {})
    # check which ones are true
    active_rep_name = ""
    active_representations = [k for k, v in representations.items() if v]
    if active_representations:
        active_rep_name +=  "_".join(active_representations)
    
    #  add time start end and num of frames "time_range": {
    #   "start_timestep": 0,
    #   "end_timestep": 2400,
    #   "num_frames": 100
    # }
    time_start = params.get('time_range', {}).get('start_timestep', 0)
    time_end = params.get('time_range', {}).get('end_timestep', 2400)
    num_frames = params.get('time_range', {}).get('num_frames', 100)

    # Clean components (remove spaces and os-unfriendly chars)
    def _clean(s: str) -> str:
        return str(s).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')

    # it should look like: animation_0-0-0_1-1-1_-6_temperature_temperature_volume_streamline_isosurface
    anim_dir_name = f"animation_{_clean(x_range[0])}-{_clean(y_range[0])}-{_clean(z_range[0])}_{_clean(x_range[1])}-{_clean(y_range[1])}-{_clean(z_range[1])}_{_clean(time_start)}-{_clean(time_end)}-{_clean(num_frames)}_{_clean(q)}_{_clean(variable)}_{_clean(active_scalar_name)}_{_clean(active_rep_name)}"

    # Place animations under ai_data/animations for consistent organization
    animations_root = os.path.join(base_ai, 'animations')
    try:
        os.makedirs(animations_root, exist_ok=True)
    except Exception:
        pass

    base_path = os.path.join(animations_root, anim_dir_name)
    out_text = os.path.join(base_path, 'Out_text')
    gad_text = os.path.join(base_path, 'GAD_text')
    rendered_frames = os.path.join(base_path, 'Rendered_frames')
    statistics = os.path.join(base_path, 'Statistics')

    for p in (base_path, out_text, gad_text, rendered_frames, statistics):
        try:
            os.makedirs(p, exist_ok=True)
            # Helpful debug log so callers can see that directories were created
            try:
                print(f"[create_animation_dirs] ensured directory: {p}")
            except Exception:
                pass
        except Exception:
            pass

    return {
        'animation_name': anim_dir_name,
        'base': base_path,
        'out_text': out_text,
        'gad_text': gad_text,
        'rendered_frames': rendered_frames,
        'statistics': statistics
    }