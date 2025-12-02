
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
    """Register a Agent instance for subsequent tool calls.

    This sets a module-global pointer that the tool wrappers will use to
    delegate work to the real implementation in `Agent.Agent`.

    Args:
        agent_instance: An initialized `Agent` object.

    """
    global _agent_instance
    _agent_instance = agent_instance

def get_agent():
    """Return the registered `Agent` instance.

    Raises:
        RuntimeError: if no Agent has been registered via
            `set_agent()`.

    Returns:
        The `Agent` instance previously passed to `set_agent()`.
    """
    global _agent_instance
    if _agent_instance is None:
        raise RuntimeError("Agent not initialized. Call set_agent() first.")
    return _agent_instance





# ============================================================================
# GEOGRAPHIC CONVERSION TOOLS
# ============================================================================

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
                
            if not os.path.exists(geographic_info_file):
                raise FileNotFoundError(f"Geographic info file not found: {geographic_info_file}")
            
            self.ds = xr.open_dataset(geographic_info_file)
            self.lat_center = self.ds["latitude"].values
            self.lon_center = self.ds["longitude"].values


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
        # Record the original requested ranges for transparency
        requested_lat = [float(lat_range[0]), float(lat_range[1])]
        requested_lon = [float(lon_range[0]), float(lon_range[1])]

        # Compute dataset bounds
        data_lat_min = float(np.nanmin(self.lat_center))
        data_lat_max = float(np.nanmax(self.lat_center))
        data_lon_min = float(np.nanmin(self.lon_center))
        data_lon_max = float(np.nanmax(self.lon_center))

        # Clamp (intersection) the requested ranges to the dataset bounds
        used_lat_min = max(requested_lat[0], data_lat_min)
        used_lat_max = min(requested_lat[1], data_lat_max)
        used_lon_min = max(requested_lon[0], data_lon_min)
        used_lon_max = min(requested_lon[1], data_lon_max)

        # If intersection is empty (used_min > used_max), snap to nearest boundary
        if used_lat_min > used_lat_max:
            if requested_lat[1] < data_lat_min:
                used_lat_min = used_lat_max = data_lat_min
            else:
                used_lat_min = used_lat_max = data_lat_max

        if used_lon_min > used_lon_max:
            if requested_lon[1] < data_lon_min:
                used_lon_min = used_lon_max = data_lon_min
            else:
                used_lon_min = used_lon_max = data_lon_max

        # Create mask for points within the used (clamped) lat/lon range
        mask = (
            (self.lat_center >= used_lat_min) & (self.lat_center <= used_lat_max) &
            (self.lon_center >= used_lon_min) & (self.lon_center <= used_lon_max)
        )

        y_indices, x_indices = np.where(mask)

        # If mask yielded no indices (edge cases), fall back to the nearest grid cell
        if len(x_indices) == 0 or len(y_indices) == 0:
            # compute center point of the used range and find nearest grid cell
            center_lat = (used_lat_min + used_lat_max) / 2.0
            center_lon = (used_lon_min + used_lon_max) / 2.0
            dist2 = (self.lat_center - center_lat) ** 2 + (self.lon_center - center_lon) ** 2
            flat_idx = int(np.argmin(dist2))
            y, x = np.unravel_index(flat_idx, self.lat_center.shape)
            x_min = int(x)
            x_max = int(x) + 1
            y_min = int(y)
            y_max = int(y) + 1
        else:
            x_min = int(x_indices.min())
            x_max = int(x_indices.max()) + 1
            y_min = int(y_indices.min())
            y_max = int(y_indices.max()) + 1

        # Get actual lat/lon bounds of the selected region
        lat_sub = self.lat_center[y_min:y_max, x_min:x_max]
        lon_sub = self.lon_center[y_min:y_max, x_min:x_max]

        return {
            'requested_lat_range': requested_lat,
            'requested_lon_range': requested_lon,
            'used_lat_range': [float(used_lat_min), float(used_lat_max)],
            'used_lon_range': [float(used_lon_min), float(used_lon_max)],
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


@tool
def get_grid_indices_from_latlon(
    lat_range: List[float],
    lon_range: List[float],
    z_range: Optional[List[int]] = None
) -> Dict:
    """
    Convert lat/lon bounds to grid x/y indices and calculate data volume.
    
    Agent should call this AFTER reasoning about the geographic region.
    
    Agent reasoning steps BEFORE calling:
    1. Detect if query mentions a location
    2. Check if dataset has geographic coordinates (metadata)
    3. Recall approximate lat/lon bounds for that location from knowledge
    4. Verify bounds are within dataset coverage
    5. Then call THIS tool to get grid indices
    
    Args:
        lat_range: [min_lat, max_lat] in degrees (from agent's knowledge)
        lon_range: [min_lon, max_lon] in degrees (from agent's knowledge)
        z_range: optional [z_min, z_max] grid-level indices to restrict depth (if provided)
    
    Returns:
        {
            'status': 'success' | 'error',
            'x_range': [x_min, x_max],
            'y_range': [y_min, y_max],
            'z_range': [z_min, z_max],
            'estimated_points': int,
            'actual_lat_range': [min, max],
            'actual_lon_range': [min, max],
            'message': str
        }
    """
    print("80[Tool] get_grid_indices_from_latlon called")
    try:
        agent = get_agent()
        dataset = agent.dataset
        print(f"[Tool] Dataset loaded: {dataset.get('name', 'unknown')}")
        # Get geographic info
        geo_info = dataset.get('spatial_info', {}).get('geographic_info', {})
        geo_file = geo_info.get('geographic_info_file')
        dims = dataset.get('spatial_info', {}).get('dimensions', {})
        
        if not geo_file:
            return {
                'status': 'error',
                'message': 'No geographic file in dataset'
            }
        
        # Initialize converter
        converter = GeographicConverter(geo_file)
        
        # Convert lat/lon to x/y indices
        result = converter.latlon_to_indices(lat_range, lon_range)
        
        x_range = result['x_range']
        y_range = result['y_range']

        # Determine z_range to use (caller-specified or full depth)
        if z_range and isinstance(z_range, (list, tuple)) and len(z_range) == 2:
            try:
                z0 = int(z_range[0])
                z1 = int(z_range[1])
            except Exception:
                z0, z1 = 0, int(dims.get('z', 90))
            # Normalize and clamp
            if z0 < 0:
                z0 = 0
            max_z = int(dims.get('z', 90))
            if z1 > max_z:
                z1 = max_z
            z_range_used = [z0, z1]
        else:
            z_range_used = [0, int(dims.get('z', 90))]
        
        estimated_points = max(0, (x_range[1] - x_range[0])) * \
                          max(0, (y_range[1] - y_range[0])) * \
                          max(0, (z_range_used[1] - z_range_used[0]))
        
        return {
            'status': 'success',
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range_used,
            'estimated_points': estimated_points,
            'requested_lat_range': result.get('requested_lat_range'),
            'requested_lon_range': result.get('requested_lon_range'),
            'used_lat_range': result.get('used_lat_range'),
            'used_lon_range': result.get('used_lon_range'),
            'actual_lat_range': result.get('actual_lat_range'),
            'actual_lon_range': result.get('actual_lon_range'),
            'message': f"Converted lat {lat_range} lon {lon_range} z_range {z_range_used} to grid indices"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }



