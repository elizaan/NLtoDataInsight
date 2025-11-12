"""
Dataset schema definitions (Pydantic models) for dataset profiling output.
This mirrors the project's `parameter_schema.py` style and provides a
strongly-typed contract for the profiler output JSON (dataset{index}.json).
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VariableComponent(BaseModel):
    id: str = Field(..., description="Component id (e.g., eastwest_velocity)")
    name: Optional[str] = Field(None, description="Human readable name")
    description: Optional[str] = Field(None, description="Short description")
    url: Optional[str] = Field(None, description="Data URL or index file for the component")


class Variable(BaseModel):
    id: str = Field(..., description="Variable id (e.g., temperature)")
    name: Optional[str] = Field(None, description="Human readable name")
    description: Optional[str] = Field(None, description="Short description")
    url: Optional[str] = Field(None, description="Primary data URL (if scalar) or index file")
    field_type: Optional[str] = Field(None, description="'scalar' or 'vector'")
    unit: Optional[str] = Field(None, description="Units (e.g., degrees Celsius)")
    components: Optional[Dict[str, VariableComponent]] = Field(None, description="Subcomponents for vector variables")


class GeographicInfo(BaseModel):
    has_geographic_info: str = Field(..., description="'yes' or 'no'")
    geographic_info_file: Optional[str] = Field(None, description="Filename containing lat/lon info (e.g., llc2160_latlon.nc)")

class SpatialInfo(BaseModel):
    dimensions: Dict[str, int] = Field(..., description="{'x': ..., 'y': ..., 'z': ...}")
    geographic_info: List[GeographicInfo] = Field(None)


class TimeRange(BaseModel):
    start: Optional[str] = Field(None, description="ISO date string or similar")
    end: Optional[str] = Field(None, description="ISO date string or similar")


class TemporalInfo(BaseModel):
    has_temporal_info: str = Field(..., description="'yes' or 'no'")
    time_range: Optional[TimeRange] = Field(None)
    total_time_steps: Optional[str] = Field(None)
    time_units: Optional[str] = Field(None)


class DatasetSchema(BaseModel):
    # Required fields per user request
    name: str = Field(..., description="Dataset human-readable name")
    id: str = Field(..., description="Dataset identifier")
    index: str = Field(..., description="Index used for filename generation")
    size: str = Field(..., description="Human-readable size description")
    variables: List[Variable] = Field(...)
    spatial_info: SpatialInfo = Field(...)

    # Optional / auxiliary fields
    description: Optional[str] = Field(None)
    type: Optional[str] = Field(None, description="Dataset type, e.g., 'oceanographic data'")
    Resolution: Optional[str] = Field(None, description="Spatial resolution description")
    temporal_info: Optional[TemporalInfo] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "name": "DYAMOND LLC2160 OCEAN SIMULATION DATA",
                "id": "dyamond_llc2160",
                "index": "1",
                "description": "High-resolution ocean simulation data from the DYAMOND project (LLC2160)",
                "type": "oceanographic data",
                "size": "petabyte-scale",
                "Resolution": "1/24° horizontal grid (~4 km), 90 vertical levels",
                "variables": [{
                        "id": "temperature",
                        "name": "Sea-surface Temperature",
                        "description": "Temperature field from DYAMOND LLC2160",
                        "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_theta/llc2160_theta.idx",
                        "field_type": "scalar",
                        "unit": "degrees Celsius"
                    },
                    {
                        "id": "salinity",
                        "name": "Sea Water Salinity",
                        "description": "Salinity field from DYAMOND LLC2160",
                        "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_salt/salt_llc2160_x_y_depth.idx",
                        "field_type": "scalar",
                        "unit": "g kg-1"
                    },
                    {
                        "id": "velocity",
                        "name": "Velocity",
                        "description": "Velocity field from DYAMOND LLC2160",
                        "field_type": "vector",
                        "components": {
                            "eastwest_velocity": {
                                "id": "eastwest_velocity",
                                "name": "Sea-surface east-west velocity",
                                "description": "Eastwest velocity (u) field from DYAMOND LLC2160",
                                "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_arco/visus.idx"
                            },
                            "northsouth_velocity": {
                                "id": "northsouth_velocity",
                                "name": "Sea-surface north-south velocity",
                                "description": "Northsouth velocity (v) field from DYAMOND LLC2160",
                                "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_v/v_llc2160_x_y_depth.idx"
                            },
                            "vertical_velocity": {
                                "id": "vertical_velocity",
                                "name": "Sea-surface vertical velocity",
                                "description": "Vertical velocity (w) field from DYAMOND LLC2160",
                                "url": "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/mit_output/llc2160_w/llc2160_w.idx"
                            }
                        }
                    }
                ],
                "spatial_info": {
                    "x_min": 0,
                    "x_max": 8640,
                    "y_min": 0,
                    "y_max": 6480,
                    "z_min": 0,
                    "z_max": 90,
                    "dimensions": {
                        "x": 8640,
                        "y": 6480,
                        "z": 90
                    },
                    "geographic_info": {
                        "has_geographic_info": "yes",
                        "geographic_info_file": "llc2160_latlon.nc",
                        "bounds": {
                            "latitude": {
                                "min": -89.995,
                                "max": 72.035
                            },
                            "longitude": {
                                "min": -180.0,
                                "max": 180.0
                            }
                        },
                        "data_variables": {
                            "latitude": [6480, 8640],
                            "longitude": [6480, 8640]
                        },
                        "attributes": {
                            "title": "LLC2160 Mosaic Grid",
                            "description": "Latitude and Longitude mosaic from LLC2160 faces 0, 1, 3, 4",
                            "Conventions": "CF-1.8",
                            "source": "Generated from MITgcm LLC2160 binary grid data"
                        },
                        "conversion_to_degrees": {
                            "latitude": "lat = lat_center[y_index, x_index] (degrees). Arrays are shaped (y, x); indices are 0-based. For ranges use lat_center[y_min:y_max, x_min:x_max] and then lat_min = lat_sub.min(), lat_max = lat_sub.max()",
                            "longitude": "lon = lon_center[y_index, x_index] (degrees). Arrays are shaped (y, x); indices are 0-based. For ranges use lon_center[y_min:y_max, x_min:x_max] and then lon_min = lon_sub.min(), lon_max = lon_sub.max()"
                        }
                    }
                },
                "temporal_info": {
                    "has_temporal_info": "yes",
                    "time_range": {
                        "start": "2020-01-20",
                        "end": "2021-03-26"
                    },
                    "min_time_step": "0",
                    "max_time_step": "10365",
                    "total_time_steps": "10366",
                    "time_units": "hours"
                }

            }
        }
