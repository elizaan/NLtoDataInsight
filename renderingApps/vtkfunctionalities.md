What works already

The helper in vtkFuns3.cpp is data‑agnostic in many places: transfer functions (colors/opacities/range) come from JSON, volume mapper/property is generic, isosurface uses thresholds from JSON, streamlines use whatever vectors are present in the VTK dataset, and texture fallback is generic. Nothing in the code enforces “ocean” semantics (like salinity/temperature names) at runtime.
Key implicit assumptions you must satisfy for non-ocean data

File / reader type

Code currently uses vtkStructuredGridReader (reader->SetFileName(...) / reader->Update()) and then treats the output as a structured grid. Your input file must be a VTK StructuredGrid file (the legacy VTK structured grid format that vtkStructuredGridReader understands). Unstructured meshes, polydata-only files, or other file formats will not work without changing the reader.
Scalars selection

The code takes scalars with reader->GetOutput()->GetPointData()->GetArray(0) (and GetScalars() used elsewhere). That assumes the scalar you want to render is the first array in the point data. If your data has a different array order or different array name, the code will pick the wrong array or fail.
Vector field for streamlines

vtkStreamTracer is given the reader output directly. Streamlines require an active vector array (point-data vectors). The code does not explicitly choose a vector array name — it relies on the dataset having an active vector attribute already. If the dataset does not have vectors or they are stored under a different name or as separate components, streamlines will not be created correctly.
Spacing / origin assumptions

imageData->SetSpacing(1.0,1.0,1.0) and SetOrigin(0,0,0) are hardcoded. If your dataset has non‑unit spacing (e.g., geographic coordinates, anisotropic voxels) the geometry/texture mapping and camera framing may be wrong. Also the texture coordinate mapping (u ← z, v ← y) assumes axis roles consistent with the original ocean data convention.
Data types & component counts

The code assumes scalars are numeric floats/doubles and that vector arrays have 3 components. If scalars are bytes or integers without intended scaling, TF ranges might be off. Opacity/color TFs expect scalar ranges matching input units.
Texture files and paths

Texture is applied only if scene_info["texture"]["textureFile"] exists and is a readable PNG. If using other image formats you’d need to add readers or convert.
Use of vtkThreshold on vtkImageData

The isosurface path uses vtkThreshold followed by vtkGeometryFilter. This works with datasets that vtkThreshold supports (point or cell data). Some dataset types or scalar associations may require different thresholding techniques (e.g., vtkMarchingCubes on vtkImageData / vtkStructuredPoints).
Edge cases / failure modes you’ll encounter with arbitrary data

Input is not structured-grid legacy VTK → reader fails or output empty.
Scalar array missing or has unexpected component count → empty volume / wrong TF mapping.
Vector array missing → streamlines produce no geometry.
Extremely large datasets → memory / performance problems (mapper / rendering time).
Non-unit spacing → textures and iso UV mapping wrong; camera framing odd.
Named scalar arrays required to reproduce the same visual semantics (e.g., you expect “temperature” but array is “T”) — code will not pick it automatically.
Minimal changes to make it robust for generic 3D data (recommended)

Make reader generic / detect file type

Replace vtkStructuredGridReader with a format-detecting reader or expose the reader selection:
Option A: use vtkDataSetReader or vtkGenericDataObjectReader for generic legacy VTK files.
Option B: allow JSON to specify file type or choose a reader by extension (e.g., .vti -> vtkXMLImageDataReader, .vtk -> vtkDataSetReader).
Allow selecting arrays by name (JSON-configurable)

Let the JSON optionally specify the scalar and vector arrays:
fields like "scalarArray": "theta" and "vectorArray": "velocity".
In code, prefer GetArray("scalarArrayName") and fall back to array(0) only if not provided.
Respect dataset spacing / origin

After reader->Update(), read spacing and origin from reader->GetOutput()->GetSpacing() and GetOrigin() and copy them to imageData instead of hardcoded unit spacing.
Validate array component counts

Before using arrays, check GetNumberOfComponents(); error/log if unexpected (e.g., vectors must be 3 components).
Explicitly set active arrays for streamlines

If JSON supplies a vectorArray name, call:
reader->GetOutput()->GetPointData()->SetActiveVectors("velocityName");
or use streamers->SetInputArrayToProcess(...) to select the vector field.
Support alternate isosurfacing for image data

For vtkImageData prefer vtkMarchingCubes (or vtkFlyingEdges where available) for isosurfaces instead of vtkThreshold+vtkGeometryFilter which is more generic for unstructured data.

```markdown
What works now

The helper in `vtkFuns3.cpp` has been refactored to be JSON-driven and to preserve the visual behavior of the previous v2 helper where possible. Key capabilities implemented in the current branch (robust-pipeline) are:

- JSON-driven transfer functions and volume properties. Colors/opacities/range are read from `scene_info.transferFunc` and applied to VTK transfer functions.
- Volume mapper and volumeProperty settings are read from JSON `mapperProperties` and `volumeProperties` (sampleDistance, imageSampleDistance, blendMode, shading, etc.).
- Isosurface (land mask) creation driven by JSON thresholds. Texture coordinates are generated using the v2 mapping (u ← z, v ← y) so the texture alignment matches v2 visuals. If a texture file exists it is applied; otherwise a fallback color or LUT is used.
- Streamlines are created from the dataset when possible. The implementation prefers an explicit field name from JSON (rep["field"]) and otherwise falls back to the dataset's active vectors. Seed plane, integration properties and streamline styling are read from JSON.
- Camera interpolation is centralized in the Animator; the renderer consumes the interpolated camera only (AnimatorKF::get_current_cam stores focalPoint when JSON provides it so renderer receives an absolute focalPoint).
- Structured-grid resampling for volume rendering (Option A): when `global_metadata.grid_type == "structured"` the code resamples the structured dataset into a `vtkImageData` using reader-native ordering for the resample target. The resampling target spacing/origin is read from `global_metadata.spacing` and `global_metadata.origin`.
- Axis-permutation detection: a common mismatch (JSON dims = x,y,z vs reader dims = z,y,x) is detected and handled by permuting spacing/origin before resampling to avoid destroyed visuals.

## Implementation details (notes for maintainers)

- Reader and data flow
	- Code currently uses the legacy VTK legacy readers but treats the reader output generically as a `vtkDataSet` (typically `vtkStructuredGrid`). `loadKF2` inspects the reader output (`ds`) and either resamples it to `vtkImageData` for volume rendering or uses `ds` directly for streamlines/isosurface generation.

- Scalars and arrays
	- By default the volume uses the primary scalar from the reader output (array index 0) unless the JSON explicitly names a `scalarArray`. If your data places the scalar of interest elsewhere, prefer adding `scalarArray` in the JSON.
	- Streamlines require point-data vectors. The code will use `rep["field"]` when present, or the dataset's active vectors as a fallback. If a named vector is not found the code logs a warning and falls back to active vectors.

- Spacing / origin and axis order
	- The loader now reads spacing/origin from `Animator::get_global_metadata()` (which reads `data_list.json` `global_metadata`). Those values are used to build the resampling target image.
	- The code detects and handles the common axis-permutation case (reader dims equal JSON dims with X/Z swapped). When detected the spacing and origin are permuted to match the reader ordering before constructing the resampling target, preventing transposed/destroyed volumes.

- Isosurface UV mapping
	- The isosurface uses the v2-style UV mapping so existing land texture artwork aligns: u is derived from the data z-coordinate and v from the y-coordinate (normalized by bounds). This preserves older visual layouts.

## Known caveats and failure modes

- Input file type: the code expects legacy VTK structured-grid style inputs for the current reader path. XML VTK filetypes (.vti, .vtr, etc.) will require switching to the corresponding XML reader or adding a reader-detection step.
- Scalar selection: if the scalar you expect is not the first array, TFs and isosurfaces will be off. Use `scalarArray` in JSON to select by name.
- Vector selection: streamlines require the vector array to be present as point-data. If your vectors are cell-data or named differently, the code will either produce no streamlines or fall back to an unintended field. See Todos for improvements.
- Non-unit spacing: if `global_metadata.spacing` is incorrect or missing, the geometry/texture mapping and camera framing may look stretched.

## Suggested JSON fields (to make your data robustly consumable)

- `scene_info.scalarArray` (string): choose the scalar array name for volume & thresholds.
- `scene_info.field` or `scene_info.vectorArray` (string): choose the vector array name for streamlines.
- `data_list.json` → `global_metadata`:
	- `grid_type`: "structured" or other
	- `spacing`: [sx, sy, sz]
	- `origin`: [ox, oy, oz]
	- `dims`: [nx, ny, nz] (JSON-side dims)

Example snippets:

Volume (explicit scalar):

```json
{
	"representation": "volume",
	"scene_info": {
		"scalarArray": "salinity",
		"transferFunc": { "range": [0, 36], "colors": [...], "opacities": [...] }
	}
}
```

Streamline (explicit vector):

```json
{
	"representation": "streamline",
	"scene_info": {
		"field": "velocity",
		"seedPlane": { "enabled": true, "positionFraction": [0.5,0.5,0.5], "resX": 40, "resY": 20 }
	}
}
```

Isosurface (land mask):

```json
{
	"representation": "isosurface",
	"scene_info": {
		"thresholdRange": [0.99, 1.0],
		"texture": { "enabled": true, "textureFile": "agulhaas_mask_land.png" },
		"surfaceProperties": { "color": [0.518,0.408,0.216] }
	}
}
```

## Quick troubleshooting checklist

- If streamlines are missing: check available arrays in the VTK file and ensure a 3-component point-data vector exists, or add `field` in JSON.
- If volume looks transposed: check `data_list.json` `global_metadata.dims` vs the reader-reported dimensions. The common X↔Z swap is handled; other permutations may need extra metadata.
- If textures are missing on the isosurface: ensure `scene_info.texture.textureFile` path is correct and readable.

## Low-risk improvements to implement next (todos)

1. Vector field selection & fallback improvements (recommended next task)
	 - Try `rep["field"]` first, then dataset active vectors, then common names ["velocity","vel","u","v","w"]. If vectors are cell-data, auto-convert to point-data.

2. Array debug prints
	 - When loading a dataset, print (or log at debug level) all point- and cell-data array names and their component counts. This immediately shows which names to use in JSON.

3. Allow scalarArray selection
	 - Prefer `scene_info.scalarArray` by name when present; fall back to array(0) with a clear warning.

4. Reader detection / filetype handling
	 - Add a small mapping by extension or a JSON filetype hint to choose the correct reader (e.g., .vti -> vtkXMLImageDataReader). This will make non-legacy VTK inputs work without editing code.

5. Resampling performance
	 - Consider using `vtkResampleToImage` instead of `vtkProbeFilter` for structured datasets; it's often faster and more robust for regular grids.

6. Visual-regression test
	 - Add a test that renders one keyframe with both helpers (v2 and v3) and computes a diff/SSIM to detect regressions in appearance.

## How to build & run (reminder)

Build the Python VTK extension (same as your workflow):

```bash
cd /home/eliza89/PhD/codes/vis_user_tool/build
make -j vistool_py_vtk3
```

Run the renderer on a JSON file (select the wrapper you use):

```bash
python python/renderVTK2.py python/test/GAD_text_v3/case2_script.json
```

## Current todo checklist (short)

- [x] Expose global metadata
	- `Animator::get_global_metadata()` added; used by `vtkFuns3` to read `global_metadata` spacing/origin.
- [x] Use global spacing/origin in `vtkFuns3`
	- `loadKF2` reads spacing/origin and applies it to the resampling target; axis-permutation handling for the common x↔z swap implemented.
- [ ] Vector field selection & fallback improvements
	- Try `rep["field"]`, active vectors, common names, convert cell→point vectors when necessary.
- [ ] Add array debug prints
	- Print available point & cell data arrays and component counts at dataset load.
- [ ] Add automated visual regression test
	- Render a single keyframe with v2 and v3 and compare output.
- [ ] Consider resampling performance improvements
	- Replace `vtkProbeFilter` with `vtkResampleToImage` or tune probe parameters.

If you want I can start implementing item 1 (vector selection + debug prints) next — I'll update the code, rebuild, and run a single-keyframe smoke test and paste the array diagnostics here.
```