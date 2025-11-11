import sys, os
import numpy as np
from threading import Thread
import pathlib

# path to the rendering app libs
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'build', 'renderingApps', 'py'))
import renderInterface3

a = renderInterface3.AnimationHandler()

#
# call offline render to produce video
#
f_path = sys.argv[1]

# Create output directory for rendered frames
input_path = pathlib.Path(f_path)
parent_dir = input_path.parent.parent  # Go up from GAD_text
output_dir = parent_dir / "rendered_frames"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set environment variable for the renderer to use
os.environ["RENDER_OUTPUT_DIR"] = str(output_dir)

print(f"Input JSON: {f_path}")
print(f"Output directory: {output_dir}")

# Start the render task in a background thread. Important: pass the callable
# as `target` and provide args separately — calling the function here would
# execute it synchronously in the main thread instead of in the new thread.
# Use daemon=True so the process can exit if the main thread finishes.
t = Thread(target=a.renderTaskOfflineVTK, args=(f_path,), daemon=True)
t.start()
