#!/bin/bash

# Setup script for the visualization tool
# This script sets up the environment to use the visualization tool

PROJECT_ROOT="/home/eliza89/PhD/codes/vis_user_tool"
BUILD_DIR="$PROJECT_ROOT/build"
VENV_DIR="$PROJECT_ROOT/venv_new"

echo "=== Visualization Tool Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"
echo "Virtual environment: $VENV_DIR"
echo

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please create it first with: python3 -m venv venv_new"
    exit 1
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    echo "Please build the project first with: ./superbuild.sh"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Set Python path
export PYTHONPATH="$BUILD_DIR/renderingApps/py:$PYTHONPATH"

echo "Environment setup complete!"
echo
echo "Available modules:"
echo "  - VTK 9.3.1"
echo "  - PyVista (alternative to tvtk)"
echo "  - vistool_py, vistool_py_osp, vistool_py_vtk"
echo "  - NumPy, matplotlib"
echo
echo "Usage examples:"
echo "  python3 test_visualization_tool.py    # Test all modules"
echo "  python3 python/render.py <json_file>  # Render visualization"
echo
echo "Note: To use in other scripts, add this to your Python code:"
echo "  import sys"
echo "  sys.path.insert(0, '$BUILD_DIR/renderingApps/py')"
echo
echo "To deactivate the environment later, run: deactivate"
