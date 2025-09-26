#!/usr/bin/env bash
set -e  # stop if anything fails

# Create the virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment…"
    python3 -m venv .venv
fi

# Activate the environment
echo "Activating virtual environment…"
source .venv/bin/activate

# Upgrade pip and install the dependency
echo "Installing DeepSDFStruct…"
pip install --upgrade pip
pip install --verbose git+https://github.com/mkofler96/DeepSDFStruct.git

echo "All done."
echo "To use this environment in a new terminal later, run:"
echo "    source .venv/bin/activate"


