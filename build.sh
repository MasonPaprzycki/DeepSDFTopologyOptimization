#bash script that should set up a venv and install all dependencies on windows

#!/usr/bin/env bash
set -e  # stop on first error

echo "🔧 Setting up DeepSDFTopologyOptimization environment..."

# --- Create venv if missing ---
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
else
    echo "✅ Virtual environment already exists."
fi

# --- Activate venv ---
echo "⚡ Activating virtual environment..."
# shellcheck disable=SC1091
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # Windows (Git Bash)
    source .venv/Scripts/activate
else
    echo "❌ Could not find activation script."
    exit 1
fi

# --- Upgrade pip etc ---
echo "⬆️ Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# --- Install dependencies ---
echo "📦 Installing dependencies..."
pip install torch numpy trimesh scikit-image mesh-to-sdf

# --- Verify imports ---
echo "🔍 Verifying imports..."
python - <<'PY'
import torch, numpy, trimesh, mesh_to_sdf
from skimage import measure
import DeepSDFStruct
print("✅ All imports OK")
PY

echo "✅ Installation complete."
echo "To activate later, run: source .venv/bin/activate  (or .venv/Scripts/activate on Windows)"
echo "🎉 Setup finished successfully."
