echo "========================================================================"
echo "Space Invaders Rainbow DQN - Setup Script"
echo "========================================================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
    VENV_ACTIVATE="venv\\Scripts\\activate"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    IS_WINDOWS=false
    VENV_ACTIVATE="venv/bin/activate"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check Python version
echo "Step 1/5: Checking Python installation..."
if command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    echo "  Found: Python $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo "  Status: OK (Python 3.8+ required)"
    else
        echo "  ERROR: Python 3.8 or higher required!"
        exit 1
    fi
else
    echo "  ERROR: Python not found!"
    exit 1
fi

# Check pip
echo ""
echo "Step 2/5: Checking pip installation..."
if command -v $PIP_CMD &> /dev/null; then
    PIP_VERSION=$($PIP_CMD --version 2>&1 | awk '{print $2}')
    echo "  Found: pip $PIP_VERSION"
    echo "  Status: OK"
else
    echo "  ERROR: pip not found!"
    exit 1
fi

# Check/create virtual environment
echo ""
echo "Step 3/5: Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
    read -p "  Do you want to recreate it? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing old virtual environment..."
        rm -rf venv
        echo "  Creating new virtual environment..."
        $PYTHON_CMD -m venv venv
        echo "  Status: RECREATED"
    else
        echo "  Status: USING EXISTING"
    fi
else
    echo "  Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -eq 0 ]; then
        echo "  Status: CREATED"
    else
        echo "  ERROR: Failed to create virtual environment!"
        exit 1
    fi
fi

# Activate virtual environment
echo ""
echo "Step 4/5: Activating virtual environment..."
if [ "$IS_WINDOWS" = true ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

if [ $? -eq 0 ]; then
    echo "  Status: ACTIVATED"
else
    echo "  ERROR: Failed to activate virtual environment!"
    exit 1
fi

# Upgrade pip in venv
echo "  Upgrading pip..."
pip install --upgrade pip --quiet
NEW_PIP_VERSION=$(pip --version 2>&1 | awk '{print $2}')
echo "  Upgraded to: pip $NEW_PIP_VERSION"

# Install packages from requirements.txt
echo ""
echo "Step 5/5: Installing packages from requirements.txt..."
echo "  This may take several minutes (especially PyTorch ~2.5 GB)..."
echo ""

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "  Status: ALL PACKAGES INSTALLED SUCCESSFULLY"
    else
        echo ""
        echo "  ERROR: Some packages failed to install!"
        echo "  Please check the error messages above."
        exit 1
    fi
else
    echo "  ERROR: requirements.txt not found!"
    exit 1
fi

# Verify critical packages
echo ""
echo "Verifying installation..."
echo ""

python << 'PYEOF'
import sys

packages_to_test = [
    ("gymnasium", "Gymnasium (Atari environment)"),
    ("ale_py", "ALE (Atari Learning Environment)"),
    ("torch", "PyTorch (deep learning)"),
    ("numpy", "NumPy (numerical computing)"),
    ("pygame", "Pygame (rendering)"),
    ("matplotlib", "Matplotlib (plotting)"),
]

failed = []
for package, description in packages_to_test:
    try:
        __import__(package)
        print(f"  ✓ {description}")
    except ImportError:
        print(f"  ✗ {description} - FAILED")
        failed.append(package)

if failed:
    print(f"\nERROR: Failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n  Status: ALL CRITICAL PACKAGES VERIFIED")
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "  ERROR: Package verification failed!"
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
python << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
else:
    print("  ℹ CUDA not available (will use CPU)")
    print("  Note: Training will be slower on CPU")
PYEOF

# Final summary
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Installation Summary:"
echo "  ✓ Python $PYTHON_VERSION verified"
echo "  ✓ Virtual environment created/verified"
echo "  ✓ All packages installed from requirements.txt"
echo "  ✓ Critical packages verified"
echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo ""
echo "1. Activate virtual environment:"
if [ "$IS_WINDOWS" = true ]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Test installation:"
echo "   python -c \"from space_invaders import *; print('✓ Ready!')\""
echo ""
echo "3. Quick test (10 episodes):"
echo "   python space_invaders.py train --episodes 10"
echo ""
echo "4. Full training (10k episodes):"
echo "   python space_invaders.py train --episodes 10000"
echo ""
echo "========================================================================"
echo ""