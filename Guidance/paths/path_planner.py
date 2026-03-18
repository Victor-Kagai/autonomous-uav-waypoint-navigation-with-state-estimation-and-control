import sys
import os
import numpy as np
import yaml
from pathlib import Path
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Guidance.paths import path_generator

def __init__(self,model):

        script_dir = Path(__file__).parent
        config_path = script_dir / 'Vehicleparams.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
