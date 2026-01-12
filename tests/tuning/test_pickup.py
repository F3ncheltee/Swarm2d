import numpy as np
import time
import sys
import os
import random
import pandas as pd
import pygame
import json
import traceback
from collections import defaultdict
import math
import argparse

# --- Python Path (ensure Swarm2DEnv is found) ---
# Calculate the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from env.env import Swarm2DEnv
    from constants import AGENT_RADIUS
    import pybullet as p
except ImportError as e:
    print(f"CRITICAL Error importing Swarm2DEnv or PyBullet: {e}")
    traceback.print_exc()
    sys.exit(1)

# ==============================================================================
# ===                 NEW TUNING WORKFLOW                                    ===
# ==============================================================================
#
# The physics tuning pipeline has been refactored into separate, focused scripts
# for clarity and efficiency. Please run them in order.
#
# Each script is self-contained and will guide you through its specific tuning phase.
#
# --- WORKFLOW ---
# 1. Tune base agent movement:
#    python Swarm2d/test_code/tune_phase0_agent_movement.py
#
# 2. Tune agent agility (braking & turning):
#    python Swarm2d/test_code/tune_phase1_agent_agility.py
#
# 3. Tune single-agent carrying:
#    python Swarm2d/test_code/tune_phase2.py
#
# 4. Tune agent grappling strength:
#    python Swarm2d/test_code/tune_phase3_agent_grapple.py
#
# 5. Tune cooperative carrying mass scaling:
#    python Swarm2d/test_code/tune_phase4_coop_carry.py
#
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" " * 20 + "Physics Tuning Pipeline Refactored")
    print("="*80)
    print("\nThe physics tuning pipeline has been split into separate scripts for each phase.")
    print("Please run the scripts individually in the recommended order.")
    print("\n--- Recommended Workflow ---")
    print("1. Tune base agent movement:")
    print("   python Swarm2d/test_code/tune_phase0_agent_movement.py")
    print("\n2. Tune agent agility (braking & turning):")
    print("   python Swarm2d/test_code/tune_phase1_agent_agility.py")
    print("\n3. Tune single-agent carrying:")
    print("   python Swarm2d/test_code/tune_phase2.py")
    print("\n4. Tune agent grappling strength:")
    print("   python Swarm2d/test_code/tune_phase3_agent_grapple.py")
    print("\n5. Tune cooperative carrying mass scaling:")
    print("   python Swarm2d/test_code/tune_phase4_coop_carry.py")
    print("\nThis script (`test_pickup.py`) no longer contains the tuning logic.")
    print("="*80)