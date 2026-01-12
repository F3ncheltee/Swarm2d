print("--- Script training/main.py starting ---")
#!/usr/bin/env python3
import warnings
# Suppress the specific UserWarning from pygame about pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module='pygame')
"""
CTDE Training Script for a 6-Team Swarm Environment (3 Limited, 3 Global View)

- Teams 0, 1, 2 use Limited Critics (Combined Team FoV).
- Teams 3, 4, 5 use Global Critics (Full Environment View).
- Teams 3, 4, 5 use copies of policies from 0, 1, 2 respectively.
- All critics use the UnifiedCriticCore architecture.
- Implements stateful OCC map generation for Limited Critics.
- Uses radius_graph for efficient graph construction.
- Assumes a state history mechanism for critic updates.
"""
import cProfile
import pstats
import io
import torch.profiler
import matplotlib.pyplot as plt # Optional for visualization
import torch
import torch.nn.functional as F
import numpy as np
import traceback
from typing import Tuple, Dict, Optional, Union, List
import sys, os, time, random, copy
import numpy as np
import gc # Garbage collector
from collections import deque
# --- Imports ---
import scipy.spatial # For KDTree
import cProfile # For profiling
import pstats # For reading profile stats
import io # For reading profile stats
import traceback
import pickle # Needed for potential buffer saving/loading

# To allow running as a script, add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
import torch_geometric.data
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter # For vectorized map generation
import torch_geometric.nn as pyg_nn 
from torch_geometric.data import Data, Batch
radius_graph = pyg_nn.radius_graph

import glob 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import traceback # For detailed error printing if needed

# --- Local Imports ---
try:
    from Swarm2d.env.env import Swarm2DEnv
    
    from Swarm2d.policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy, DecentralizedTrailMemory, TrailManager
    from Swarm2d.policies.actors.MAAC.maac_attentionGNN import MAACPolicy
    from Swarm2d.policies.actors.NCA.nca_networkGNN import NCA_PINSANPolicy
    
    from Swarm2d.policies.critics.advanced_criticGNN import UnifiedCriticCore, BaseCriticWrapper
    
    from Swarm2d.env.occlusion import generate_gpu_occlusion_field, check_los_batched_gpu_sampling
    
    from Swarm2d.constants import (
        WIDTH, HEIGHT, FPS, OBS_RADIUS, AGENT_RADIUS, MAX_STEPS, HIVE_MAX_HEALTH, 
        AGENT_MAX_HEALTH, AGENT_MAX_ENERGY, RESOURCE_MAX_SIZE, BEE_SPEED, 
        AGENT_BASE_STRENGTH, HIVE_RADIUS_ASSUMED, NODE_TYPE, NUM_NODE_TYPES, 
        NODE_FEATURE_MAP, NODE_FEATURE_DIM, MEM_NODE_FEAT_IDX, MEM_NODE_FEATURE_DIM, 
        RAW_CH_IDX_TO_NAME, GLOBAL_CUE_DIM, CRITIC_EDGE_FEATURE_DIM, OCC_CH, 
        OCC_CH_COUNT, RAW_CH, RAW_CH_COUNT, REWARD_COMPONENT_KEYS, CLUSTER_CELL_SIZE
    )
    from Swarm2d.training.log_utils import log_input_details
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure the script is run from the correct directory and PYTHONPATH is set.")
    sys.exit(1)

torch.autograd.set_detect_anomaly(False)

from Swarm2d.training.PlateauManager import TeamPlateauManager
from run_training import run_training_trial
from Swarm2d.training.utils import set_seeds, assign_conceptual_teams_for_episode, get_innate_properties_for_episode, soft_update_targets
from config import (
    critic_config,
    maac_hyperparameters,
    nca_hyperparameters,
    shared_hyperparameters,
    default_critic_config,
    best_maac_params,
    best_nca_params,
    best_sharedactor_params
)

# --- Configuration ---
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force CPU for debugging if needed
print(f"Using device: {device}")
# Ensure reproducibility for some PyTorch operations
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
DEBUG_INPUT_CHECKS = False # Set to True to enable checks, False to disable
# --- DEBUGGING & LOGGING ---
DEBUG_LOG_INPUTS = True     # Master switch to enable/disable detailed input logging
DEBUG_EPISODE = 0           # The specific episode to log inputs for
DEBUG_STEP = 5              # The specific step within that episode to log actor inputs
DEBUG_UPDATE_COUNT = 0      # Counter to log critic inputs only a few times
MAX_DEBUG_UPDATES = 2       # How many times to log critic inputs per run
LOG_FIRST_MAP = True        # Set to True to log the map once
LOG_MAP_EPISODE = 0         # Episode to log the map
LOG_MAP_STEP = 10           # Step within the episode to log
LOG_MAP_AGENT_IDX = 69      # Global environment index of the agent whose map we want to see
VISUALIZE_MAP = False       # Set to True to pop up matplotlib plots (pauses execution!)

# --- Environment Instantiation ---
try:
    env = Swarm2DEnv(render_mode=False, debug=False, num_agents_per_team=20, num_teams=6, use_batched_memory=True)
    print(f"Swarm2DEnv initialized with {env.num_agents} agents ({env.num_agents_per_team} per team for({env.num_teams}) teams).")
except Exception as e:
    print(f"Error initializing env: {e}. Exiting.")
    sys.exit(1)

# --- Derive/Define Constants AFTER Env Instantiation ---
print("Deriving constants from environment and configuration...")
try:
    # Core Env Params
    NUM_TEAMS = getattr(env, 'num_teams', 6)
    AGENTS_PER_TEAM_ENV = getattr(env, 'num_agents_per_team', 50) # Use env value
    TOTAL_AGENTS_ENV = getattr(env, 'num_agents', NUM_TEAMS * AGENTS_PER_TEAM_ENV)
    WIDTH = getattr(env, 'width', 1000)
    HEIGHT = getattr(env, 'height', 1000)
    MAX_STEPS = getattr(env, 'max_steps', 100) # Use env value
    AGENT_BASE_RADIUS = env.metadata.get('agent_radius', AGENT_RADIUS)
    AGENT_BASE_SPEED = env.metadata.get('bee_speed', BEE_SPEED)
    HIVE_BASE_RADIUS = env.metadata.get('hive_radius_assumed', 30.0)
    OBSTACLE_MAX_RADIUS = env.metadata.get('obstacle_radius_max_config', 50.0)
    MAP_DIAGONAL = np.sqrt(WIDTH**2 + HEIGHT**2)
    ROLE_EMBED_DIM = 16 # Dimension for the continuous role vector from policies

    # Conceptual Teams (No change)
    NUM_CONCEPTUAL_TEAMS = 6
    NUM_CONCEPTUAL_AGENTS_PER_TEAM = AGENTS_PER_TEAM_ENV
    TOTAL_CONCEPTUAL_AGENTS = NUM_CONCEPTUAL_TEAMS * NUM_CONCEPTUAL_AGENTS_PER_TEAM

    max_grad_norm = 1.0

    # Action Space (Keep as before)
    MOVEMENT_DIM = 2; PICKUP_DIM = 3; JOINT_ACTION_DIM = MOVEMENT_DIM + 1

    # Grid Size (Consistent)
    CRITIC_GRID_SIZE = 32 # For map representations fed to critic/actor CNNsexit
    ACTOR_MAP_GRID_SIZE = 32   # Use your project's ACTOR_MAP_GRID_SIZE

    CRITIC_SEQUENCE_LENGTH = 3  # How many consecutive steps the critic sees

    # Graph Construction Params
    OBS_RADIUS = env.metadata.get('obs_radius') # Base observation radius

    FOVEA_RADIUS = OBS_RADIUS * 1.0 # High-detail memory zone is twice the live perception radius
    GRAPH_PROX_RADIUS = OBS_RADIUS    
    # GRAPH_PROX_RADIUS = OBS_RADIUS # Use obs radius for graph connectivity
    MAX_GRAPH_NEIGHBORS = 32

except Exception as e:
    print(f"Error deriving constants: {e}"); traceback.print_exc(); sys.exit(1)

# --- EXPANDED Global Cue Dimension ---
print(f"  Global Cue Dim: {GLOBAL_CUE_DIM}")
# --- MAAC Roles (Keep as before) ---
MAAC_ROLES_GLOBAL = ["scout", "collector", "defender", "attacker"]
ROLE_NAME_TO_IDX_MAP = {name: idx for idx, name in enumerate(MAAC_ROLES_GLOBAL)}
env_self_obs_dim = env.self_obs_dim # Store the dimension after env init
print(f"  Env Self Obs Dim: {env_self_obs_dim}")
ACTOR_SELF_OBS_DIM = env_self_obs_dim 

# --- Hyperparameter Loading ---
NUM_REWARD_COMPONENTS = len(REWARD_COMPONENT_KEYS)

if len(REWARD_COMPONENT_KEYS) != NUM_REWARD_COMPONENTS:
    print(f"FATAL WARNING: Length of REWARD_COMPONENT_KEYS ({len(REWARD_COMPONENT_KEYS)}) does not match NUM_REWARD_COMPONENTS ({NUM_REWARD_COMPONENTS}). Check definitions.")
    # sys.exit(1) # Optional: Make this fatal if strict matching is required
print(f"Defined REWARD_COMPONENT_KEYS for logging ({len(REWARD_COMPONENT_KEYS)} keys).")

print("Hyperparameters determined (Critic Defaults).")


MAAC_ROLES_GLOBAL = ["scout", "collector", "defender", "attacker"]
ROLE_NAME_TO_IDX_MAP = {name: idx for idx, name in enumerate(MAAC_ROLES_GLOBAL)}
print(f"Defined Global ROLE_NAME_TO_IDX_MAP: {ROLE_NAME_TO_IDX_MAP}")


# ===================================================================
#                       MAIN EXECUTION BLOCK
# ===================================================================
if __name__ == "__main__":
    # --- Environment Instantiation ---
    # Create the ONE and ONLY environment instance here.
    # print("Initializing environment for main training run...")
    # env = Swarm2DEnv(render_mode=True, debug=False, num_agents_per_team=20, num_teams=6)
    # print(f"Swarm2DEnv initialized with {env.num_agents} agents...")
    # Create the ONE and ONLY environment instance here.

    print("Initializing environment for main training run...")
        
    trial_params = {
        'run_name': "MultiTeamFinalTest",
        'seed': 42,
        'buffer_capacity': 100000,
        'total_agents': TOTAL_AGENTS_ENV
    }

    # Call the main training function and PASS THE ENV INSTANCE to it.
    try:
        print("Calling run_training_trial...")
        result = run_training_trial(env, trial_params, trial_hyperparams=None, optuna_trial=None) # <--- PASS 'env' HERE
        print(f"Training trial completed. Result: {result}")
    except Exception as e:
        print(f"\n!!!!!! A FATAL ERROR OCCURRED IN THE TRAINING TRIAL !!!!!!")
        print(f"Error: {e}")
        traceback.print_exc()
        print("\n" + "="*60)
        # The 'env' object is available here for cleanup.
        try:
            if 'env' in locals() or 'env' in globals():
                env.close()
                print("Environment closed after error.")
        except:
            pass