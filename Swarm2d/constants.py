import sys
import numpy as np
import torch.nn as nn

class ConstantsMain(nn.Module):
    ###########################################
    # Global Constants & Parameters (Physical/General)
    ###########################################
    # Arena dimensions
    WIDTH, HEIGHT = 1000, 1000
    D_MAX = np.sqrt(WIDTH**2 + HEIGHT**2)

    # Default Agent parameters
    AGENT_RADIUS = 3
    AGENT_BASE_STRENGTH = 1.0
    # For physics tuning, this is the desired solo agent max speed.
    # The tuning script will calibrate force/damping to achieve this.
    BEE_SPEED = 250.0 # Default agent speed
    AGENT_BASE_SPEED = BEE_SPEED  # Alias for compatibility
    EXTRA_RANDOMNESS_STD = 0.025
    ROLE_EMBED_DIM = 16  # Dimension for the continuous role vector from policies

    # Team Colors (Updated for visualization distinctness)
    TEAM_COLORS = { 
        0: [0, 100, 100, 255],   # Dark Cyan/Teal (Contrast with Cyan/Blue)
        1: [101, 67, 33, 255],   # Dark Brown (Contrast with Red)
        2: [255, 105, 180, 255], # Hot Pink (Contrast with Magenta)
        3: [75, 0, 130, 255],    # Indigo/Deep Purple (Contrast with Blue)
        4: [128, 128, 0, 255],   # Olive (Contrast with Yellow/Green)
        5: [70, 130, 180, 255]   # Steel Blue (Contrast with pure Blue)
    }

    # Respawn / Cooldown
    RESPAWN_COOLDOWN = 5
    # Discovery
    CELL_SIZE = 5
    DISCOVERY_COOLDOWN = 50
    # Resource ID Counter
    RESOURCE_ID_COUNTER = 0

    # For Directional Combat (Attacker Facing)
    ATTACKER_FACING_THRESHOLD = 0.3         # Dot product: cosine of angle. 0.5 is ~60deg cone, 0.3 is wider.
    ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER = 0.3 # Damage multiplier if attacker isn't facing target
    ATTACKER_STATIONARY_DAMAGE_MULTIPLIER = 0.6 # Damage multiplier if attacker is stationary

    # --- Memory & Certainty Decay ---
    # These constants are no longer used as the hardcoded memory decay mechanism
    # has been replaced by passing `last_observed_step` to the agent's policy.

    # For Attacker Speed Bonus
    ATTACKER_SPEED_BONUS_THRESHOLD_RATIO = 0.6 # Attacker must be moving at this fraction of their max speed for bonus
    MAX_ATTACKER_SPEED_DAMAGE_MOD = 0.20       # Max bonus to damage % from speed (e.g., 0.15 means up to +15% damage)

    # For Defender Mitigations
    DEFENDER_STRENGTH_MITIGATION_MOD = 0.15   # e.g., 0.05 means each point of eff_strength reduces damage by 15% of that strength value
    DEFENDER_MASS_MITIGATION_MOD = 0.05       # e.g., 0.03 means each unit of mass over 1.0 reduces damage by 3% (of that mass diff)
    MIN_DAMAGE_AFTER_MITIGATION = 0.1         # Minimum % of original damage after all personal mitigations (e.g. 10%)

    # For Flank/Rear Attack Bonus
    FLANK_REAR_ATTACK_BONUS_MULTIPLIER = 1.25 # e.g., 25% damage bonus
    FLANK_REAR_DEFENDER_FACING_THRESHOLD = -0.2 # Dot product: if defender is facing away from attacker beyond this

    AGENT_HEALTH_EFFECTIVENESS_FLOOR = 0.25 # At very low health, agent operates at 25% of (strength*energy*carry_penalty)
    AGENT_HEALTH_EFFECTIVENESS_EXPONENT = 0.75 # Controls the curve of health impact (0.5-1.0 common)

    # For Normalization
    MIN_AGENT_VELOCITY_FOR_DIRECTION = 0.2    # Min speed to be considered "moving" for directional checks

    # Sensing / Interaction Radii
    SENSING_RANGE_FRACTION = 0.04
    OBS_RADIUS = SENSING_RANGE_FRACTION * D_MAX # Recalculated in __init__
    COMM_RADIUS = OBS_RADIUS
    COMBAT_RADIUS = AGENT_RADIUS * 3
    # Teammate death vicinity: if an agent sees a teammate die within this radius and doesn't help, they're at fault
    # Range: 0.5-0.75 * obs_radius (default: 0.65 for midpoint)
    TEAMMATE_DEATH_VICINITY_RADIUS_FACTOR = 0.65

    # Obstacle parameters
    NUM_OBSTACLES = 30
    OBSTACLE_RADIUS_MIN = 10
    OBSTACLE_RADIUS_MAX = 50
    OBSTACLE_HALF_HEIGHT = 20

    # Resource parameters
    RESOURCE_MIN_SIZE = 3.0
    RESOURCE_MAX_SIZE = 10.0
    RESOURCE_PICKUP_RADIUS = 7.5 # How close an agent must be to pick up a resource.
    RESOURCE_MIN_RADIUS_PB = 3.0 # Physical radius in PyBullet for a resource of minimum conceptual size.
    RESOURCE_MAX_RADIUS_PB = 12.0 # Physical radius in PyBullet for a resource of maximum conceptual size.
    RESOURCE_RESPAWN_COOLDOWN = 20
    NUM_RESOURCES = 250
    ACTIVE_RESOURCE_LIMIT = 350

    # Agent Combat/Health/Energy parameters
    AGENT_MAX_ENERGY = 100.0
    ENERGY_MOVEMENT_COST = 0.00005  # Increased 10x for more strategic movement
    ENERGY_BASE_COST = 0.00005      # Increased 40x for faster depletion
    ENERGY_GRAPPLE_COST_MULTIPLIER = 5.0 
    ENERGY_RECHARGE_RATE = 0.5     # Increased 2.5x for faster recovery
    RECHARGE_DISTANCE_THRESHOLD = 50
    AGENT_MAX_HEALTH = 100.0
    # === AGENT COMBAT & GRAPPLING ===
    AGENT_BASE_DAMAGE = 2.0  # Base damage dealt in combat (Increased from 0.15 to make combat lethal)
    GRAPPLE_STAMINA_DRAIN = 0.25 # Extra energy cost per step for pulling while grappling
    GRAPPLE_TORQUE_ESCAPE_STRENGTH = 0.3 # How much opposing torque weakens a grapple (0.0 to 1.0)
    # The percentage of their grip strength a grappled agent can use to counter-grapple.
    GRAPPLED_AGENT_COUNTER_GRIP_SCALE = 0.4
    GRAPPLE_FATIGUE_RATE = 0.005 # Percentage of grip strength lost per step while grappling
    GRAPPLE_CRUSH_DAMAGE_RATE = 2.0 # Base damage per step dealt to a grappled agent (Increased from 0.1)
    AGENT_STRENGTH_DAMAGE_MOD = 0.5  # How much strength modifies base damage
    AGENT_ENERGY_DAMAGE_MOD = 0.5    # How much energy level modifies base damage
    AGENT_SIZE_DAMAGE_MOD = 0.2
    AGENT_DAMAGE_STOCHASTICITY = 0.1
    AGENT_SLOWED_DURATION = 5  # Reduced from 30 for quicker recovery
    AGENT_SLOWED_FACTOR = 0.5   # Reduced from 0.8 for more severe penalty
    AGENT_HEALTH_RECOVERY_RATE = 0.1
    AGENT_DEATH_ENERGY_THRESHOLD = 0.0
    AGENT_DEATH_DROP_RESOURCE_SIZE = AGENT_RADIUS  # Example size, can be tuned. Typically non-cooperative.
    
    # This will be the NEW central hub for all reward configurations.
    # Each reward has a default base value and a default multiplier (scale).
    # The GUI will only tune the 'multiplier'.
    REWARD_CONFIG = {
        # Resource & Delivery
        'r_attachment':           {'default_value': 5.0,   'default_multiplier': 1.0},
        'r_holding':              {'default_value': 0.1,   'default_multiplier': 1.0}, # NEW: Continuous reward for holding
        'r_progress':             {'default_value': 10.0,  'default_multiplier': 1.0},
        'r_progress_positive':    {'default_value': 10.0,  'default_multiplier': 1.0}, # NEW: Reward for moving closer
        'r_progress_negative':    {'default_value': 10.0,  'default_multiplier': 1.0}, # NEW: Penalty for moving away
        'r_delivery':             {'default_value': 10.0,  'default_multiplier': 1.0},
        'coop_collection_bonus':  {'default_value': 2.0,   'default_multiplier': 1.0}, # This is a multiplier itself

        # Grappling
        'r_grapple_control':      {'default_value': 0.5,   'default_multiplier': 1.0},
        'r_grapple_controlled':   {'default_value': -0.5,  'default_multiplier': 1.0},
        'r_grapple_break':        {'default_value': 0.5,  'default_multiplier': 1.0},
        'r_torque_win':           {'default_value': 0.5,   'default_multiplier': 1.0},

        # Combat & Agent Survival
        'r_combat_win':           {'default_value': 25.0,  'default_multiplier': 1.0},
        'r_combat_lose':          {'default_value': -25.0, 'default_multiplier': 1.0},
        'r_combat_continuous':    {'default_value': 1.0,   'default_multiplier': 1.0},
        'r_death':                {'default_value': -50.0, 'default_multiplier': 1.0},
        'r_teammate_lost_nearby': {'default_value': -5.0,  'default_multiplier': 1.0},

        # HIVE REWARDS
        'r_hive_attack_continuous': {'default_value': 0.2,   'default_multiplier': 1.0}, # For damaging enemy hives
        'r_hive_health_continuous': {'default_value': 0.001, 'default_multiplier': 1.0}, # For total health of owned hives
        'r_hive_capture':           {'default_value': 50.0,  'default_multiplier': 1.0}, # For capturing a destroyed hive
        'r_hive_rebuild':           {'default_value': 25.0,  'default_multiplier': 1.0}, # For rebuilding your own hive
        'r_hive_destroyed_penalty': {'default_value': -75.0, 'default_multiplier': 1.0}, # Team-wide penalty for losing a hive
        
        # Discovery & Exploration
        'r_resource_found':       {'default_value': 2.0,   'default_multiplier': 1.0},
        'r_obstacle_found':       {'default_value': 1.0,   'default_multiplier': 1.0},
        'r_enemy_found':          {'default_value': 5.0,   'default_multiplier': 1.0},
        'r_hive_found':           {'default_value': 10.0,  'default_multiplier': 1.0},
        'r_exploration_intrinsic':{'default_value': 0.1,   'default_multiplier': 1.0},
    }

    # === NEW GRAPPLE PARAMETER ===
    GRAPPLE_INITIATION_DISTANCE_MULTIPLIER = 1.0 # Multiplier for agent+enemy radius to determine grapple range

    # Hive parameters
    HIVE_DECAY_RATE = 0.02
    HIVE_HEALTH_DECAY_IF_EMPTY = 0.03
    HIVE_DELIVERY_RADIUS = 42.0
    HIVE_MAX_HEALTH = 100.0
    HIVE_MAX_FOOD = 100.0
    HIVE_ATTACK_RADIUS = 42
    HIVE_DEFENSE_RADIUS = 42
    HIVE_DAMAGE_FACTOR = 0.2
    HIVE_LOST_TIME_THRESHOLD = 50
    HIVE_HEALTH_RECOVERY_RADIUS = 42
    HIVE_RADIUS_ASSUMED = 25.0

    # For the "Bleed" effect on damage
    HIVE_DAMAGE_POINTS_PER_BLEED_CHUNK = 5.0  # Every 5 points of damage (to food or health) triggers one resource chunk to drop.
    HIVE_BLEED_RESOURCE_SIZE = 3.0           # The conceptual size of a single dropped chunk (same as min size)

    FOOD_COST_PER_AGENT_HEALTH_POINT = 0.5
    FOOD_COST_PER_AGENT_ENERGY_POINT = 0.25  

    # For the "Core" drop on hive destruction
    HIVE_CORE_MIN_SIZE = 8.0                 # The minimum size of a dropped coop core
    HIVE_CORE_MAX_SIZE = 15.0                # The maximum size of a dropped coop core
    HIVE_CORE_FOOD_TO_SIZE_RATIO = 0.1       # e.g., 100 food points = 10.0 size core
    
    # Action Space Dimensions
    MOVEMENT_DIM = 2
    PICKUP_DIM = 3
    JOINT_ACTION_DIM = MOVEMENT_DIM + 1

    # Debug and rendering parameters
    FPS = 30
    DISP_WIDTH = 750
    DISP_HEIGHT = 750
    MAX_STEPS = 500  # Reduced from 1500 to 500 for faster episodes

    DEFAULT_RESOURCE_PUSH_FORCE_SCALE = 0.0
    DEFAULT_RESOURCE_BASE_MASS = 0.5
    DEFAULT_RESOURCE_MASS_SCALE = 0.69
    DEFAULT_AGENT_FORCE_SCALE = 1.0  # Reduced from 42.0 for realistic physics
    DEFAULT_MOVEMENT_FORCE_SCALE = 1.0  # New: separate movement force scale
    DEFAULT_INTERACTION_FORCE_SCALE = 1.0  # New: separate interaction force scale

    # PyBullet Collision Groups
    COLLISION_GROUP_AGENT = 1
    COLLISION_GROUP_RESOURCE = 2
    COLLISION_GROUP_OBSTACLE = 4
    COLLISION_GROUP_HIVE = 8
    COLLISION_GROUP_GROUND = 32 # Or another unused power of 2 bit

    # --- GPU Occlusion Parameters ---
    OCCLUSION_FIELD_RESOLUTION = 64  # Resolution of the global occlusion field (e.g., 64x64 or 128x128)
                                    # Higher means more accuracy but more memory/computation.
    NUM_LOS_SAMPLE_POINTS = 3        # Number of intermediate points to check along LOS segment (e.g., 3-5)
    LOS_OCCLUSION_THRESHOLD = 0.7    # If summed occlusion at sample points > threshold, LOS is blocked. Tune this.
                                    # Lower value means entities occlude more easily.
    LOS_GRID_CELL_SIZE = 5.0 # world units per cell for the global occlusion map

    # --- Adaptive Clustering Parameters ---
    ADAPTIVE_CLUSTERING_MAX_NEIGHBORS = 32  # Maximum neighbors before clustering distant nodes
    ADAPTIVE_CLUSTERING_CELL_SIZE_FACTOR = 2.0  # Factor to divide CLUSTER_CELL_SIZE by for adaptive clustering



class ChannelConstants(nn.Module):
    #### Constants
    CLUSTER_CELL_SIZE = 100.0 # Define the base cell size for the farthest periphery clustering
    CLUSTER_MERGE_THRESHOLD = 50.0 # For two-pass clustering, the distance at which to merge clusters.

    REWARD_COMPONENT_KEYS = [
        # Resource & Delivery
        "r_attachment", "r_holding", "r_progress", "r_progress_positive", "r_progress_negative", "r_delivery", 
        # Combat
        "r_combat_win", "r_combat_lose", "r_combat_continuous", 
        "r_death", "r_teammate_lost_nearby",
        # Grappling
        "r_grapple_control", "r_grapple_controlled", "r_grapple_break", "r_torque_win",
        # Hive 
        "r_hive_attack_continuous", "r_hive_health_continuous", "r_hive_capture", 
        "r_hive_rebuild", "r_hive_destroyed_penalty",
        # Discovery & Exploration
        "r_resource_found", "r_obstacle_found", "r_enemy_found", "r_hive_found",
        "r_exploration_intrinsic"
    ]
    NUM_REWARD_COMPONENTS = len(REWARD_COMPONENT_KEYS)

    GLOBAL_CUE_DIM = 20 # Increased to 20

    # --- De-duplicated) Map Channel Constants ---
    RAW_CH = {
        # Presence flags for entities within the agent's immediate Field of View (FoV)
        'ally_presence': 0,
        'enemy_presence': 1,
        'resource_presence': 2,
        'coop_resource_presence': 3,
        'hive_ally_presence': 4,
        'hive_enemy_presence': 5,
        'obstacle_presence': 6,
        'self_presence': 7,
    }
    RAW_CH_COUNT = len(RAW_CH)


    # Ensure RAW_CH is defined before this point (it should be)
    try:
        RAW_CH_IDX_TO_NAME = {v: k for k, v in RAW_CH.items()}
    except NameError:
        print("FATAL ERROR: RAW_CH dictionary not defined before creating inverse map.")
        sys.exit(1) # Make this fatal if RAW_CH is critical path


    # Channels for the persistent, long-term memory map (occlusion map)
    OCC_CH = {
        'obstacle_presence': 0,            # Binary flag for obstacle presence (1.0 if ever seen)
        'last_seen_resource': 1,           # Normalized age of the last resource observation
        'last_seen_coop_resource': 2,      # Normalized age of the last coop resource observation
        'last_seen_hive_ally': 3,          # Normalized age of the last allied hive observation
        'last_seen_hive_enemy': 4,         # Normalized age of the last enemy hive observation
        'last_seen_ally': 5,               # Normalized age of the last ally observation
        'last_seen_enemy': 6,              # Normalized age of the last enemy observation
        'last_seen_self': 7,               # ADDED: Normalized age of the agent's own last known position
        'explored': 8,                     # Binary flag if the cell has ever been observed
        'vec_hive_x': 9,                   # X component of the vector to the agent's home hive
        'vec_hive_y': 10,                   # Y component of the vector to the agent's home hive
        'step_norm': 11,                    # Current normalized step count of the episode
        'you_are_here': 12,                # A 1-hot channel indicating the agent's own cell
        'coverage': 13,                    # Channel for spatial density/coverage
    }
    OCC_CH_COUNT = len(OCC_CH)

    # Mapping from the OCC_CH indices to human-readable names
    OCC_CH_IDX_TO_NAME = {v: k for k, v in OCC_CH.items()}
    CRITIC_EDGE_FEATURE_DIM = 36 # used for GNN

    NODE_TYPE = {
        'agent': 0, # Example, ensure this matches what your system expects
        'resource': 1,
        'hive': 2,
        'obstacle': 3
    }
    NUM_NODE_TYPES = len(NODE_TYPE)


    NODE_FEATURE_MAP = {
        # Core attributes
        'pos_x_norm': 0, 'pos_y_norm': 1, 'vel_x_norm': 2, 'vel_y_norm': 3,
        'size_norm': 4,
        
        # Type & Team
        'node_type_encoded': 5, 'team_id': 6,
        
        # Agent-specific
        'energy_norm': 7, 'health_norm': 8, 'strength_norm': 9,
        'base_speed_norm': 10, 'obs_radius_norm': 11,
        'is_carrying': 12, 'is_slowed': 13,
        
        # Resource-specific
        'value_or_size_norm': 14, # For agents: health*strength, for resources: conceptual size
        'is_cooperative': 15, 'is_delivered': 16,
        
        # Hive-specific
        'hive_food_norm': 17, 'is_destroyed': 18,
        
        # ID & Action
        'agent_id': 19, # <<< NEW: Generic ID for ALL entities
        'pickup_action': 20,
        
        # Special Flags
        'is_ego': 21,
        'is_grappling': 22,
        'is_grappled': 23,
    }
    _max_index = max(NODE_FEATURE_MAP.values())
    NODE_FEATURE_DIM = _max_index + 1

    SELF_OBS_MAP = {
        'pos_x_norm': 0,
        'pos_y_norm': 1,
        'vel_x_norm': 2,
        'vel_y_norm': 3,
        'rel_res_x_norm': 4,
        'rel_res_y_norm': 5,
        'rel_hive_x_norm': 6,
        'rel_hive_y_norm': 7,
        'is_carrying': 8,
        'hive_dist_norm': 9,
        'res_dist_norm': 10,  # NEW: Distance to nearest resource
        'speed_norm': 11,
        'radius_norm': 12,
        'obs_radius_norm': 13,
        'strength_norm': 14,
        'hive_health_norm': 15,
        'energy_norm': 16,
        'health_norm': 17,
        'team_energy_norm': 18,
        'boundary_x1': 19,
        'boundary_x2': 20,
        'boundary_y1': 21,
        'boundary_y2': 22,
        'grip_strength_norm': 23,
        'agent_id': 24,
        'team_id_val': 25,
        'alignment_score': 26, # NEW: dot(vel, rel_hive) - Explicit navigation signal
        # --- V4: GRAPPLE AWARENESS ---
        'is_grappling': 27,
        'is_grappled': 28,
        'applied_torque_norm': 29,
        'grapple_tension_norm': 30,
    }
    SELF_OBS_DIM = len(SELF_OBS_MAP)


    # Expanded Node Features for Persistent GNN Memory
    MEM_NODE_FEAT_IDX = {
        **NODE_FEATURE_MAP,
        'last_observed_step': NODE_FEATURE_DIM,
        'node_status': NODE_FEATURE_DIM + 1, # e.g., 0=normal, 1=clustered
    }
    MEM_NODE_FEATURE_DIM = len(MEM_NODE_FEAT_IDX)

# --- Expose commonly used constants at module level for convenience ---
_CM = ConstantsMain
_CH = ChannelConstants

# Selected names from ConstantsMain
for _name in [
    'WIDTH','HEIGHT','D_MAX','AGENT_RADIUS','AGENT_BASE_STRENGTH','BEE_SPEED','AGENT_BASE_SPEED','EXTRA_RANDOMNESS_STD','ROLE_EMBED_DIM',
    'TEAM_COLORS','RESPAWN_COOLDOWN','CELL_SIZE','DISCOVERY_COOLDOWN','RESOURCE_ID_COUNTER',
    'ATTACKER_FACING_THRESHOLD','ATTACKER_NOT_FACING_DAMAGE_MULTIPLIER','ATTACKER_STATIONARY_DAMAGE_MULTIPLIER',
    'ATTACKER_SPEED_BONUS_THRESHOLD_RATIO','MAX_ATTACKER_SPEED_DAMAGE_MOD','DEFENDER_STRENGTH_MITIGATION_MOD',
    'DEFENDER_MASS_MITIGATION_MOD','MIN_DAMAGE_AFTER_MITIGATION','FLANK_REAR_ATTACK_BONUS_MULTIPLIER',
    'FLANK_REAR_DEFENDER_FACING_THRESHOLD','AGENT_HEALTH_EFFECTIVENESS_FLOOR','AGENT_HEALTH_EFFECTIVENESS_EXPONENT',
    'MIN_AGENT_VELOCITY_FOR_DIRECTION','SENSING_RANGE_FRACTION','OBS_RADIUS','COMM_RADIUS','COMBAT_RADIUS','TEAMMATE_DEATH_VICINITY_RADIUS_FACTOR',
    'NUM_OBSTACLES','OBSTACLE_RADIUS_MIN','OBSTACLE_RADIUS_MAX','OBSTACLE_HALF_HEIGHT','RESOURCE_MIN_SIZE',
    'RESOURCE_MAX_SIZE','RESOURCE_PICKUP_RADIUS','RESOURCE_PICKUP_DISTANCE_MULTIPLIER','RESOURCE_MIN_RADIUS_PB','RESOURCE_MAX_RADIUS_PB','RESOURCE_RESPAWN_COOLDOWN','NUM_RESOURCES','ACTIVE_RESOURCE_LIMIT','AGENT_MAX_ENERGY',
    'ENERGY_MOVEMENT_COST','ENERGY_BASE_COST','ENERGY_GRAPPLE_COST_MULTIPLIER','ENERGY_RECHARGE_RATE',
    'RECHARGE_DISTANCE_THRESHOLD','AGENT_MAX_HEALTH','AGENT_BASE_DAMAGE',
    'GRAPPLE_STAMINA_DRAIN',
    'GRAPPLE_TORQUE_ESCAPE_STRENGTH',
    'GRAPPLED_AGENT_COUNTER_GRIP_SCALE',
    'GRAPPLE_FATIGUE_RATE',
    'GRAPPLE_CRUSH_DAMAGE_RATE',
    'AGENT_STRENGTH_DAMAGE_MOD','AGENT_ENERGY_DAMAGE_MOD','AGENT_SIZE_DAMAGE_MOD','AGENT_DAMAGE_STOCHASTICITY','AGENT_SLOWED_DURATION',
    'AGENT_SLOWED_FACTOR','AGENT_HEALTH_RECOVERY_RATE','AGENT_DEATH_ENERGY_THRESHOLD','AGENT_DEATH_DROP_RESOURCE_SIZE',
    'REWARD_CONFIG', # <-- ADD NEW CONFIG
    'GRAPPLE_INITIATION_DISTANCE_MULTIPLIER',
    'HIVE_DECAY_RATE','HIVE_HEALTH_DECAY_IF_EMPTY','HIVE_DELIVERY_RADIUS','HIVE_MAX_HEALTH',
    'HIVE_MAX_FOOD','HIVE_ATTACK_RADIUS','HIVE_DEFENSE_RADIUS','HIVE_DAMAGE_FACTOR','HIVE_LOST_TIME_THRESHOLD',
    'HIVE_HEALTH_RECOVERY_RADIUS','HIVE_RADIUS_ASSUMED','HIVE_DAMAGE_POINTS_PER_BLEED_CHUNK','HIVE_BLEED_RESOURCE_SIZE',
    'HIVE_CORE_FOOD_TO_SIZE_RATIO','HIVE_CORE_MIN_SIZE','HIVE_CORE_MAX_SIZE',
    'FOOD_COST_PER_AGENT_HEALTH_POINT','FOOD_COST_PER_AGENT_ENERGY_POINT',
    'MOVEMENT_DIM', 'PICKUP_DIM', 'JOINT_ACTION_DIM',
    'FPS','DISP_WIDTH','DISP_HEIGHT','MAX_STEPS',
    'DEFAULT_RESOURCE_PUSH_FORCE_SCALE','DEFAULT_RESOURCE_BASE_MASS','DEFAULT_RESOURCE_MASS_SCALE','DEFAULT_AGENT_FORCE_SCALE',
    'DEFAULT_MOVEMENT_FORCE_SCALE','DEFAULT_INTERACTION_FORCE_SCALE',
    'COLLISION_GROUP_AGENT','COLLISION_GROUP_RESOURCE','COLLISION_GROUP_OBSTACLE','COLLISION_GROUP_HIVE','COLLISION_GROUP_GROUND',
    'OCCLUSION_FIELD_RESOLUTION','NUM_LOS_SAMPLE_POINTS','LOS_OCCLUSION_THRESHOLD','LOS_GRID_CELL_SIZE',
    'ADAPTIVE_CLUSTERING_MAX_NEIGHBORS','ADAPTIVE_CLUSTERING_CELL_SIZE_FACTOR'
]:
    # Check if the attribute exists in _CM before getting it
    if hasattr(_CM, _name):
        globals()[_name] = getattr(_CM, _name)

# Selected names from ChannelConstants
for _name in [
    'CLUSTER_CELL_SIZE','REWARD_COMPONENT_KEYS','GLOBAL_CUE_DIM','RAW_CH','RAW_CH_COUNT','RAW_CH_IDX_TO_NAME',
    'OCC_CH','OCC_CH_COUNT','CRITIC_EDGE_FEATURE_DIM','NODE_TYPE','NUM_NODE_TYPES','NODE_FEATURE_MAP','NODE_FEATURE_DIM',
    'MEM_NODE_FEAT_IDX','MEM_NODE_FEATURE_DIM', 'SELF_OBS_MAP', 'SELF_OBS_DIM',
    'NUM_REWARD_COMPONENTS'
]:
    globals()[_name] = getattr(_CH, _name)
# --- V2 ---
# I'm moving CLUSTER_MERGE_THRESHOLD to be exported here as well.
if hasattr(_CH, 'CLUSTER_MERGE_THRESHOLD'):
    globals()['CLUSTER_MERGE_THRESHOLD'] = getattr(_CH, 'CLUSTER_MERGE_THRESHOLD')


# Derived convenience values
CRITIC_GRID_SIZE = 32
ACTOR_MAP_GRID_SIZE = 32
