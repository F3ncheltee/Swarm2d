import numpy as np
import torch
from torch_geometric.data import Data
import traceback
from torch_geometric.data import Batch
import sys
from collections import defaultdict

# To run this script, you must be in the 'Swarm2d' directory.
# This allows the imports within the environment to work correctly.
try:
    from env.env import Swarm2DEnv
    # REMOVED: BatchedPersistentGraphMemory is handled internally by the environment
    # from env.batched_graph_memory import BatchedPersistentGraphMemory
    from constants import OBS_RADIUS, CLUSTER_CELL_SIZE, MEM_NODE_FEAT_IDX, NODE_FEATURE_DIM, MEM_NODE_FEATURE_DIM, SELF_OBS_MAP
except ImportError:
    # If the script is run from the project root, this should work
    from Swarm2d.env.env import Swarm2DEnv
    # REMOVED: BatchedPersistentGraphMemory is handled internally by the environment
    # from Swarm2d.env.batched_graph_memory import BatchedPersistentGraphMemory
    from Swarm2d.constants import OBS_RADIUS, CLUSTER_CELL_SIZE, MEM_NODE_FEAT_IDX, NODE_FEATURE_DIM, MEM_NODE_FEATURE_DIM, SELF_OBS_MAP


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

def main():
    # Open a log file
    f = open('test_output.log', 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    print("Running Swarm2DEnv example with DETAILED observation inspection...")
    env = None  # Initialize env to None for cleanup
    try:
        # Configuration dictionary with updated parameters from tuning scripts
        env_config = {
            # Base simulation setup from the original test_env.py
            'num_teams': 6,
            'num_agents_per_team': 20,
            'num_resources': 100,
            'num_obstacles': 10,
            'max_steps': 1000,
            'render_mode': 'both', # Use 'gui', 'human', 'both', or None (for headless/fastest mode).
            'debug': False,
            'use_gpu_occlusion_in_env': False,
            'use_pybullet_raycasting': True,

            # --- NEWLY ADDED Agent Base Parameters (from env.__init__) ---
            'agent_radius': 3.0,
            'agent_base_strength': 1.0,
            'agent_max_energy': 100.0,
            'agent_max_health': 100.0,
            'sensing_range_fraction': 0.05,
            'recency_normalization_period': 250.0,

            # --- Parameters from tune_phase5_combat.py ---
            # From base_env_config
            'movement_force_scale': 15.0,
            'pb_agent_linear_damping': 0.11,
            'pb_agent_lateral_friction': 0.5,
            'pb_agent_angular_damping': 0.4,
            'resource_base_mass': 0.075,
            'resource_interaction_force_scale': 1.2,
            'pb_resource_constraint_max_force': 3000,
            'pb_res_friction_dynamic': 0.25,
            'pb_res_damping_dynamic': 0.25,
            'bee_speed': 200.0,
            
            'resource_mass_scale_factor': 1.4,
            'pb_coop_resource_constraint_max_force': 10000,

            # From parameter_sweep (using first/representative values)
            'grappled_agent_counter_grip_scale': 0.3,
            'grapple_fatigue_rate': 0.02,
            'grapple_crush_damage_rate': 1.0,
            'grapple_struggle_damage_rate': 0.5,
            'pb_agent_constraint_max_force': 10000,
            'grapple_torque_escape_strength': 0.6,
            'grapple_torque_scale': 25.0, # NEWLY ADDED
            'grapple_momentum_bonus_scale': 0.1,
            'agent_interaction_force_scale': 0.35,
            'grapple_momentum_decay': 0.95,
            'grapple_crit_chance': 0.05,
            'grapple_crit_multiplier': 3.0,
            'grapple_rear_crit_bonus_multiplier': 2.5,
            'debug_mode': False,
            'agent_randomization_factors': {
                'bee_speed': {'base': 200.0, 'rand': 0.2},
                'agent_radius': {'base': 3.0, 'rand': 0.2},
                'agent_base_strength': {'base': 1.0, 'rand': 0.2},
                'agent_max_energy': {'base': 100.0, 'rand': 0.2},
                'agent_max_health': {'base': 100.0, 'rand': 0.2},
                'sensing_range_fraction': {'base': 0.05, 'rand': 0.2}
            }
        }

        env = Swarm2DEnv(**env_config)
        print("Environment created with tuned parameters.")


        # --- Helper function to print observation details ---
        # MODIFIED: Added 'current_env' parameter
        def print_detailed_observation(current_env, obs_item, agent_id_str, step_str):
            print(f"\n--- Agent {agent_id_str} Observation ({step_str}) ---")
            if not isinstance(obs_item, dict):
                print(f"  ERROR: Observation is not a dict, type: {type(obs_item)}")
                return

            # 1. Self Observation
            self_obs = obs_item.get('self')
            print("\n  1. Self Observation ('self'):")
            if self_obs is not None and isinstance(self_obs, torch.Tensor):
                print(f"    Shape: {self_obs.shape}, Dtype: {self_obs.dtype}, Device: {self_obs.device}")
                print(f"    Values (rounded to 3 decimal places):")
                self_obs_np = self_obs.cpu().numpy()
                self_feat_names = [
                    "pos_x_norm", "pos_y_norm", "vel_x_norm", "vel_y_norm",
                    "rel_res_x_norm", "rel_res_y_norm", "rel_hive_x_norm", "rel_hive_y_norm",
                    "carrying_flag", "current_hive_dist_norm", "norm_eff_speed",
                    "agent_radius_norm", "obs_radius_norm", "norm_eff_strength",
                    "own_hive_health_norm", "energy_norm", "health_norm", "team_avg_energy_norm",
                    "boundary_left", "boundary_right", "boundary_bottom", "boundary_top",
                    "grip_strength_norm", "agent_id_val", "team_id_val"
                ]
                for i, val in enumerate(self_obs_np):
                    feat_name = self_feat_names[i] if i < len(self_feat_names) else f"feat_{i}"
                    print(f"      {feat_name:<25}: {val:.3f}")
                if torch.isnan(self_obs).any() or torch.isinf(self_obs).any():
                    print(f"    WARNING: 'self' contains NaN/Inf!")
            else:
                print(f"    Type: {type(self_obs)}, Value: {self_obs}")

            # 2. Map Observation
            map_obs = obs_item.get('map')
            print("\n  2. Map Observation ('map'):")
            if map_obs is not None and isinstance(map_obs, torch.Tensor):
                print(f"    Shape: {map_obs.shape}, Dtype: {map_obs.dtype}, Device: {map_obs.device}")
                print(f"    Min: {map_obs.min().item():.3f}, Max: {map_obs.max().item():.3f}, Mean: {map_obs.mean().item():.3f}, Sum: {map_obs.sum().item():.3f}")
                if torch.isnan(map_obs).any() or torch.isinf(map_obs).any():
                    print(f"    WARNING: 'map' contains NaN/Inf!")
                if map_obs.sum().item() == 0 and map_obs.numel() > 0:
                    print(f"    INFO: 'map' is all zeros.")
                else:
                    ally_presence_ch_idx = current_env.raw_ch_def_dict_const.get('ally_presence', -1)
                    if ally_presence_ch_idx != -1 and ally_presence_ch_idx < map_obs.shape[0]:
                         print(f"    Sum of 'ally_presence' channel ({ally_presence_ch_idx}): {map_obs[ally_presence_ch_idx].sum().item():.3f}")
                    enemy_presence_ch_idx = current_env.raw_ch_def_dict_const.get('enemy_presence', -1)
                    if enemy_presence_ch_idx != -1 and enemy_presence_ch_idx < map_obs.shape[0]:
                         print(f"    Sum of 'enemy_presence' channel ({enemy_presence_ch_idx}): {map_obs[enemy_presence_ch_idx].sum().item():.3f}")
                    resource_presence_ch_idx = current_env.raw_ch_def_dict_const.get('resource_presence', -1)
                    if resource_presence_ch_idx != -1 and resource_presence_ch_idx < map_obs.shape[0]:
                         print(f"    Sum of 'resource_presence' channel ({resource_presence_ch_idx}): {map_obs[resource_presence_ch_idx].sum().item():.3f}")
            else:
                print(f"    Type: {type(map_obs)}, Value: {map_obs}")

            # 3. Unified Graph Observation (Live + Memory)
            graph_obs = obs_item.get('graph')
            print("\n  3. Unified Graph Observation ('graph'):")
            print("     (This graph combines live perceptions with persistent memory)")
            if graph_obs is not None and isinstance(graph_obs, Data):
                print(f"    Type: {type(graph_obs)}")
                print(f"    Num Nodes: {graph_obs.num_nodes}")
                print(f"    Num Edges: {graph_obs.num_edges}")

                if hasattr(graph_obs, 'x') and graph_obs.x is not None and graph_obs.num_nodes > 0:
                    print(f"    Node Features (x) - Shape: {graph_obs.x.shape}, Dtype: {graph_obs.x.dtype}, Device: {graph_obs.x.device}")
                    
                    # --- DETAILED NODE ANALYSIS ---
                    node_type_idx = current_env.node_feature_map_const.get('node_type_encoded', -1)
                    certainty_idx = MEM_NODE_FEAT_IDX.get('certainty', -1)
                    cluster_count_idx = graph_obs.x.shape[1] - 1 # Assumed to be the last feature
                    is_ego_idx = current_env.node_feature_map_const.get('is_ego', -1)
                    team_id_idx = current_env.node_feature_map_const.get('team_id', -1)
                    agent_team_id = obs_item['self'][SELF_OBS_MAP['team_id_val']]


                    type_counts = defaultdict(int)
                    type_names = {v: k for k, v in current_env.node_type_def_const.items()} # For non-agent types
                    
                    live_nodes = 0
                    mem_nodes = 0
                    clustered_nodes = 0

                    if graph_obs.num_nodes > 0:
                        for i in range(graph_obs.num_nodes):
                            # Count types with more detail for agents
                            if node_type_idx != -1:
                                node_type_val = int(graph_obs.x[i, node_type_idx].item())
                                
                                # Check if it's an agent type
                                is_agent = False
                                for agent_type_key in ['agent', 'self_agent', 'ally_agent', 'enemy_agent']:
                                    if current_env.node_type_def_const.get(agent_type_key) == node_type_val:
                                        is_agent = True
                                        break
                                
                                if is_agent:
                                    if is_ego_idx != -1 and graph_obs.x[i, is_ego_idx].item() > 0.5:
                                        type_counts['self_agent'] += 1
                                    elif team_id_idx != -1:
                                        node_team_id = graph_obs.x[i, team_id_idx].item()
                                        if node_team_id == agent_team_id:
                                            type_counts['ally_agent'] += 1
                                        else:
                                            type_counts['enemy_agent'] += 1
                                    else:
                                        type_counts['unknown_agent'] += 1 # Fallback
                                else:
                                    # It's a non-agent type (resource, hive, obstacle)
                                    type_name = type_names.get(node_type_val, 'Unknown')
                                    type_counts[type_name] += 1
                            
                            # Count live vs memory
                            if certainty_idx != -1 and certainty_idx < graph_obs.x.shape[1]:
                                certainty = graph_obs.x[i, certainty_idx].item()
                                if certainty < 0.99: # Allow for float precision
                                    mem_nodes += 1
                                else:
                                    live_nodes += 1
                            else: # If no certainty, assume all are live
                                live_nodes += 1

                            # Count clustered
                            if cluster_count_idx != -1:
                                if graph_obs.x[i, cluster_count_idx].item() > 1:
                                    clustered_nodes += 1
                        
                        print("\n    --- Graph Content Summary ---")
                        print(f"      Live Nodes: {live_nodes}")
                        print(f"      Memory Nodes: {mem_nodes}")
                        print(f"      Clustered Nodes: {clustered_nodes}")
                        print(f"      Node Type Breakdown:")
                        # Define a preferred order for display
                        display_order = ['self_agent', 'ally_agent', 'enemy_agent', 'resource', 'hive', 'obstacle']
                        
                        for type_name in display_order:
                            if type_counts[type_name] > 0:
                                print(f"        - {type_name:<15}: {type_counts[type_name]}")
                        
                        # Print any other types that might not be in the display order
                        for type_name, count in type_counts.items():
                             if type_name not in display_order and count > 0:
                                 print(f"        - {type_name:<15}: {count}")

                        print("    ---------------------------\n")


                    if graph_obs.x.numel() > 0 : # Check if tensor is not empty
                        print(f"      Min: {graph_obs.x.min().item():.3f}, Max: {graph_obs.x.max().item():.3f}, Mean: {graph_obs.x.mean().item():.3f}")
                        print(f"      First Node Features (first 5 of {graph_obs.x.shape[1]}): {np.round(graph_obs.x[0, :min(5, graph_obs.x.shape[1])].cpu().numpy(), 3)}")
                    else:
                        print("      Node Features (x): Tensor is empty but num_nodes > 0 (unexpected).")


                    # CORRECTED PLACEMENT AND USAGE OF is_ego_idx_print
                    is_ego_idx_print = current_env.node_feature_map_const.get('is_ego', -1)

                    # Create a reverse mapping from index to name for robust feature printing
                    idx_to_name_map = {v: k for k, v in current_env.node_feature_map_const.items()}
                    
                    num_nodes_to_print_detailed = min(graph_obs.num_nodes, 3)
                    for node_idx_print in range(num_nodes_to_print_detailed):
                        feature_values_str = []
                        is_this_node_ego = False
                        if graph_obs.x.shape[1] > 0: # Ensure there are features to iterate over
                            num_features = graph_obs.x.shape[1]
                                
                            for f_idx in range(num_features):
                                # Determine feature name robustly
                                feat_name = idx_to_name_map.get(f_idx, "UNKNOWN")
                                
                                # Handle features specific to memory/unified graphs
                                if feat_name == "UNKNOWN":
                                    if f_idx == MEM_NODE_FEAT_IDX.get('last_observed_step'):
                                        feat_name = 'last_observed_step'
                                    elif f_idx == MEM_NODE_FEAT_IDX.get('certainty'):
                                        feat_name = 'certainty'
                                    elif f_idx == MEM_NODE_FEAT_IDX.get('node_status'):
                                        feat_name = 'node_status'
                                    elif f_idx == num_features - 1 and num_features > NODE_FEATURE_DIM:
                                        feat_name = 'cluster_count'
                                
                                val = graph_obs.x[node_idx_print, f_idx].item()
                                feature_values_str.append(f"{feat_name}={val:.2f}")
                                if f_idx == is_ego_idx_print and val > 0.5: # Now is_ego_idx_print is defined
                                    is_this_node_ego = True
                            print(f"      Node {node_idx_print} (Ego: {is_this_node_ego}): {', '.join(feature_values_str)}")
                        else:
                             print(f"      Node {node_idx_print}: No features to display (feature dimension is 0).")


                    if is_ego_idx_print != -1 and graph_obs.x.numel() > 0 and is_ego_idx_print < graph_obs.x.shape[1]:
                        ego_sum = graph_obs.x[:, is_ego_idx_print].sum().item()
                        print(f"      Sum of 'is_ego' column (idx {is_ego_idx_print}) in this graph: {ego_sum:.1f} (should be 1.0 if ego is present and unique)")
                    
                    if graph_obs.x.numel() > 0 and (torch.isnan(graph_obs.x).any() or torch.isinf(graph_obs.x).any()):
                        print(f"      WARNING: Graph 'x' contains NaN/Inf!")

                elif graph_obs.num_nodes == 0:
                    print("    Node Features (x): Empty (0 nodes)")
                else:
                    print(f"    Node Features (x): None or Missing, Type: {type(getattr(graph_obs, 'x', None))}")

                if hasattr(graph_obs, 'edge_index') and graph_obs.edge_index is not None and graph_obs.num_edges > 0:
                    print(f"    Edge Index (edge_index) - Shape: {graph_obs.edge_index.shape}, Dtype: {graph_obs.edge_index.dtype}, Device: {graph_obs.edge_index.device}")
                    print(f"      First 5 Edges (if available):\n{graph_obs.edge_index[:, :min(5, graph_obs.num_edges)].cpu().numpy()}")
                elif graph_obs.num_edges == 0 :
                    print("    Edge Index (edge_index): Empty (0 edges)")
                else:
                    print(f"    Edge Index (edge_index): None or Missing, Type: {type(getattr(graph_obs, 'edge_index', None))}")
                
                if hasattr(graph_obs, 'pos') and graph_obs.pos is not None and graph_obs.num_nodes > 0:
                    print(f"    Node Positions (pos) - Shape: {graph_obs.pos.shape}, Dtype: {graph_obs.pos.dtype}, Device: {graph_obs.pos.device}")
                    if graph_obs.pos.numel() > 0:
                        print(f"      First Node Position: {np.round(graph_obs.pos[0].cpu().numpy(), 1)}")
                    else:
                        print(f"      Node Positions (pos): Tensor is empty but num_nodes > 0 (unexpected).")

                elif graph_obs.num_nodes == 0:
                     print("    Node Positions (pos): Empty (0 nodes)")
                else:
                    print(f"    Node Positions (pos): None or Missing, Type: {type(getattr(graph_obs, 'pos', None))}")
            else:
                print(f"    Type: {type(graph_obs)}, Value: {graph_obs}")

            # 4. Memory Map Observation
            mem_map_obs = obs_item.get('memory_map')
            print("\n  4. Memory Map Observation ('memory_map'):")
            if mem_map_obs is not None and isinstance(mem_map_obs, torch.Tensor):
                print(f"    Shape: {mem_map_obs.shape}, Dtype: {mem_map_obs.dtype}, Device: {mem_map_obs.device}")
                print(f"    Min: {mem_map_obs.min().item():.3f}, Max: {mem_map_obs.max().item():.3f}, Mean: {mem_map_obs.mean().item():.3f}, Sum: {mem_map_obs.sum().item():.3f}")
                if torch.isnan(mem_map_obs).any() or torch.isinf(mem_map_obs).any():
                    print(f"    WARNING: 'memory_map' contains NaN/Inf!")
            else:
                print(f"    Type: {type(mem_map_obs)}, Value: {mem_map_obs}")
                
            print("------------------------------------------")

        # --- Helper function to print MEMORY observation details ---
        # THIS FUNCTION IS NOW OBSOLETE as memory is part of the unified graph observation.
        # It is kept here for reference but is no longer called.
        def print_detailed_memory_observation(current_env, mem_obs_item, agent_id_str, step_str):
            print(f"\n--- Agent {agent_id_str} MEMORY Observation ({step_str}) ---")
            if not isinstance(mem_obs_item, dict):
                print(f"  ERROR: Memory observation is not a dict, type: {type(mem_obs_item)}")
                return

            # 1. Memory Map Observation
            mem_map_obs = mem_obs_item.get('memory_map')
            print("\n  1. Memory Map ('memory_map'):")
            if mem_map_obs is not None and isinstance(mem_map_obs, torch.Tensor):
                print(f"    Shape: {mem_map_obs.shape}, Dtype: {mem_map_obs.dtype}, Device: {mem_map_obs.device}")
                print(f"    Min: {mem_map_obs.min().item():.3f}, Max: {mem_map_obs.max().item():.3f}, Mean: {mem_map_obs.mean().item():.3f}, Sum: {mem_map_obs.sum().item():.3f}")
                if mem_map_obs.sum().item() == 0 and mem_map_obs.numel() > 0:
                    print(f"    INFO: 'memory_map' is all zeros.")
            else:
                print(f"    Type: {type(mem_map_obs)}, Value: {mem_map_obs}")

            # 2. Memory Graph Observation
            mem_graph_obs = mem_obs_item.get('memory_graph')
            print("\n  2. Memory Graph ('memory_graph'):")
            if mem_graph_obs is not None and isinstance(mem_graph_obs, Data):
                print(f"    Type: {type(mem_graph_obs)}")
                print(f"    Num Nodes: {mem_graph_obs.num_nodes}")
                print(f"    Num Edges: {mem_graph_obs.num_edges}")

                if hasattr(mem_graph_obs, 'x') and mem_graph_obs.x is not None and mem_graph_obs.num_nodes > 0:
                    print(f"    Node Features (x) - Shape: {mem_graph_obs.x.shape}, Dtype: {mem_graph_obs.x.dtype}, Device: {mem_graph_obs.x.device}")
                    if mem_graph_obs.x.numel() > 0:
                        print(f"      Min: {mem_graph_obs.x.min().item():.3f}, Max: {mem_graph_obs.x.max().item():.3f}, Mean: {mem_graph_obs.x.mean().item():.3f}")
                        
                        # Define feature names for memory graph, including the cluster count
                        mem_node_feature_map = MEM_NODE_FEAT_IDX.copy()
                        if mem_graph_obs.x.shape[1] > len(mem_node_feature_map):
                             mem_node_feature_map['cluster_count'] = mem_graph_obs.x.shape[1] - 1

                        num_nodes_to_print_detailed = min(mem_graph_obs.num_nodes, 3)
                        for node_idx_print in range(num_nodes_to_print_detailed):
                            feature_values_str = []
                            for f_idx in range(mem_graph_obs.x.shape[1]):
                                feat_name = "UNKNOWN"
                                for name, map_idx in mem_node_feature_map.items():
                                    if map_idx == f_idx:
                                        feat_name = name
                                        break
                                val = mem_graph_obs.x[node_idx_print, f_idx].item()
                                feature_values_str.append(f"{feat_name}={val:.2f}")
                            print(f"      Node {node_idx_print}: {', '.join(feature_values_str)}")

                if hasattr(mem_graph_obs, 'edge_index') and mem_graph_obs.edge_index is not None and mem_graph_obs.num_edges > 0:
                    print(f"    Edge Index (edge_index) - Shape: {mem_graph_obs.edge_index.shape}, Dtype: {mem_graph_obs.edge_index.dtype}, Device: {mem_graph_obs.edge_index.device}")
                else:
                    print(f"    Edge Index (edge_index): Empty or missing.")
            else:
                print(f"    Type: {type(mem_graph_obs)}, Value: {mem_graph_obs}")
            print("------------------------------------------")


        # --- 1. Inspect Observations from env.reset() ---
        obs_list_reset, info_reset = env.reset()
        
        # REMOVED: All memory updates and observation fusion are now handled
        # internally by the environment's ObservationManager. The obs_list_reset
        # already contains the complete observation, including the memory graph.

        print(f"\nReset complete. Number of observations from reset: {len(obs_list_reset)}")
        num_agents_to_inspect_reset = min(1, env.num_agents)
        for i_obs_r in range(num_agents_to_inspect_reset):
            agent_data_reset = env.agents[i_obs_r]
            agent_id_reset = agent_data_reset.get('id', f"EnvIdx_{i_obs_r}")
            # MODIFIED: Pass 'env' to the helper function
            print_detailed_observation(env, obs_list_reset[i_obs_r], f"ID {agent_id_reset} (EnvIdx {i_obs_r})", "from reset()")
            # REMOVED: The memory graph is now part of the unified 'graph' observation, so this separate print is obsolete.
            # print_detailed_memory_observation(env, obs_list_reset[i_obs_r], f"ID {agent_id_reset} (EnvIdx {i_obs_r})", "from reset()")


        # --- 2. Simulation Loop with Step Observation Inspection ---
        steps_to_inspect = [0, env.max_steps // 2, env.max_steps - 1]
        agents_indices_to_inspect_step = [0] 
        if env.num_agents > 1:
            agents_indices_to_inspect_step.append(min(1, env.num_agents -1 )) 
        if env.num_agents > env.num_agents_per_team: 
            agents_indices_to_inspect_step.append(min(env.num_agents_per_team, env.num_agents -1))
        agents_indices_to_inspect_step = sorted(list(set(agents_indices_to_inspect_step)))

        for step_idx in range(env.max_steps):
            actions = [{'movement': np.random.uniform(-1, 1, size=2),
                        'pickup': np.random.randint(0, 3)} for _ in range(env.num_agents)]

            obs_list_step, rewards_step, terminated_step, truncated_step, infos_step = env.step(actions)
            
            # All memory updates and observation fusion are now handled
            # internally by the environment's ObservationManager. The obs_list_step
            # already contains the complete observation.

            done_step = terminated_step or truncated_step

            if step_idx in steps_to_inspect:
                print(f"\n\n LLLLLLLLLLLLLLLLLLLLL Inspecting Observations from env.step() (at step {step_idx}) LLLLLLLLLLLLLLLLLLLLL ")
                for agent_env_idx_s in agents_indices_to_inspect_step:
                    if agent_env_idx_s < len(obs_list_step) and agent_env_idx_s < len(env.agents):
                        agent_data_step = env.agents[agent_env_idx_s]
                        agent_id_step = agent_data_step.get('id', f"EnvIdx_{agent_env_idx_s}")
                        alive_status = agent_data_step.get('alive', 'Unknown')
                        # MODIFIED: Pass 'env' to the helper function
                        print_detailed_observation(env, obs_list_step[agent_env_idx_s], f"ID {agent_id_step} (EnvIdx {agent_env_idx_s}, Alive: {alive_status})", f"from step {step_idx}")
                        # REMOVED: The memory graph is now part of the unified 'graph' observation, so this separate print is obsolete.
                        # print_detailed_memory_observation(env, obs_list_step[agent_env_idx_s], f"ID {agent_id_step} (EnvIdx {agent_env_idx_s}, Alive: {alive_status})", f"from step {step_idx}")
                    else:
                        print(f"  Skipping inspection for agent index {agent_env_idx_s} at step {step_idx} - index out of bounds.")
            
            if env.render_mode:
                env.render()
            
            if done_step:
                print(f"Episode finished at step {step_idx}. Terminated={terminated_step}, Truncated={truncated_step}")
                break
        
        if env.render_mode:
            print("\nSimulation finished. Close Pygame window or press ESC to exit.")
            running_pygame = True
            # Ensure pygame and time are imported if this block is to run stand-alone
            import pygame 
            import time
            while running_pygame:
                try:
                    for event_pygame in pygame.event.get(): 
                        if event_pygame.type == pygame.QUIT:
                            running_pygame = False
                        if event_pygame.type == pygame.KEYDOWN and event_pygame.key == pygame.K_ESCAPE:
                            running_pygame = False
                    if hasattr(env, 'render_mode') and env.render_mode: # Check if rendering still active
                        env.render()
                    if hasattr(env, 'clock') and env.clock: env.clock.tick(10) 
                    else: time.sleep(0.1) 
                except pygame.error: 
                    running_pygame = False
                except AttributeError: # If env or clock is None
                    running_pygame = False
        
    except Exception as main_e:
        print(f"\n--- ERROR in main execution block ---")
        traceback.print_exc()
    finally:
        if env is not None:
            # Import pygame and time locally if not already globally available for cleanup
            import pygame 
            import time
            env.close()
            print("Environment closed.")
    
    print("\n--- Script finished ---")
    
    # Close the log file and restore stdout
    sys.stdout = original_stdout
    f.close()

if __name__ == "__main__":
    main()