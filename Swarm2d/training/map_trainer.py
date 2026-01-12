import numpy as np
import torch
import torch.optim as optim
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# Set backend to avoid display issues
matplotlib.use('Agg')

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.constants import ChannelConstants
from Swarm2d.policies.simple_policies.map_policy import SimpleCNNMapPolicy, CompositePolicy, CNNPolicyWrapper
from Swarm2d.policies.heuristicPolicy.map_heuristic import MapHeuristic

class CombatHeuristic:
    def __init__(self, action_space):
        self.action_space = action_space
        self.state = {}
        
    def act(self, obs: dict) -> dict:
        self_obs = obs['self']
        if torch.is_tensor(self_obs):
            self_obs = self_obs.cpu().numpy()
            
        map_obs = obs['map']
        if torch.is_tensor(map_obs):
            map_obs = map_obs.cpu().numpy()
            
        # map_obs shape: (C, H, W)
        enemy_ch = map_obs[1] # Channel 1 is enemy
        
        move_x, move_y = 0.0, 0.0
        pickup = 0 # 0=None, 1=Grapple/Pickup, 2=Attack/Break
        
        # Find average position of enemies
        enemy_indices = np.argwhere(enemy_ch > 0)
        if len(enemy_indices) > 0:
            # Found enemies! Move towards average
            center = np.array([enemy_ch.shape[0]/2, enemy_ch.shape[1]/2])
            avg_enemy_pos = np.mean(enemy_indices, axis=0) # [y, x]
            
            # Vector from center (self) to enemy
            rel_y = avg_enemy_pos[0] - center[0]
            rel_x = avg_enemy_pos[1] - center[1]
            
            # Normalize
            mag = np.sqrt(rel_x**2 + rel_y**2) + 1e-6
            move_x = rel_x / mag
            move_y = rel_y / mag
            
            # If very close, GRAPPLE (1)
            if mag < 3.0: # Close in grid cells
                pickup = 1
        else:
            # No enemies? Patrol / Random
            curr_x, curr_y = self_obs[0], self_obs[1]
            target_x, target_y = 0.5, 0.5
            
            dir_x = target_x - curr_x
            dir_y = target_y - curr_y
            
            # Add noise
            dir_x += np.random.normal(0, 0.5)
            dir_y += np.random.normal(0, 0.5)
            
            mag = np.sqrt(dir_x**2 + dir_y**2) + 1e-6
            move_x = dir_x / mag
            move_y = dir_y / mag
            
        return {"movement": np.array([move_x, move_y]), "pickup": pickup}

class MapRLTrainer:
    def __init__(self, num_episodes=500, max_steps=200, scenario='resource', opponent_checkpoint=None, team2_checkpoint=None):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.scenario = scenario
        self.opponent_checkpoint = opponent_checkpoint
        self.team2_checkpoint = team2_checkpoint
        
        # Base config
        self.env_config = {
            'num_teams': 6 if team2_checkpoint is None else 3,
            'num_agents_per_team': 5, 
            'max_steps': max_steps,
            'render_mode': "headless",
            'debug': True, # ENABLED for Physics Debug Print
            'use_gpu_occlusion_in_env': False,
            'use_pybullet_raycasting': False,
            'generate_memory_graph': False,  # DISABLE GRAPHS - huge speedup!
            'generate_memory_map': False,    # DISABLE MEMORY MAP - Simplify inputs for faster learning!
            'width': 500, # SMALLER MAP
            'height': 500,
        }
        
        # SCENARIO SPECIFIC CONFIGURATION
        if scenario == 'resource':
            print("CONFIGURING FOR RESOURCE GATHERING PROOF...")
            self.env_config.update({
                'num_resources': 120,     # DENSITY ADJUSTED (Was 100 for 500x500)
                'num_obstacles': 3,
                'num_agents_per_team': 1,
                'width': 400,            # SMALLER MAP FOR FASTER TRAINING
                'height': 400,
                'hive_min_distance': 120.0,
                'resource_hive_buffer': 30.0, 
                'sensing_range_fraction': 0.08,
                'coop_resource_probability': 0.05, # DISABLE COOP: Simpler physics for learning
                
                # REWARDS: Focus on Delivery (Prevent Pickup Farming)
                'team_reward_overrides': {
                    '0': { 
                        'r_delivery': 200.0,
                        'r_progress_positive': 2.0,
                        'r_progress_negative': 1.0,
                        'r_progress': 0.0,            
                        'r_holding': 0.0,
                        'r_attachment': 0.0, 
                        'r_resource_found': 1.0,
                        'r_exploration_intrinsic': 0.5,
                        'r_enemy_found': 0.0,
                        'r_combat_win': 0.0,
                        'r_combat_lose': 0.0,
                        'r_combat_continuous': 0.0,
                        'r_grapple_control': 0.0,
                        'r_grapple_controlled': 0.0,
                        'r_teammate_lost_nearby': 0.0,  
                        'r_grapple_break': 0.0,
                        'r_torque_win': 0.0,
                        'r_hive_attack_continuous': 0.0,
                        'r_hive_capture': 0.0,
                        'r_hive_rebuild': 0.0,
                        'r_hive_destroyed_penalty': 0.0,
                        'r_hive_health_continuous': 0.0,
                        'r_death': 1.0,
                        'r_obstacle_found': 0.0,
                        'r_hive_found': 0.0,
                    }
                }
            })
        elif scenario == 'exploration':
            print("CONFIGURING FOR PURE EXPLORATION PROOF...")
            self.env_config.update({
                'num_resources': 100, 
                'num_obstacles': 15,
                'width': 600,
                'height': 600,
                'hive_min_distance': 100.0,
                'sensing_range_fraction': 0.15,
                
                'team_reward_overrides': {
                    '0': {
                        'r_delivery': 0.0,
                        'r_holding': 0.0,
                        'r_progress_positive': 0.0,
                        'r_progress_negative': 0.0,
                        'r_exploration_intrinsic': 1.0,
                        'r_enemy_found': 1.0,
                        'r_resource_found': 1.0,
                        'r_hive_found': 1.0,
                        'r_obstacle_found': 1.0,
                        'r_combat_win': 0.0,
                        'r_combat_lose': 0.0,
                        'r_combat_continuous': 0.0,
                        'r_grapple_control': 0.0,
                        'r_grapple_break': 0.0,
                        'r_torque_win': 0.0,
                        'r_hive_attack_continuous': 0.0,
                        'r_hive_capture': 0.0,
                        'r_hive_rebuild': 0.0,
                        'r_hive_destroyed_penalty': 0.0,
                        'r_death': 0.1,
                    }
                }
            })

        elif scenario == 'unified':
            print("CONFIGURING FOR UNIFIED TRAINING (ALL MECHANICS)...")
            self.env_config.update({
                'num_resources': 120,
                'num_obstacles': 15,
                'num_agents_per_team': 3,
                'width': 500,
                'height': 500,
                'hive_min_distance': 100.0,
                'sensing_range_fraction': 0.09,

                'team_reward_overrides': {
                    '0': {
                        'r_delivery': 200.0,
                        'r_holding': 10.0,
                        'r_progress_positive': 0.5,
                        'r_progress_negative': 0.5,
                        'r_resource_found': 5.0,
                        'r_attachment': 2.0,
                        'r_combat_win': 80.0,
                        'r_combat_lose': 0.0,
                        'r_combat_continuous': 0.5,
                        'r_enemy_found': 2.0,
                        'r_grapple_control': 2.0,
                        'r_grapple_break': 5.0,
                        'r_torque_win': 5.0,
                        'r_exploration_intrinsic': 0.5,
                        'r_hive_found': 5.0,
                        'r_obstacle_found': 1.0,
                        'r_death': 10.0,
                    }
                }
            })
            
        elif scenario == 'combat':
            print("CONFIGURING FOR COMBAT PROOF...")
            
            target_sensing_fraction = 0.08
            self.env_config['raw_ch_count'] = 8

            # Load base randomization from environment config
            try:
                from Swarm2d.gui_config import get_default_config
                default_config = get_default_config()
                
                agent_rand_factors = {}
                agent_attrs = default_config.get('Constants', {}).get('Agent Attributes', {})
                for key, data in agent_attrs.items():
                    if isinstance(data, dict) and 'value' in data:
                        val = data['value']
                        if isinstance(val, dict) and 'base' in val and 'rand' in val:
                            agent_rand_factors[key] = val
                
                obs_settings = default_config.get('Observations', {}).get('Observation Settings', {})
                if 'sensing_range_fraction' in obs_settings:
                    val = obs_settings['sensing_range_fraction']['value']
                    if isinstance(val, dict) and 'base' in val and 'rand' in val:
                        print(f"[Override] Updating randomization base for sensing_range_fraction from {val.get('base')} to {target_sensing_fraction}")
                        val['base'] = target_sensing_fraction
                        agent_rand_factors['sensing_range_fraction'] = val
                        
                print(f"Loaded Environment Randomization: {list(agent_rand_factors.keys())}")
                
            except ImportError as e:
                print(f"WARNING: Could not load default config ({e}). Using empty randomization.")
                agent_rand_factors = {}

            self.env_config.update({
                'num_resources': 220,     
                'num_obstacles': 15,     
                'num_agents_per_team': 8,
                'width': 1000,           
                'height': 1000,
                'hive_min_distance': 100.0,
                'hive_spawn_radius_factor': 0.35,
                'agent_spawn_radius': 150.0,
                'agent_randomization_factors': agent_rand_factors,
                'sensing_range_fraction': target_sensing_fraction,
                'raw_ch_count': 8,
                'team_reward_overrides': {
                    '0': { 
                        'r_delivery': 0.0,
                        'r_pickup': 0.0,
                        'r_holding': 0.0,
                        'r_progress_positive': 0.0,
                        'r_progress_negative': 0.0,
                        'r_resource_found': 0.0,
                        'r_exploration_intrinsic': 0.0,
                        'r_attachment': 0.0,
                        'r_combat_win': 50.0,
                        'r_combat_lose': 1.0,     
                        'r_combat_continuous': 1.5,
                        'r_enemy_found': 1.0,
                        'r_hive_attack_continuous': 5.0, 
                        'r_hive_capture': 10.0,       
                        'r_grapple_control': 3.5,
                        'r_grapple_controlled': 4.0,
                        'r_grapple_break': 5.0,
                        'r_torque_win': 15.0,
                        'r_teammate_lost_nearby': 5.0,      
                        'r_death': 10.0,
                    }
                }
            })
        
        # Create environment
        print(f"[Trainer] Initializing environment with config: {json.dumps({k: str(v) for k, v in self.env_config.items() if 'num' in k}, indent=2)}")
        self.env = Swarm2DEnv(**self.env_config)
        
        # VERIFY RESOURCE COUNT
        if self.env.num_resources_config != 69 and scenario == 'resource':
             print(f"[Trainer] WARNING: num_resources mismatch! Force updating to 69.")
             self.env.num_resources_config = 69

        self.team_0_size = self.env_config['num_agents_per_team']
        self.team_1_size = self.env_config['num_agents_per_team']
        self.team_2_size = self.env_config['num_agents_per_team'] if self.env_config['num_teams'] >= 3 else 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get dimensions
        sample_obs, _ = self.env.reset()
        actual_map_shape = sample_obs[0]['map'].shape
        actual_memory_shape = sample_obs[0]['memory_map'].shape
        self.self_dim = sample_obs[0]['self'].shape[0]
        
        self.map_channels, self.map_size, _ = actual_map_shape
        self.memory_channels, self.memory_size, _ = actual_memory_shape
        
        print(f"FAST Multi-Agent RL Setup (CNN Policy + Parameter Sharing):")
        print(f"  - Map Shape: {actual_map_shape}")
        print(f"  - Memory Shape: {actual_memory_shape}")
        print(f"  - Self Dim: {self.self_dim}")
        print(f"  - Device: {self.device}")
        
        # Create SHARED RL policy for Team 0
        self.rl_policy = SimpleCNNMapPolicy(
                map_channels=self.map_channels, 
                map_size=self.map_size,
                memory_channels=self.memory_channels,
            memory_size=self.memory_size,
            self_dim=self.self_dim
            ).to(self.device) 
        
        # Optimize CUDNN
        torch.backends.cudnn.benchmark = True
        
        self.optimizer = optim.Adam(self.rl_policy.parameters(), lr=3e-4)
        
        # Heuristic for opponents
        if scenario == 'combat':
            # --- Team 1 (Opponent 1) ---
            if self.opponent_checkpoint:
                print(f"Loading OPPONENT (Team 1) from checkpoint: {self.opponent_checkpoint}")
                opp_policy = SimpleCNNMapPolicy(
                    map_channels=self.map_channels, 
                    map_size=self.map_size,
                    memory_channels=self.memory_channels,
                    memory_size=self.memory_size,
                    self_dim=self.self_dim
                ).to(self.device)
                
                if os.path.exists(self.opponent_checkpoint):
                    ckpt = torch.load(self.opponent_checkpoint, map_location=self.device, weights_only=False)
                    opp_policy.load_state_dict(ckpt['policy'])
                    print(f"  [OK] Team 1 loaded (Episode {ckpt.get('episode', '?')})")
                    self.heuristic_policy = CNNPolicyWrapper(opp_policy)
                else:
                    print(f"  [ERROR] Opponent checkpoint not found! Falling back to Heuristic.")
                    self.heuristic_policy = CombatHeuristic(self.env.action_space)
            else:
                print("Using COMBAT Heuristic for Team 1")
                self.heuristic_policy = CombatHeuristic(self.env.action_space)

            # --- Team 2 (Opponent 2) ---
            self.team2_policy = None
            if self.team_2_size > 0:
                print(f"Configuring Team 2 (3rd Team) with checkpoint: {self.team2_checkpoint}")
                if self.team2_checkpoint:
                    t2_policy = SimpleCNNMapPolicy(
                        map_channels=self.map_channels, 
                        map_size=self.map_size,
                        memory_channels=self.memory_channels,
                        memory_size=self.memory_size,
                        self_dim=self.self_dim
                    ).to(self.device)
                    
                    if os.path.exists(self.team2_checkpoint):
                        ckpt = torch.load(self.team2_checkpoint, map_location=self.device, weights_only=False)
                        t2_policy.load_state_dict(ckpt['policy'])
                        print(f"  [OK] Team 2 loaded (Episode {ckpt.get('episode', '?')})")
                        self.team2_policy = CNNPolicyWrapper(t2_policy)
                    else:
                        print(f"  [ERROR] Team 2 checkpoint not found! Falling back to Heuristic.")
                        self.team2_policy = self.heuristic_policy # Re-use heuristic
                else:
                     print(f"  [WARNING] No checkpoint provided for Team 2. Using Heuristic.")
                     self.team2_policy = self.heuristic_policy
        elif scenario == 'composite':
            print("CONFIGURING COMPOSITE POLICY (Resource + Combat Checkpoints)...")
            
            res_policy = SimpleCNNMapPolicy(self.map_channels, self.map_size, self.memory_channels, self.memory_size, self.self_dim).to(self.device)
            res_ckpt = 'rl_training_results_cnn_resource_fast/checkpoint_latest.pt' 
            if os.path.exists(res_ckpt):
                print(f"Loading Resource Expert from {res_ckpt}")
                res_policy.load_state_dict(torch.load(res_ckpt, map_location=self.device)['policy'])
            else:
                print("WARNING: Resource checkpoint not found for composite!")

            com_policy = SimpleCNNMapPolicy(self.map_channels, self.map_size, self.memory_channels, self.memory_size, self.self_dim).to(self.device)
            com_ckpt = 'rl_training_results_cnn_combat_fast_v2/checkpoint_latest.pt' 
            if os.path.exists(com_ckpt):
                print(f"Loading Combat Expert from {com_ckpt}")
                com_policy.load_state_dict(torch.load(com_ckpt, map_location=self.device)['policy'])
            else:
                print("WARNING: Combat checkpoint not found for composite!")
                
            self.rl_policy = CompositePolicy(res_policy, com_policy)
            for p in self.rl_policy.parameters():
                p.requires_grad = False
                
            self.heuristic_policy = MapHeuristic(self.env.action_space)
            
        else:
            print("Using RESOURCE Heuristic for Team 1")
            self.heuristic_policy = MapHeuristic(self.env.action_space)
        
        # Metrics
        self.episode_rewards = []
        self.episode_resources = []
        self.start_episode = 0
        self.output_dir = f'rl_training_results_cnn_{scenario}_fast_v5'
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, 'training_log.csv')
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("episode,reward,resources,loss,time,pickups,deliveries,rew_delivery,rew_explore,rew_combat,rew_pickup,rew_prog_pos,rew_prog_neg,act_move_x,act_move_y,act_pickup_0,act_pickup_1,act_pickup_2,res_disp,res_moved,kills,deaths,damage_dealt,grapples_won\n")
    
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pt')
            if not os.path.exists(checkpoint_path) and '_v4' in self.output_dir:
                 v3_path = checkpoint_path.replace('_v4', '_v3')
                 if os.path.exists(v3_path):
                     print(f"[AUTO-MIGRATE] Checkpoint not found in v4, loading from v3: {v3_path}")
                     checkpoint_path = v3_path
        
        if not os.path.exists(checkpoint_path):
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        try:
            saved_self_dim = checkpoint['policy']['self_fc.0.weight'].shape[1]
            if saved_self_dim != self.self_dim:
                print(f"[WARNING] Dimension mismatch! Checkpoint expects {saved_self_dim} self-features, but Env provides {self.self_dim}.")
                if saved_self_dim < self.self_dim:
                    print(f"          [GRAFTING] Expanding Policy Input from {saved_self_dim} to {self.self_dim}...")
                    state_dict = checkpoint['policy']
                    compatible_state = {k: v for k, v in state_dict.items() if k != 'self_fc.0.weight'}
                    self.rl_policy.load_state_dict(compatible_state, strict=False)
                    with torch.no_grad():
                        saved_w = state_dict['self_fc.0.weight']
                        new_w = self.rl_policy.self_fc[0].weight
                        new_w[:, :saved_self_dim] = saved_w
                        new_w[:, saved_self_dim:] = 0.0
                    print(f"          [OK] Weights grafted. New features (Cols {saved_self_dim}-{self.self_dim}) init to 0.0.")
                    self.optimizer = optim.Adam(self.rl_policy.parameters(), lr=3e-4)
                    print(f"          [RESET] Optimizer reset due to architecture change.")
                    self.start_episode = checkpoint['episode']
                    self.episode_rewards = checkpoint['episode_rewards']
                    self.episode_resources = checkpoint['episode_resources']
                    return True
                else:
                    print(f"          [LEGACY MODE] Re-initializing Policy to match CHECKPOINT (legacy mode).")
                    self.rl_policy = SimpleCNNMapPolicy(
                        map_channels=self.map_channels, 
                        map_size=self.map_size,
                        memory_channels=self.memory_channels,
                        memory_size=self.memory_size,
                        self_dim=saved_self_dim 
                    ).to(self.device)
                    self.optimizer = optim.Adam(self.rl_policy.parameters(), lr=3e-4)
                    self.rl_policy.load_state_dict(checkpoint['policy'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.start_episode = checkpoint['episode']
                    self.episode_rewards = checkpoint['episode_rewards']
                    self.episode_resources = checkpoint['episode_resources']
                    return True
        except KeyError:
            pass 
            
        self.rl_policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_episode = checkpoint['episode']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_resources = checkpoint['episode_resources']
        
        with torch.no_grad():
             self.rl_policy.log_std.fill_(0.0)
        
        print(f"[OK] Resumed from episode {self.start_episode}")
        return True
        
    def collect_episode(self, episode_idx=0):
        obs_list, _ = self.env.reset()
        
        if episode_idx == 0 or episode_idx % 10 == 0:
             print(f"[DEBUG] Episode {episode_idx} Start:")
        
        trajectories = [{'log_probs': [], 'rewards': [], 'entropies': []} for _ in range(self.team_0_size)]
        
        episode_reward = 0
        resources_collected = 0
        
        metrics = {
            'pickups_attempted': 0,
            'deliveries': 0,
            'rew_delivery': 0.0,
            'rew_explore': 0.0,
            'rew_combat': 0.0,
            'rew_pickup': 0.0,
            'rew_holding': 0.0,             
            'rew_progress': 0.0,            
            'rew_progress_positive': 0.0,   
            'rew_progress_negative': 0.0,   
            'rew_details': {}, 
            'action_stats': {
                'move_x': [],
                'move_y': [],
                'pickup_dist': [0, 0, 0] 
            },
            'res_displacement': 0.0,
            'res_moved_count': 0,
            'total_holding_steps': 0, 
            'max_holding_streak': 0,   
            'kills': 0,
            'deaths': 0,
            'damage_dealt': 0.0,
            'grapples_won': 0,
        }
        
        agent_holding_streak = {i: 0 for i in range(self.team_0_size)}
        agent_combat_hits = {i: 0 for i in range(self.team_0_size)}

        initial_res_pos = {}
        for r in self.env.resources:
            if r.get('pos') is not None:
                initial_res_pos[r['id']] = r['pos'].copy()
        
        combat_log_limit = 10 
        combat_log_count = 0
        first_contact_reported = False

        for step in range(self.max_steps):
            actions = []
            
            # Team 0: Shared RL Policy
            for i in range(self.team_0_size):
                obs = obs_list[i]
                move, pickup, log_prob, entropy = self.rl_policy.act(obs['map'], obs['memory_map'], obs['self'])
                actions.append({'movement': move, 'pickup': pickup})
                trajectories[i]['log_probs'].append(log_prob)
                trajectories[i]['entropies'].append(entropy)
                
                try:
                    is_carrying_val = obs['self'][8]
                    if hasattr(is_carrying_val, 'item'):
                        is_carrying = is_carrying_val.item() > 0.5
                    else:
                        is_carrying = is_carrying_val > 0.5
                except:
                    is_carrying = False

                if is_carrying:
                    metrics['total_holding_steps'] += 1
                    agent_holding_streak[i] += 1
                else:
                    agent_holding_streak[i] = 0
                
                if agent_holding_streak[i] > metrics['max_holding_streak']:
                    metrics['max_holding_streak'] = agent_holding_streak[i]

                if pickup == 1:
                    metrics['pickups_attempted'] += 1
                
                metrics['action_stats']['move_x'].append(move[0])
                metrics['action_stats']['move_y'].append(move[1])
                metrics['action_stats']['pickup_dist'][pickup] += 1
            
            # Team 1
            start_idx_t1 = self.team_0_size
            end_idx_t1 = start_idx_t1 + self.team_1_size
            for i in range(start_idx_t1, end_idx_t1):
                obs = obs_list[i]
                actions.append(self.heuristic_policy.act(obs))
            
            # Team 2
            if self.team_2_size > 0:
                start_idx_t2 = end_idx_t1
                end_idx_t2 = start_idx_t2 + self.team_2_size
                for i in range(start_idx_t2, end_idx_t2):
                     obs = obs_list[i]
                     if self.team2_policy:
                         actions.append(self.team2_policy.act(obs))
                     else:
                         actions.append(self.heuristic_policy.act(obs))
            
            obs_list, rewards, terminated, truncated, infos = self.env.step(actions)

            # --- COMBAT EVENT LOGGING ---
            if not first_contact_reported:
                t0_pos = [self.env.agents[i]['pos'] for i in range(self.team_0_size) if self.env.agents[i]['alive']]
                t1_pos = [self.env.agents[i]['pos'] for i in range(self.team_0_size, self.env.num_agents) if self.env.agents[i]['alive']]
                
                if t0_pos and t1_pos:
                    min_dist = float('inf')
                    for p0 in t0_pos:
                        for p1 in t1_pos:
                            d = np.linalg.norm(p0 - p1)
                            if d < min_dist: min_dist = d
                    
                    if min_dist < 50.0:
                        print(f"    [!] FIRST CONTACT at Step {step} (Dist: {min_dist:.1f})")
                        first_contact_reported = True

            if combat_log_count < combat_log_limit:
                for i in range(self.team_0_size):
                    r_dict = rewards[i]
                    if r_dict.get('r_combat_continuous', 0) > 0:
                        agent_combat_hits[i] += 1

                    if r_dict.get('r_combat_win', 0) > 0:
                        hits = agent_combat_hits[i]
                        print(f"    [COMBAT] Step {step}: Agent {i} (Team 0) KILLED ENEMY! (Rew: {r_dict['r_combat_win']:.2f}) - After {hits} contact steps.")
                        combat_log_count += 1
                        agent_combat_hits[i] = 0
                        
                    if r_dict.get('r_grapple_control', 0) > 0:
                        if step % 10 == 0: 
                             print(f"    [COMBAT] Step {step}: Agent {i} is GRAPPLING an enemy.")
                             combat_log_count += 1
                             
                    if r_dict.get('r_grapple_break', 0) > 0:
                        print(f"    [COMBAT] Step {step}: Agent {i} BROKE a grapple!")
                        combat_log_count += 1

            if 'kills_by_team' in infos:
                metrics['kills'] += infos['kills_by_team'][0] 
            if 'deaths_by_team' in infos:
                metrics['deaths'] += infos['deaths_by_team'][0] 
            if 'damage_by_team' in infos:
                metrics['damage_dealt'] += infos['damage_by_team'][0]
            if 'grapples_broken_by_team' in infos:
                metrics['grapples_won'] += infos['grapples_broken_by_team'][0]
                
            if (step + 1) % 50 == 0:
                step_rew = sum([sum(rewards[i].values()) for i in range(self.team_0_size)])
                print(f"    Step {step+1}/{self.max_steps} | Current Ep Resources: {resources_collected} | Team Step Reward: {step_rew:.1f}")

            step_team_0_delivery_reward = 0.0
            
            for i in range(self.team_0_size):
                r_dict = rewards[i]
                total_r = sum(r_dict.values())

                try:
                    agent_pos = self.env.agents[i]['pos']
                    w, h = self.env_config.get('width', 1000), self.env_config.get('height', 1000)
                    margin = 50.0 
                    if agent_pos[0] < margin or agent_pos[0] > w - margin or \
                       agent_pos[1] < margin or agent_pos[1] > h - margin:
                        wall_pen = -5.0 
                        total_r += wall_pen
                        metrics['rew_progress_negative'] += wall_pen

                    center_x, center_y = w/2, h/2
                    dist_sq = (agent_pos[0] - center_x)**2 + (agent_pos[1] - center_y)**2
                    center_pen = -1.0 * (dist_sq / 80000.0) 
                    total_r += center_pen
                    metrics['rew_explore'] += center_pen 

                except: pass
                
                trajectories[i]['rewards'].append(total_r)
                episode_reward += total_r
                
                if 'r_delivery' in r_dict: 
                    metrics['rew_delivery'] += r_dict['r_delivery']
                    step_team_0_delivery_reward += r_dict['r_delivery'] 
                    
                if 'r_progress' in r_dict: 
                    val = r_dict['r_progress']
                    metrics['rew_progress'] += val
                    if val > 0: metrics['rew_progress_positive'] += val
                    else: metrics['rew_progress_negative'] += val
                
                if 'r_progress_positive' in r_dict:
                    metrics['rew_progress_positive'] += r_dict['r_progress_positive']
                    metrics['rew_progress'] += r_dict['r_progress_positive']
                    
                if 'r_progress_negative' in r_dict:
                    metrics['rew_progress_negative'] += r_dict['r_progress_negative']
                    metrics['rew_progress'] += r_dict['r_progress_negative']
                    
                if 'r_resource_found' in r_dict: metrics['rew_explore'] += r_dict['r_resource_found']
                if 'r_exploration_intrinsic' in r_dict: metrics['rew_explore'] += r_dict['r_exploration_intrinsic']
                if 'r_combat_win' in r_dict: metrics['rew_combat'] += r_dict['r_combat_win']
                if 'r_attachment' in r_dict: metrics['rew_pickup'] += r_dict['r_attachment']
                if 'r_holding' in r_dict: metrics['rew_holding'] += r_dict['r_holding']
                
                for k, v in r_dict.items():
                    metrics['rew_details'][k] = metrics['rew_details'].get(k, 0.0) + v
            
            if step_team_0_delivery_reward > 0:
                resources_collected += 1
                metrics['deliveries'] += 1
                print(f"    [+] Team 0 Delivered! Total: {resources_collected} (Rew: {step_team_0_delivery_reward:.1f})")
            
            if terminated or truncated:
                break
        
        for r in self.env.resources:
            if r.get('pos') is not None and r['id'] in initial_res_pos:
                dist = np.linalg.norm(r['pos'] - initial_res_pos[r['id']])
                metrics['res_displacement'] += dist
                if dist > 5.0:
                    metrics['res_moved_count'] += 1
        
        return trajectories, episode_reward, resources_collected, metrics

    def train_episode(self, episode_idx=0):
        trajectories, episode_reward, resources, metrics = self.collect_episode(episode_idx)
        
        all_returns = []
        all_log_probs = []
        all_entropies = []
        
        for trajectory in trajectories:
            if not trajectory['rewards']: continue
            returns = []
            G = 0
            for r in reversed(trajectory['rewards']):
                G = r + 0.99 * G
                returns.insert(0, G)
            all_returns.extend(returns)
            all_log_probs.extend(trajectory['log_probs'])
            all_entropies.extend(trajectory['entropies'] if trajectory['entropies'] else [])

        all_returns = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        if all_returns.std() > 0:
            all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)
            
        policy_loss = []
        for log_prob, G, entropy in zip(all_log_probs, all_returns, all_entropies):
            entropy_bonus = 0.1 * entropy
            policy_loss.append(-log_prob * G - entropy_bonus)
            
        loss_val = 0.0
        if policy_loss:
            loss = torch.stack(policy_loss).mean()
            loss_val = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rl_policy.parameters(), 1.0)
            self.optimizer.step()
            
        return episode_reward, resources, loss_val, metrics
    
    def train(self):
        print("\nSTARTING TRAINING...")
        start_time = time.time()
        
        try:
            for episode in range(self.start_episode, self.num_episodes):
                ep_start = time.time()
                reward, resources, loss, metrics = self.train_episode(episode)
                duration = time.time() - ep_start
                
                self.episode_rewards.append(reward)
                self.episode_resources.append(resources)
                
                avg_move_x = np.mean(metrics['action_stats']['move_x'])
                avg_move_y = np.mean(metrics['action_stats']['move_y'])
                p_counts = metrics['action_stats']['pickup_dist']
                
                with open(self.log_file, 'a') as f:
                    f.write(f"{episode+1},{reward:.2f},{resources},{loss:.4f},{duration:.2f},"
                            f"{metrics['pickups_attempted']},{metrics['deliveries']},"
                            f"{metrics['rew_delivery']:.2f},{metrics['rew_explore']:.2f},{metrics['rew_combat']:.2f},{metrics['rew_pickup']:.2f},"
                            f"{metrics['rew_progress_positive']:.2f},{metrics['rew_progress_negative']:.2f},"
                            f"{avg_move_x:.3f},{avg_move_y:.3f},{p_counts[0]},{p_counts[1]},{p_counts[2]},"
                            f"{metrics['res_displacement']:.1f},{metrics['res_moved_count']},"
                            f"{metrics['kills']},{metrics['deaths']},{metrics['damage_dealt']:.1f},{metrics['grapples_won']}\n")
                
                if (episode + 1) % 1 == 0:
                    avg_rew = np.mean(self.episode_rewards[-10:])
                    active_res_count = len([r for r in self.env.resources if not r.get('delivered')])
                    avg_hold = metrics['total_holding_steps'] / max(1, metrics['pickups_attempted'])
                    
                    print(f"Ep {episode+1}/{self.num_episodes} | Rew: {reward:.1f} (Avg: {avg_rew:.1f}) | Res: {resources} (Active: {active_res_count}) | "
                          f"Pickups: {metrics['pickups_attempted']} | Deliv_Rew: {metrics['rew_delivery']:.1f} | "
                          f"HoldRew: {metrics['rew_holding']:.1f} | HoldSteps: {metrics['total_holding_steps']} (Avg: {avg_hold:.1f}, MaxStreak: {metrics['max_holding_streak']}) | "
                          f"ProgRew: {metrics['rew_progress']:.1f} | "
                          f"Moved: {metrics['res_moved_count']} (Dist: {metrics['res_displacement']:.1f}) | Time: {duration:.1f}s\n"
                          f"    [COMBAT]: Kills: {metrics['kills']} | Deaths: {metrics['deaths']} | Dmg: {metrics['damage_dealt']:.1f} | GrapBreak: {metrics['grapples_won']}")
                    
                    if episode == 0 or (episode+1) % 2 == 0:
                        all_x = metrics['action_stats']['move_x']
                        all_y = metrics['action_stats']['move_y']
                        min_x, max_x = (min(all_x), max(all_x)) if all_x else (0,0)
                        min_y, max_y = (min(all_y), max(all_y)) if all_y else (0,0)
                        
                        print(f"    [DEBUG BREAKDOWN]:")
                        print(f"       Action Dist: None={p_counts[0]}, Pickup={p_counts[1]}, Drop={p_counts[2]}")
                        print(f"       Move Avg: x={avg_move_x:.2f}, y={avg_move_y:.2f} | Range X:[{min_x:.2f}, {max_x:.2f}] Y:[{min_y:.2f}, {max_y:.2f}]")
                        print(f"       Progress Rew: +{metrics['rew_progress_positive']:.1f} / {metrics['rew_progress_negative']:.1f}")
                        print(f"       Reward Details: {json.dumps({k: round(v, 1) for k, v in metrics['rew_details'].items()}, sort_keys=True)}")

                if (episode + 1) % 25 == 0:
                    self.save_checkpoint(episode + 1)
                    
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")
            self.save_checkpoint(episode + 1)
            print("[!] Saved INTERRUPTED checkpoint. Resume with --resume")
            return

        print("Training Complete!")

    def save_checkpoint(self, episode):
        path = os.path.join(self.output_dir, f'checkpoint_ep{episode}.pt')
        data = {
            'episode': episode,
            'policy': self.rl_policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_resources': self.episode_resources
        }
        torch.save(data, path)
        torch.save(data, os.path.join(self.output_dir, 'checkpoint_latest.pt'))
        print(f"  [SAVE] Saved checkpoint to {path}")
