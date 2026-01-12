"""
FAST CTDE Training (Actor-Critic) - MAP OBSERVATIONS ONLY
Uses PPO-style clipped updates with a Centralized Critic for better stability.
Faster convergence than REINFORCE.

ARCHITECTURE:
1. Actor (Decentralized): Uses local observations (Map + Self).
2. Critic (Centralized): Uses GLOBAL STATE (All Maps + All Agents).
   - During training: Critic guides the Actor.
   - During inference: Critic is discarded.

ALGORITHM: PPO (Proximal Policy Optimization)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import time
import json 
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.constants import ChannelConstants

# --- CONFIGURATION ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LR = 1e-4
MINIBATCH_SIZE = 128
PPO_EPOCHS = 4

class RunningMeanStd:
    """
    Standardizes data by tracking running mean and std.
    Standard practice in SOTA RL (StableBaselines3, OpenAI Baselines).
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def get_global_critic_obs(env, grid_size=64):
    """
    Generates a simplified 'God View' global map for the Centralized Critic.
    Resolution: grid_size x grid_size
    Channels: 
    0: Allies
    1: Enemies
    2: Resources
    3: Hives
    4: Obstacles
    """
    # Initialize grid
    grid = torch.zeros((5, grid_size, grid_size), dtype=torch.float32, device='cuda')
    
    # Scaling factors
    scale_x = grid_size / env.width
    scale_y = grid_size / env.height
    
    def to_grid(pos):
        x = int(pos[0] * scale_x)
        y = int(pos[1] * scale_y)
        return min(max(x, 0), grid_size-1), min(max(y, 0), grid_size-1)

    # 1. Agents (Allies vs Enemies - relative to Team 0)
    # We assume training Team 0. 
    for agent in env.agents:
        if not agent['alive']: continue
        gx, gy = to_grid(agent['pos'])
        if agent['team'] == 0:
            grid[0, gy, gx] = 1.0 # Ally
        else:
            grid[1, gy, gx] = 1.0 # Enemy
            
    # 2. Resources
    for res in env.resources:
        if res['delivered']: continue
        gx, gy = to_grid(res['pos'])
        grid[2, gy, gx] = 1.0
        
    # 3. Hives
    for hive in env.hives.values():
        gx, gy = to_grid(hive['pos'])
        grid[3, gy, gx] = 1.0
        
    # 4. Obstacles
    for obs in env.obstacles:
        gx, gy = to_grid(obs['pos'])
        grid[4, gy, gx] = 1.0
        
    return grid

class SimpleCNNActor(nn.Module):
    """
    Decentralized Actor (The Agent).
    Same architecture as the REINFORCE policy.
    """
    def __init__(self, map_channels, map_size, self_dim, hidden_dim=256):
        super(SimpleCNNActor, self).__init__()
        self.map_size = map_size
        
        # CNN for Map Processing
        # Input: (Batch, Map_Channels, 32, 32)
        self.conv1 = nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32x32 -> 16x16 -> 8x8
        self.flat_map_size = 64 * 8 * 8
        
        self.map_fc = nn.Sequential(
            nn.Linear(self.flat_map_size, 256),
            nn.ReLU()
        )
        
        self.self_fc = nn.Sequential(
            nn.Linear(self_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.move_head = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.pickup_head = nn.Linear(hidden_dim, 3)
        
    def forward(self, obs_map, self_vec):
        # Map
        x = F.relu(self.conv1(obs_map))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        
        map_features = self.map_fc(x)
        self_features = self.self_fc(self_vec)
        
        combined = torch.cat([map_features, self_features], dim=1)
        features = self.fc(combined)
        
        move_mean = torch.tanh(self.move_head(features))
        move_std = torch.exp(self.log_std).expand_as(move_mean)
        pickup_logits = self.pickup_head(features)
        
        return move_mean, move_std, pickup_logits

class CentralizedCritic(nn.Module):
    """
    Centralized Critic (The Coach).
    Takes GLOBAL GOD VIEW map to estimate V(s).
    Input: Global Map (5, 64, 64)
    """
    def __init__(self, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        
        # Global Map Channels: 5 (Ally, Enemy, Res, Hive, Obs)
        # Input: (Batch, 5, 64, 64)
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1) # -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> 8x8
        
        self.flat_size = 128 * 8 * 8
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Value
        )

    def forward(self, global_map):
        x = F.relu(self.conv1(global_map))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)

class PPOAgent:
    def __init__(self, map_channels, map_size, self_dim, lr=3e-4, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.actor = SimpleCNNActor(map_channels, map_size, self_dim).to(self.device)
        self.critic = CentralizedCritic().to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

    def get_action(self, obs_map, obs_self, global_state):
        with torch.no_grad():
            # Actor Inputs
            if obs_map.dim() == 3: obs_map = obs_map.unsqueeze(0)
            if obs_self.dim() == 1: obs_self = obs_self.unsqueeze(0)
            
            # Critic Input
            if global_state.dim() == 3: global_state = global_state.unsqueeze(0)
            
            obs_map = obs_map.to(self.device)
            obs_self = obs_self.to(self.device)
            global_state = global_state.to(self.device)
            
            # Actor Pass
            mean, std, logits = self.actor(obs_map, obs_self)
            
            # Critic Pass (Global)
            val = self.critic(global_state)
            
            move_dist = Normal(mean, std)
            move_action = move_dist.sample()
            move_log_prob = move_dist.log_prob(move_action).sum(dim=-1)
            
            pickup_dist = Categorical(logits=logits)
            pickup_action = pickup_dist.sample()
            pickup_log_prob = pickup_dist.log_prob(pickup_action)
            
            return (move_action.cpu().numpy()[0], pickup_action.item(), 
                    move_log_prob.cpu().item(), pickup_log_prob.cpu().item(), val.cpu().item())

    def update(self, rollouts):
        if not rollouts:
            return
            
        # rollouts: list of (obs_map, obs_self, global_state, act_move, act_pick, log_prob, ret, adv)
        # Convert to tensors
        obs_maps = torch.stack([r[0] for r in rollouts]).to(self.device)
        obs_selfs = torch.stack([r[1] for r in rollouts]).to(self.device)
        
        # Safe handling for global states
        gs_list = [r[2] for r in rollouts]
        if isinstance(gs_list[0], np.ndarray):
            global_states = torch.tensor(np.array(gs_list), dtype=torch.float32).to(self.device)
        else:
            global_states = torch.stack(gs_list).to(self.device)
        
        act_moves = torch.tensor(np.array([r[3] for r in rollouts]), dtype=torch.float32).to(self.device)
        act_picks = torch.tensor(np.array([r[4] for r in rollouts]), dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(np.array([r[5] for r in rollouts]), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array([r[6] for r in rollouts]), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array([r[7] for r in rollouts]), dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(rollouts)
        if dataset_size == 0: return
        
        batch_size = min(MINIBATCH_SIZE, dataset_size)
        indices = np.arange(dataset_size)
        
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                mb_maps = obs_maps[idx]
                mb_selfs = obs_selfs[idx]
                mb_glob = global_states[idx] # New
                
                mb_moves = act_moves[idx]
                mb_picks = act_picks[idx]
                mb_old_lp = old_log_probs[idx]
                mb_ret = returns[idx]
                mb_adv = advantages[idx]
                
                # New pass
                mean, std, logits = self.actor(mb_maps, mb_selfs)
                values = self.critic(mb_glob).squeeze() # Use Global State
                
                # Log Probs
                move_dist = Normal(mean, std)
                new_move_lp = move_dist.log_prob(mb_moves).sum(dim=-1)
                move_entropy = move_dist.entropy().sum(dim=-1)
                
                pickup_dist = Categorical(logits=logits)
                new_pick_lp = pickup_dist.log_prob(mb_picks)
                pick_entropy = pickup_dist.entropy()
                
                new_log_prob = new_move_lp + new_pick_lp
                entropy = move_entropy + pick_entropy
                
                # Ratio
                ratio = torch.exp(new_log_prob - mb_old_lp)
                
                # Surrogate Loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (Clipped)
                value_pred_clipped = mb_ret + (values - mb_ret).clamp(-CLIP_EPS, CLIP_EPS)
                value_loss_1 = (values - mb_ret).pow(2)
                value_loss_2 = (value_pred_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                # Total Loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

class CTDETrainer:
    def __init__(self, num_episodes=2500, scenario='resource', resume_path=None, randomize=False, run_name=None):
        self.num_episodes = num_episodes
        self.scenario = scenario
        self.randomize = randomize
        self.run_name = run_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- ENV CONFIG (Copied from simple_rl) ---
        self.env_config = {
            'num_teams': 2,
            'num_agents_per_team': 10,
            'max_steps': 200,
            'render_mode': "headless",
            'width': 500, 'height': 500,
            'generate_memory_graph': False,
            'generate_memory_map': False,
            'sensing_range_fraction': 0.10,
        }
        
        # Apply Scenario Settings
        if scenario == 'resource':
            self.env_config.update({'num_resources': 60, 'num_obstacles': 5})
        elif scenario == 'combat':
            self.env_config.update({'num_resources': 10, 'num_obstacles': 10})
        elif scenario == 'unified':
             self.env_config.update({
                'num_resources': 120, 'num_obstacles': 10,
                'team_reward_overrides': {
                    '0': {
                        # --- PRIMARY OBJECTIVES (Strong Signals) ---
                        'r_delivery': 200.0,       # Base 10 * 200 = 2000 (Delivery)
                        'r_holding': 5.0,          # Base 0.1 * 5 = 0.5 per step (Hold)
                        'r_progress_positive': 5.0,# Base 10 * 5 = 50 (Move to Hive)
                        'r_progress': 0.0,              # Disable generic progress (use pos/neg specific)
                        'r_progress_negative': 2.0,     # Base 10 * 2 = 20 (Penalty for moving away)
                        'r_attachment': 0.0,           # Redundant with r_holding
                        'coop_collection_bonus': 2.0,    # Active

                        # --- COMBAT/SURVIVAL ---
                        'r_combat_lose': 5.0,      # Base -25 * 5 = -125 (Lost fight)
                        'r_combat_continuous': 1.0,# Base 1 * 1 = 1 (Damage dealt - REDUCED to prioritize delivery)
                        'r_grapple_controlled': 2.0,     # Active
                        'r_grapple_control': 2.0,        # Base 0.5 * 2 = 1.0 (Maintain grapple)
                        'r_torque_win': 10.0,      # Base 0.5 * 10 = 5.0 (Win Grapple)
                        'r_grapple_break': 5.0,    # Base 0.5 * 5 = 2.5 (Escape)
                        'r_combat_win': 20.0,      # Base 25 * 20 = 500 (Kill - REDUCED from 5000 to make delivery main goal)
                        'r_teammate_lost_nearby': 0.0,   # Disabled (Too noisy)
     
                        # --- HIVE COMBAT
                        'r_hive_rebuild': 50.0,     # Base 25 * 50 = 1250 (Strategic Goal)
                        'r_hive_capture': 100.0,   # Base 50 * 100 = 5000 (Capture)
                        'r_hive_health_continuous': 0.0, # Passive
                        'r_hive_attack_continuous': 5.0, # Siege damage (Active)
                        'r_hive_destroyed_penalty': 5.0, # Active
                        
                        'r_death': 10.0,           # Base -50 * 10 = -500 (Death Penalty)
                        
                        # --- EXPLORATION ---
                        'r_exploration_intrinsic': 0.0, # Disable random exploration reward
                        'r_resource_found': 0.1,        # Reduce visual discovery noise
                        'r_enemy_found': 0.1,           # Reduce visual discovery noise
                        'r_obstacle_found': 0.0,        # Disable obstacle finding reward
                        'r_hive_found': 0.1,           # Minor visual ack
    
                    }
                }
            })
            
        self.env = Swarm2DEnv(**self.env_config)
        
        # Init Agent
        sample_obs, _ = self.env.reset()
        map_ch = sample_obs[0]['map'].shape[0]
        map_sz = sample_obs[0]['map'].shape[1]
        self_dim = sample_obs[0]['self'].shape[0]
        
        self.agent = PPOAgent(map_ch, map_sz, self_dim, device=self.device)
        
        # Resume if requested
        self.start_episode = 0
        if resume_path and os.path.exists(resume_path):
            print(f"LOADING CHECKPOINT: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            
            # Helper to load state dict with shape handling
            def load_mismatched(model, state_dict, prefix=""):
                model_state = model.state_dict()
                filtered_state = {}
                mismatch_found = False
                
                for k, v in state_dict.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state[k] = v
                        else:
                            print(f"SKIPPING {prefix}{k}: Shape mismatch {v.shape} vs {model_state[k].shape}")
                            mismatch_found = True
                    else:
                        print(f"SKIPPING {prefix}{k}: Not in current model")
                        mismatch_found = True
                        
                model.load_state_dict(filtered_state, strict=False)
                return mismatch_found

            mismatch_a = load_mismatched(self.agent.actor, checkpoint['actor_state_dict'], "Actor: ")
            mismatch_c = load_mismatched(self.agent.critic, checkpoint['critic_state_dict'], "Critic: ")
            
            # Optimizer: Only load if model matched perfectly, otherwise start fresh
            if mismatch_a or mismatch_c:
                print("Architecture mismatch detected. RESETTING OPTIMIZER to defaults.")
            else:
                try:
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("WARNING: Optimizer state mismatch. Resetting optimizer.")
            
            # Try to infer start episode from filename if possible, else 0
            try:
                import re
                match = re.search(r'ep(\d+)', resume_path)
                if match:
                    self.start_episode = int(match.group(1)) + 1
                    print(f"Resuming from episode {self.start_episode}")
            except:
                pass
        
        # Heuristic for Team 1
        from Swarm2d.policies.heuristicPolicy.map_heuristic import MapHeuristic
        self.heuristic = MapHeuristic(self.env.action_space)
        
        # Reward Normalizer (SOTA Practice)
        # We normalize RETURNS, not just immediate rewards, but for simplicity in this script
        # we will normalize the rewards seen by the critic update or the advantages.
        self.reward_scaler = RunningMeanStd(shape=())

    def compute_gae(self, rewards, values, next_val, dones):
        # Generalized Advantage Estimation
        # ... (This method is currently unused as we implemented GAE inline in train) ...
        pass

    def save_checkpoint(self, filename):
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }, filename)

    def train(self):
        print(f"STARTING CTDE PPO TRAINING ({self.scenario})...")
        
        # --- DEBUG: PRINT FULL REWARD CONFIGURATION ---
        if self.scenario == 'unified':
            print("\n=== UNIFIED SCENARIO REWARD CONFIGURATION (Base * Multiplier) ===")
            print(f"{'REWARD KEY':<30} | {'BASE':<8} | {'MULT':<8} | {'FINAL VALUE':<12}")
            print("-" * 65)
            
            # We need to access constants to get base values
            from Swarm2d.constants import REWARD_CONFIG
            
            # Get overrides for Team 0
            overrides = self.env_config['team_reward_overrides']['0']
            
            # Sort by final value magnitude (descending)
            sorted_rewards = []
            for key in overrides:
                base = REWARD_CONFIG.get(key, {}).get('default_value', 0.0)
                mult = overrides[key]
                final = base * mult
                sorted_rewards.append((key, base, mult, final))
            
            sorted_rewards.sort(key=lambda x: abs(x[3]), reverse=True)
            
            for key, base, mult, final in sorted_rewards:
                print(f"{key:<30} | {base:<8.1f} | {mult:<8.1f} | {final:<12.1f}")
            print("=================================================================\n")

        os.makedirs("ctde_checkpoints", exist_ok=True)
        
        for ep in range(self.start_episode, self.num_episodes):
            # Domain Randomization (if enabled)
            override_config = None
            if self.randomize:
                # Vary resources +/- 20%
                base_res = self.env_config.get('num_resources', 40)
                new_res = np.random.randint(int(base_res * 0.8), int(base_res * 1.2))
                
                # Vary obstacles +/- 20%
                base_obs = self.env_config.get('num_obstacles', 10)
                new_obs = np.random.randint(int(base_obs * 0.8), int(base_obs * 1.2))
                
                override_config = {'num_resources': new_res, 'num_obstacles': new_obs}
                # Note: We pass this to reset() if env supports it, or set directly
            
            # Since we couldn't modify env.reset() easily due to signature mismatch issues,
            # we will set the properties directly on the env instance before reset.
            if override_config:
                self.env.num_resources_config = override_config['num_resources']
                self.env.num_obstacles = override_config['num_obstacles']
            
            obs_list, _ = self.env.reset()
            
            # Storage for PPO
            batch_obs_map = []
            batch_obs_self = []
            batch_acts_mv = []
            batch_acts_pk = []
            batch_log_probs = []
            batch_vals = []
            batch_rews = []
            batch_dones = []
            
            # Metric Trackers
            ep_metrics = {
                "r_delivery": 0,
                "r_combat": 0, # Sum of win/lose/damage
                "r_survival": 0, # death/health
                "r_misc": 0,     # progress/holding/exploration
                "act_move_mag": [],
                "act_pick_counts": {0:0, 1:0, 2:0} # 0:None, 1:Pick, 2:Drop
            }
            
            ep_reward = 0
            
            # Keep track of actual steps performed
            steps_performed = 0
            
            # Get actual agent counts from env
            num_agents = len(self.env.agents)
            half_agents = num_agents // 2
            
            # --- GLOBAL STATE CONSTRUCTION ---
            # Construct global map for the critic (Initialize before loop)
            global_critic_map = get_global_critic_obs(self.env) # (5, 64, 64)
            
            # Also initialize batch_global storage
            batch_global = []
            
            for step in range(200):
                steps_performed += 1
                # 1. Get Actions
                actions = []
                # Team 0 (RL)
                # Prepare batch input for efficiency
                curr_maps = []
                curr_selfs = []
                
                # Assume Team 0 is the first half of agents
                for i in range(half_agents):
                    o = obs_list[i]
                    om = o['map'].clone().detach().float()
                    osf = o['self'].clone().detach().float()
                    curr_maps.append(om)
                    curr_selfs.append(osf)

                # Stack for batch processing if desired, but get_action handles singles.
                # Let's keep loop for now to match structure, but we could batch this.
                
                for i in range(half_agents):
                    mv, pk, mv_lp, pk_lp, val = self.agent.get_action(curr_maps[i], curr_selfs[i], global_critic_map)
                    
                    batch_obs_map.append(curr_maps[i])
                    batch_obs_self.append(curr_selfs[i])
                    batch_global.append(global_critic_map) # Store for update
                    
                    batch_acts_mv.append(mv)
                    batch_acts_pk.append(pk)
                    batch_log_probs.append(mv_lp + pk_lp)
                    batch_vals.append(val)
                    
                    actions.append({'movement': mv, 'pickup': pk})
                    
                    # Track Actions
                    ep_metrics["act_move_mag"].append(np.linalg.norm(mv))
                    ep_metrics["act_pick_counts"][pk] += 1
                    
                # Team 1 (Heuristic)
                for i in range(half_agents, num_agents):
                    actions.append(self.heuristic.act(obs_list[i]))
                    
                # 2. Step
                next_obs, rewards, term, trunc, _ = self.env.step(actions)
                
                # 3. Store Rewards
                step_rews = []
                for i in range(half_agents):
                    agent_rew = rewards[i]
                    total_r = sum(agent_rew.values())
                    
                    step_rews.append(total_r)
                    ep_reward += total_r
                    
                    # Breakdown
                    for k, v in agent_rew.items():
                        if 'delivery' in k: ep_metrics['r_delivery'] += v
                        elif 'combat' in k or 'win' in k or 'lose' in k: ep_metrics['r_combat'] += v
                        elif 'death' in k or 'health' in k: ep_metrics['r_survival'] += v
                        else: ep_metrics['r_misc'] += v

                batch_rews.extend(step_rews)
                batch_dones.extend([term or trunc] * half_agents)
                
                obs_list = next_obs
                global_critic_map = get_global_critic_obs(self.env)
                
                if term or trunc: break
                
            # --- POST EPISODE UPDATE ---
            
            # Bootstrapping for GAE
            # We need the value of the NEXT state (obs_list is now next_obs)
            # If done, next_val is 0.
            
            next_vals = np.zeros(half_agents)
            if not (term or trunc):
                 # Compute values for the last observation
                 # We need to construct the NEXT global map if we want accuracy,
                 # but global_critic_map is already updated to next_global_critic_map at end of loop
                 # EXCEPT on the last step where we break.
                 # Wait, if we break, global_critic_map IS the terminal state map.
                 # So we can just use global_critic_map.
                 
                 for i in range(half_agents):
                    o = obs_list[i]
                    om = o['map'].clone().detach().float()
                    osf = o['self'].clone().detach().float()
                    _, _, _, _, val = self.agent.get_action(om, osf, global_critic_map)
                    next_vals[i] = val

            # Reshape data to (Steps, Agents)
            batch_vals = np.array(batch_vals).reshape(steps_performed, half_agents)
            batch_rews = np.array(batch_rews).reshape(steps_performed, half_agents)
            batch_dones = np.array(batch_dones).reshape(steps_performed, half_agents)
            
            # --- UPDATE REWARD SCALER ---
            # We update the scaler with the raw rewards seen this episode
            # But usually, it's better to update with returns? 
            # Standard practice: Update with R (rewards) to normalize them to N(0,1)
            self.reward_scaler.update(batch_rews.flatten())
            
            # Normalize Rewards
            # r_hat = r / sqrt(var + eps)
            batch_rews = batch_rews / np.sqrt(self.reward_scaler.var + 1e-8)
            
            # GAE Calculation
            advantages = np.zeros_like(batch_rews)
            last_gae_lam = np.zeros(half_agents)
            
            for t in reversed(range(steps_performed)):
                if t == steps_performed - 1:
                    next_non_terminal = 1.0 - (1.0 if (term or trunc) else 0.0) # If episode ended, next is terminal
                    next_values = next_vals
                else:
                    next_non_terminal = 1.0 - batch_dones[t] # Should be 1 unless died? But we use global term.
                    # Actually batch_dones tracks global termination.
                    # Individual agent death is not "done" for the env, but maybe for the agent?
                    # The env wrapper returns global term.
                    next_values = batch_vals[t+1]
                
                delta = batch_rews[t] + GAMMA * next_values * next_non_terminal - batch_vals[t]
                last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
                advantages[t] = last_gae
                last_gae_lam = last_gae
                
            returns = advantages + batch_vals
            
            # Flatten back
            flat_advs = advantages.flatten()
            flat_rets = returns.flatten()
            
            # Construct Rollout Batch
            rollouts = []
            for i in range(len(batch_obs_map)):
                rollouts.append((
                    batch_obs_map[i], 
                    batch_obs_self[i], 
                    batch_global[i],
                    batch_acts_mv[i], 
                    batch_acts_pk[i], 
                    batch_log_probs[i], 
                    flat_rets[i], 
                    flat_advs[i]
                ))
            
            # Update Policy
            self.agent.update(rollouts)
            
            # --- LOGGING TO CSV ---
            log_file = f"training_log_{self.scenario}.csv"
            file_exists = os.path.isfile(log_file)
            
            total_acts = sum(ep_metrics["act_pick_counts"].values()) + 1e-8
            move_mag = np.mean(ep_metrics["act_move_mag"])
            pick_pct = ep_metrics["act_pick_counts"][1] / total_acts * 100
            drop_pct = ep_metrics["act_pick_counts"][2] / total_acts * 100
            
            with open(log_file, "a") as f:
                if not file_exists:
                    f.write("Episode,RawReward,Advantage,Deliv,Combat,Surv,Misc,MoveMag,PickPct,DropPct\n")
                f.write(f"{ep},{ep_reward:.2f},{flat_advs.mean():.4f},{ep_metrics['r_delivery']:.2f},"
                        f"{ep_metrics['r_combat']:.2f},{ep_metrics['r_survival']:.2f},{ep_metrics['r_misc']:.2f},"
                        f"{move_mag:.2f},{pick_pct:.2f},{drop_pct:.2f}\n")

            if ep % 10 == 0:
                # Calculate Action Stats
                
                print(f"Ep {ep} | Raw Reward: {ep_reward:.1f} | Adv: {flat_advs.mean():.3f}")
                print(f"   Breakdown -> Deliv: {ep_metrics['r_delivery']:.1f}, Cmbt: {ep_metrics['r_combat']:.1f}, Surv: {ep_metrics['r_survival']:.1f}, Misc: {ep_metrics['r_misc']:.1f}")
                print(f"   Actions   -> MoveMag: {move_mag:.2f}, Pick: {pick_pct:.1f}%, Drop: {drop_pct:.1f}%")
                
                suffix = f"_{self.run_name}" if self.run_name else ""
                self.save_checkpoint(f"ctde_checkpoints/checkpoint_{self.scenario}{suffix}_ep{ep}.pt")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='unified')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--randomize', action='store_true', help='Enable domain randomization')
    parser.add_argument('--name', type=str, default=None, help='Optional run name suffix for checkpoints')
    args = parser.parse_args()
    
    trainer = CTDETrainer(scenario=args.scenario, resume_path=args.resume, randomize=args.randomize, run_name=args.name)
    trainer.train()
