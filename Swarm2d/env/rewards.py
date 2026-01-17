import torch
import torch.nn as nn
from typing import List, Dict, Set, Union
import math
import numpy as np
from constants import (
    CELL_SIZE, REWARD_CONFIG, HIVE_DELIVERY_RADIUS, HIVE_MAX_HEALTH, 
    AGENT_RADIUS, REWARD_COMPONENT_KEYS, NUM_REWARD_COMPONENTS, DISCOVERY_COOLDOWN
)
from env.helper import distance, normalized_distance



class RewardManager (nn.Module):
    """
    Manages the calculation and distribution of rewards to agents in the environment.

    This class is responsible for all reward-related logic, from calculating rewards for
    specific events like resource delivery and combat, to handling continuous rewards
    for exploration and maintaining hive health. It supports dynamic reward structures
    through a multiplier system, which is crucial for curriculum learning.

    Attributes:
        env (Swarm2DEnv): A reference to the main environment class.
        device (torch.device): The device (CPU or GPU) for tensor computations.
        team_reward_multipliers (dict): A dictionary storing team-specific multipliers
                                        for different reward components, allowing for
                                        dynamic curriculum-based reward shaping.
    """
    def __init__(self, env_ref, team_reward_overrides=None):
        super(RewardManager, self).__init__()
        self.env = env_ref
        self.device = self.env.device
        # team_reward_overrides will now store MULTIPLIERS from the GUI, not absolute values.
        self.team_reward_multipliers = team_reward_overrides if team_reward_overrides is not None else {}

    def update_reward_multipliers(self, new_multipliers: Dict[str, Dict[str, float]]):
        """
        Updates the reward multipliers for each team.

        This is the primary method for implementing curriculum learning or applying
        GUI-driven changes to the reward structure. It replaces the existing multipliers
        with a new set.

        Args:
            new_multipliers: A dictionary where keys are team IDs (as strings)
                             and values are dictionaries of reward component multipliers.
                             Example: {'0': {'r_delivery': 1.5, 'r_combat_light': 0.5}}
        """
        self.team_reward_multipliers = new_multipliers
        # No need to re-initialize a separate weights dictionary anymore.
        print(f"INFO: Reward multipliers updated for Team 0: {self.team_reward_multipliers.get('0', {})}")

    def get_reward(self, team_id: int, reward_key: str, base_value: float) -> float:
        """
        Calculates a final reward value by applying a team-specific multiplier to a base value.

        This function acts as the central point for reward calculation. It fetches the
        default multiplier from the global `REWARD_CONFIG`, checks for any team-specific
        overrides, and then applies the final multiplier to the provided base value.

        Args:
            team_id (int): The ID of the team receiving the reward.
            reward_key (str): The string identifier for the reward (e.g., 'r_delivery').
            base_value (float): The dynamically calculated or constant base value for the reward event.

        Returns:
            float: The final calculated reward value after applying the multiplier.
        """
        # 1. Get the default multiplier from the central REWARD_CONFIG
        default_multiplier = REWARD_CONFIG.get(reward_key, {}).get('default_multiplier', 1.0)

        # 2. Check for a team-specific multiplier override from the GUI or curriculum
        team_multipliers = self.team_reward_multipliers.get(str(team_id), {})
        multiplier = team_multipliers.get(reward_key, default_multiplier)
        
        # 3. The final reward is the base value scaled by the multiplier
        return base_value * multiplier

    # --- Helper: Vectorize Reward ---
    def vectorize_reward(reward_dict: Dict) -> torch.Tensor:
        """
        Converts a reward dictionary into a fixed-size tensor.

        This static method is a utility for converting the dictionary of named reward
        components into a standardized tensor format, which is often required by
        learning algorithms. The order of elements in the tensor is determined by
        the global `REWARD_COMPONENT_KEYS` list.

        Args:
            reward_dict (Dict): A dictionary of reward components for a single agent.

        Returns:
            torch.Tensor: A 1D tensor representing the vectorized rewards.
        """
        # REWARD_COMPONENT_KEYS should be defined globally in the script
        vec = torch.zeros(NUM_REWARD_COMPONENTS, dtype=torch.float32) # Use global constant
        if not reward_dict: # Handle empty dict case
            return vec
        try:
            for i, key in enumerate(REWARD_COMPONENT_KEYS):
                if i < NUM_REWARD_COMPONENTS: # Safety check
                    # Use .get() with default 0.0 for missing keys
                    value = reward_dict.get(key, 0.0)
                    # Ensure the value is a number before assigning
                    if isinstance(value, (int, float)):
                        vec[i] = float(value)
                    # else: # Optional: Warn about non-numeric values if needed
                    #    print(f"Warning: Non-numeric value found for reward key '{key}': {value}")
        except IndexError:
            print(f"Warning: REWARD_COMPONENT_KEYS length mismatch during vectorization. Expected {NUM_REWARD_COMPONENTS}.")
        except Exception as e:
            print(f"Error vectorizing reward dict {reward_dict}: {e}")
        return vec

    def _process_resource_delivery(self, rewards: List[Dict], gnn_idx_to_res_obj_map: Dict) -> set[int]:
        """
        Handles all logic related to resource delivery, including progress and final drop-off rewards.

        This method checks each resource being carried by agents. It calculates a 'progress'
        reward for agents moving closer to their target hive. If a resource is within the
        hive's delivery radius, it awards a final delivery bonus, updates the hive's stats,
        and marks the resource for cleanup.

        Args:
            rewards (List[Dict]): The list of reward dictionaries for all agents, to be updated.
            gnn_idx_to_res_obj_map (Dict): A mapping used to retrieve resource objects (not currently used here).

        Returns:
            set[int]: A set of IDs for the resources that were successfully delivered in this step.
        """
        delivered_ids = set()
        # Carried delivery
        for res in self.env.resources:
            if res.get('delivered') or not res.get("carriers"): continue
            carriers = [self.env.agents[c['id']] for c in res.get("carriers",[]) if self.env.agents[c['id']].get('alive')]
            if carriers and res.get("pos") is not None and self.env.hives.get(res.get("target_hive"),{}).get("state")=="active":
                target_hive = self.env.hives[res["target_hive"]]
                
                # --- APPLY HOLDING REWARD ALWAYS (Base Salary) ---
                # Give reward just for holding, regardless of movement
                hold_reward = self.get_reward(carriers[0]['team'], "r_holding", REWARD_CONFIG['r_holding']['default_value'])
                for c in carriers:
                    rewards[c['id']]["r_holding"] = rewards[c['id']].get("r_holding", 0.0) + hold_reward

                current_dist = distance(res["pos"], target_hive["pos"])
                progress = normalized_distance(res.get("prev_distance",current_dist),self.env.d_max) - normalized_distance(current_dist,self.env.d_max)
                if abs(progress) > 1e-6:
                    if progress > 0:
                        # POSITIVE PROGRESS
                        base_reward_val = REWARD_CONFIG['r_progress_positive']['default_value']
                        dynamic_base = progress * base_reward_val
                        reward_per_agent = dynamic_base / len(carriers)
                        for c in carriers:
                            final_reward = self.get_reward(c['team'], "r_progress_positive", reward_per_agent)
                            rewards[c['id']]["r_progress_positive"] += final_reward
                    else:
                        # NEGATIVE PROGRESS (Penalty)
                        # progress is negative, so we use abs() to keep the magnitude, but apply it to the negative reward key
                        base_reward_val = REWARD_CONFIG['r_progress_negative']['default_value']
                        
                        # Ensure this is NEGATIVE. 
                        # 'progress' is negative. 'base_reward_val' (50.0) is positive.
                        # 'dynamic_base' will be negative (e.g., -0.01 * 50.0 = -0.5).
                        dynamic_base = progress * base_reward_val 
                        
                        reward_per_agent = dynamic_base / len(carriers)
                        for c in carriers:
                            # Pass to get_reward. 
                            # If multiplier is POSITIVE (0.1), then final_reward is NEGATIVE.
                            final_reward = self.get_reward(c['team'], "r_progress_negative", reward_per_agent)
                            rewards[c['id']]["r_progress_negative"] += final_reward

                res["prev_distance"] = current_dist
                if current_dist < HIVE_DELIVERY_RADIUS:
                    self.env.resources_delivered_count+=1
                    delivered_ids.add(res['id'])
                    
                    # Base reward now considers size and cooperation
                    coop_bonus = self.get_reward(carriers[0]['team'], 'coop_collection_bonus', REWARD_CONFIG['coop_collection_bonus']['default_value'])
                    base_delivery_reward = REWARD_CONFIG['r_delivery']['default_value']
                    # Don't multiply by base value twice. base_delivery_reward is the base value.
                    dynamic_base = (base_delivery_reward * res.get("size", 1.0) * (coop_bonus if res.get("cooperative") else 1.0))
                    
                    reward_per_agent = dynamic_base / len(carriers)
                    for c in carriers:
                        # Use 1.0 as base for get_reward because we calculated the full amount in dynamic_base
                        # If we pass reward_per_agent as base, get_reward multiplies it by the multiplier AGAIN.
                        final_reward = reward_per_agent * self.env.reward_manager.team_reward_multipliers.get(str(c['team']), {}).get('r_delivery', 1.0)
                        rewards[c['id']]["r_delivery"] += final_reward
                        self.env.physics_manager._cleanup_agent_attachments(self.env.agents[c['id']])
                    food_val=res.get("food_value",0.); health_needed=max(0,self.env.metadata.get('hive_max_health',100)-target_hive["health"])
                    to_health=min(food_val,health_needed)
                    target_hive["health"]+=to_health
                    target_hive["food_store"]+=food_val-to_health
                    self.env.physics_manager._cleanup_resource_attachments(res)
        return delivered_ids

    def _process_discovery_and_misc_rewards(self, rewards: List[Dict], proximity_data: Dict = None):
        """
        Handles exploration, discovery of entities, and continuous grapple-related rewards.
        
        Now uses pre-computed proximity data from _vectorized_proximity_search() to avoid 
        recalculating O(n²) distance checks. This provides a massive performance improvement.

        This method iterates through all living agents and calculates:
        1. An intrinsic exploration reward based on visiting new or less-visited grid cells.
        2. A discovery reward for seeing a resource for the first time (subject to a cooldown).
        3. A small continuous reward/penalty for being in a grappling/grappled state.

        Args:
            rewards (List[Dict]): The list of reward dictionaries for all agents, to be updated.
            proximity_data (Dict): Pre-computed proximity information from _vectorized_proximity_search(),
                                  containing agent_resource_map, agent_enemy_map, etc.
        """
        # Extract proximity maps if provided, otherwise use empty dicts (fallback for compatibility)
        agent_resource_map = proximity_data.get('agent_resource_map', {}) if proximity_data else {}
        agent_enemy_map = proximity_data.get('agent_enemy_map', {}) if proximity_data else {}
        
        for idx, agent in enumerate(self.env.agents):
            if agent and agent.get('alive') and agent.get('pos') is not None:
                # --- 1. Exploration Reward (unchanged) ---
                cell_key=(int(agent['pos'][0]//CELL_SIZE),int(agent['pos'][1]//CELL_SIZE))
                visit_count=self.env.exploration_counts.get(cell_key,0)+1
                self.env.exploration_counts[cell_key]=visit_count
                
                base_reward_val = REWARD_CONFIG['r_exploration_intrinsic']['default_value']
                dynamic_base = base_reward_val / math.sqrt(visit_count)
                final_reward = self.get_reward(agent['team'], "r_exploration_intrinsic", dynamic_base)
                rewards[idx]["r_exploration_intrinsic"] += final_reward

                # --- 2. Resource Discovery (OPTIMIZED: Use pre-computed proximity) ---
                # The agent_resource_map already contains only resources within observation radius
                nearby_resources = agent_resource_map.get(idx, [])
                for res in nearby_resources:
                    if res and not res.get('delivered'):
                        discovery_multiplier = self.try_award_discovery_reward('resource', res['id'], self.env.step_counter, agent['team'])
                        base_reward = discovery_multiplier * REWARD_CONFIG['r_resource_found']['default_value']
                        final_reward = self.get_reward(agent['team'], "r_resource_found", base_reward)
                        rewards[idx]["r_resource_found"] += final_reward

                # --- 3. Obstacle Discovery (OPTIMIZED: Vectorized distance check) ---
                # Use numpy vectorization for obstacle checks (typically 10-20 obstacles)
                if self.env.obstacles:
                    agent_pos = agent['pos']
                    obs_radius = agent['obs_radius']
                    
                    # Vectorized distance calculation for all obstacles at once
                    obs_positions = np.array([obs['pos'] for obs in self.env.obstacles if obs and obs.get('pos') is not None])
                    obs_radii = np.array([obs.get('radius', 10) for obs in self.env.obstacles if obs and obs.get('pos') is not None])
                    valid_obstacles = [obs for obs in self.env.obstacles if obs and obs.get('pos') is not None]
                    
                    if len(obs_positions) > 0:
                        # Calculate all distances at once
                        diffs = obs_positions - agent_pos
                        distances = np.linalg.norm(diffs, axis=1)
                        
                        # Find obstacles within observation radius
                        in_range_mask = distances < (obs_radius + obs_radii)
                        
                        for obs_idx in np.where(in_range_mask)[0]:
                            obs = valid_obstacles[obs_idx]
                            discovery_multiplier = self.try_award_discovery_reward('obstacle', obs['id'], self.env.step_counter, agent['team'])
                            base_reward = discovery_multiplier * REWARD_CONFIG['r_obstacle_found']['default_value']
                            final_reward = self.get_reward(agent['team'], "r_obstacle_found", base_reward)
                            rewards[idx]["r_obstacle_found"] += final_reward

                # --- 4. Enemy Agent Discovery (OPTIMIZED: Use pre-computed proximity) ---
                # This eliminates the O(n²) agent-to-agent loop!
                nearby_enemies = agent_enemy_map.get(idx, [])
                for enemy_agent in nearby_enemies:
                    if enemy_agent and enemy_agent.get('alive'):
                        discovery_multiplier = self.try_award_discovery_reward('enemy', enemy_agent['id'], self.env.step_counter, agent['team'])
                        base_reward = discovery_multiplier * REWARD_CONFIG['r_enemy_found']['default_value']
                        final_reward = self.get_reward(agent['team'], "r_enemy_found", base_reward)
                        rewards[idx]["r_enemy_found"] += final_reward

                # --- 5. Hive Discovery (OPTIMIZED: Vectorized distance check) ---
                # Use numpy vectorization for hive checks (typically 6 hives)
                agent_team = agent['team']
                enemy_hives = [h for h in self.env.hives.values() if h and h.get('owner') != agent_team and h.get('pos') is not None]
                
                if enemy_hives:
                    agent_pos = agent['pos']
                    obs_radius = agent['obs_radius']
                    
                    # Vectorized distance calculation for all enemy hives at once
                    hive_positions = np.array([h['pos'] for h in enemy_hives])
                    hive_radii = np.array([h.get('radius', 25) for h in enemy_hives])
                    
                    # Calculate all distances at once
                    diffs = hive_positions - agent_pos
                    distances = np.linalg.norm(diffs, axis=1)
                    
                    # Find hives within observation radius
                    in_range_mask = distances < (obs_radius + hive_radii)
                    
                    for hive_idx in np.where(in_range_mask)[0]:
                        hive = enemy_hives[hive_idx]
                        discovery_multiplier = self.try_award_discovery_reward('hive', hive['id'], self.env.step_counter, agent['team'])
                        base_reward = discovery_multiplier * REWARD_CONFIG['r_hive_found']['default_value']
                        final_reward = self.get_reward(agent['team'], "r_hive_found", base_reward)
                        rewards[idx]["r_hive_found"] += final_reward
            
            # --- 6. Grapple Rewards (unchanged) ---
            if agent.get('is_grappling'):
                base_reward = REWARD_CONFIG['r_grapple_control']['default_value']
                final_reward = self.get_reward(agent['team'], "r_grapple_control", base_reward)
                rewards[idx]['r_grapple_control'] += final_reward
            
            if agent.get('is_grappled'):
                base_reward = REWARD_CONFIG['r_grapple_controlled']['default_value']
                final_reward = self.get_reward(agent['team'], "r_grapple_controlled", base_reward)
                rewards[idx]['r_grapple_controlled'] += final_reward

    
    def _process_continuous_hive_rewards(self, rewards: List[Dict]):
        """
        Calculates and applies a continuous reward based on the total health of hives owned by each team.

        This provides a continuous incentive for teams to maintain and protect their hives.
        The reward for each agent is proportional to the total current health of all hives
        owned by their team.

        Args:
            rewards (List[Dict]): The list of reward dictionaries for all agents, to be updated.
        """
        team_hive_health = {i: 0 for i in range(self.env.num_teams)}

        # 1. Sum up the health of all active hives for each team
        for hive in self.env.hives.values():
            if hive and hive.get('state') == 'active':
                owner_team = hive.get('owner')
                if owner_team in team_hive_health:
                    team_hive_health[owner_team] += hive.get('health', 0)

        # 2. Apply the reward to each agent on the respective team
        for agent in self.env.agents:
            if agent and agent.get('alive'):
                agent_team = agent['team']
                total_health_for_team = team_hive_health.get(agent_team, 0)

                if self.env.debug_mode and agent['id'] == 0:
                    print(f"Debug Agent 0 (Team {agent_team}): Total Hive Health for team = {total_health_for_team}")
                
                if total_health_for_team > 0:
                    # The base value of the reward is the total health of the team's hives
                    # The base value should be proportional to the team's total hive health.
                    # We'll use the total health and let the multiplier in REWARD_CONFIG scale it.
                    base_reward_val = total_health_for_team * REWARD_CONFIG['r_hive_health_continuous']['default_value']
                    final_reward = self.get_reward(agent_team, 'r_hive_health_continuous', base_value=base_reward_val)
                    rewards[agent['id']]['r_hive_health_continuous'] += final_reward

    def try_award_discovery_reward(self, objective: str, object_id: Union[int, str], current_step: int, team_id: int): 
        """
        Awards a discovery reward if a cooldown has passed for a specific object for a SPECIFIC TEAM.

        This function manages a 'discovery record' to prevent agents from repeatedly
        receiving rewards for seeing the same object. A reward is only given if the
        `DISCOVERY_COOLDOWN` period has elapsed since the last time a reward was given
        for this object to this team.

        Args:
            objective (str): The type of object discovered (e.g., 'resource').
            object_id (Union[int, str]): The unique ID of the discovered object.
            current_step (int): The current step count of the environment.
            team_id (int): The ID of the team discovering the object.

        Returns:
            float: Returns 1.0 if a reward is given, otherwise 0.0.
        """
        key = (objective, object_id, team_id) # Include team_id in key!
        # Use DISCOVERY_COOLDOWN defined globally
        cooldown_duration = DISCOVERY_COOLDOWN 
        last_confirmed_or_rewarded_time = self.env.discovery_records.get(key, -cooldown_duration)

        eligible_for_reward = (current_step - last_confirmed_or_rewarded_time >= cooldown_duration)
        if eligible_for_reward:
            self.env.discovery_records[key] = current_step # Update because reward is given
            return 1.0
        else:
            # NO REWARD IS GIVEN, BUT THE OBJECT IS STILL "CONFIRMED" AS SEEN
            self.env.discovery_records[key] = current_step # Update because it was seen & processed
            return 0.0


