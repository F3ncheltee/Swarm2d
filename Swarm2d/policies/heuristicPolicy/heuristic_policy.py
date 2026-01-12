import numpy as np
import torch
import random
from Swarm2d.constants import NODE_FEATURE_MAP, NODE_TYPE

class HeuristicPolicy:
    def __init__(self, action_space):
        """
        A rule-based heuristic policy for the Swarm2d environment.
        The policy follows a priority of actions:
        1. Survival (retreat to hive if low health)
        2. Combat (attack nearby enemies)
        3. Resource Gathering (collect and deliver resources)
        4. Exploration (move towards the center or randomly)
        """
        self.action_space = action_space
        # Constants for decision making
        self.low_health_threshold = 0.3
        self.combat_radius = 20  # Engage enemies within this distance

    def act(self, obs: dict) -> dict:
        """
        Determines the action based on the current observation.

        Args:
            obs: A dictionary containing 'self', 'graph', 'map', and 'memory_map' observations.

        Returns:
            A dictionary representing the action to take.
        """
        # --- 1. Parse Observations ---
        self_obs = obs['self']
        graph = obs['graph']

        if not isinstance(graph.x, torch.Tensor) or graph.x.shape[0] == 0:
            return self._get_default_action()

        # Find own agent node in the graph
        ego_mask = graph.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5
        if not torch.any(ego_mask):
            return self._get_default_action()
        
        ego_node_idx = torch.where(ego_mask)[0][0]
        ego_node = graph.x[ego_node_idx]
        ego_pos = graph.pos[ego_node_idx]

        # --- 2. Decision Logic ---
        
        # Survival: If health is low, retreat
        health = ego_node[NODE_FEATURE_MAP['health_norm']]
        if health < self.low_health_threshold:
            action = self._retreat(ego_pos, graph)
            return action

        # Combat: Engage nearby enemies
        action = self._handle_combat(ego_node, ego_pos, graph)
        if action:
            return action

        # Resource Gathering
        action = self._handle_resources(ego_node, ego_pos, graph)
        if action:
            return action

        # Exploration
        action = self._explore(ego_pos)
        return action

    def _get_entities(self, graph, entity_type: int):
        """Helper to get entities of a specific type from the graph."""
        type_mask = graph.x[:, NODE_FEATURE_MAP['node_type_encoded']] == entity_type
        return graph.pos[type_mask], graph.x[type_mask]

    def _find_closest_entity(self, ego_pos, entity_positions):
        """Finds the closest entity to the agent."""
        if entity_positions.shape[0] == 0:
            return None, float('inf')
        distances = torch.norm(entity_positions - ego_pos, dim=1)
        min_dist, min_idx = torch.min(distances, dim=0)
        return entity_positions[min_idx], min_dist.item()

    def _retreat(self, ego_pos, graph):
        """Move towards the allied hive."""
        # Find own hive
        hives_pos, hives_feat = self._get_entities(graph, NODE_TYPE['hive'])
        ego_team = (graph.x[graph.x[:, NODE_FEATURE_MAP['is_ego']] > 0.5][0][NODE_FEATURE_MAP['team_id']]).item()

        allied_hives_mask = hives_feat[:, NODE_FEATURE_MAP['team_id']] == ego_team
        allied_hives_pos = hives_pos[allied_hives_mask]

        if allied_hives_pos.shape[0] > 0:
            hive_pos = allied_hives_pos[0] # Assume one hive per team
            direction = hive_pos - ego_pos
            normalized_direction = direction / (torch.norm(direction) + 1e-6)
            return {"movement": normalized_direction.cpu().numpy(), "pickup": 0}
        return self._explore(ego_pos) # If no hive found, explore

    def _handle_combat(self, ego_node, ego_pos, graph):
        """Handle combat logic."""
        ego_team = ego_node[NODE_FEATURE_MAP['team_id']].item()
        agents_pos, agents_feat = self._get_entities(graph, NODE_TYPE['agent'])
        
        enemy_mask = agents_feat[:, NODE_FEATURE_MAP['team_id']] != ego_team
        enemy_positions = agents_pos[enemy_mask]
        
        if enemy_positions.shape[0] > 0:
            closest_enemy_pos, dist = self._find_closest_entity(ego_pos, enemy_positions)
            if dist < self.combat_radius:
                direction = closest_enemy_pos - ego_pos
                normalized_direction = direction / (torch.norm(direction) + 1e-6)
                return {"movement": normalized_direction.cpu().numpy(), "pickup": 1} # 1 to grapple/attack
        return None

    def _handle_resources(self, ego_node, ego_pos, graph):
        """Handle resource gathering logic."""
        is_carrying = ego_node[NODE_FEATURE_MAP['is_carrying']] > 0.5

        if is_carrying:
            return self._retreat(ego_pos, graph) # Same as retreat: move to hive
        
        resources_pos, _ = self._get_entities(graph, NODE_TYPE['resource'])
        if resources_pos.shape[0] > 0:
            closest_res_pos, dist = self._find_closest_entity(ego_pos, resources_pos)
            if dist < 5: # If very close, pickup
                return {"movement": np.zeros(2), "pickup": 1}
            else: # Move towards it
                direction = closest_res_pos - ego_pos
                normalized_direction = direction / (torch.norm(direction) + 1e-6)
                return {"movement": normalized_direction.cpu().numpy(), "pickup": 0}
        return None

    def _explore(self, ego_pos):
        """Default exploration behavior."""
        # Move towards the center
        center_pos = torch.tensor([500.0, 500.0], dtype=torch.float32, device=ego_pos.device)
        direction = center_pos - ego_pos
        if torch.norm(direction) < 10:
             return self._get_default_action() # If close to center, move randomly
        normalized_direction = direction / (torch.norm(direction) + 1e-6)
        return {"movement": normalized_direction.cpu().numpy(), "pickup": 0}

    def _get_default_action(self):
        """Return a default action (e.g., random movement)."""
        return {"movement": np.random.randn(2), "pickup": 0}
