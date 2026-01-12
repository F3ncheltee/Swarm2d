import numpy as np
from Swarm2d.constants import COMBAT_RADIUS


class SimpleHeuristicPolicy:
    def __init__(self, env, team_id=None):
        self.env = env
        self.team_id = team_id
        self.center_pos = np.array([env.width / 2, env.height / 2])
        self.low_health_threshold = 30  # Retreat if health is below 30%
        # DEPRECATED: self.resource_pickup_distance = 7.5 # From constants.py

    def get_actions(self, obs, agent_ids):
        """
        Gets actions for a specific subset of agent_ids.
        The `obs` is global, but this policy only computes actions for the agents
        it is responsible for.
        """
        actions = []
        # New: Keep track of resources targeted in this step to prevent dogpiling
        targeted_resources = set()

        # Create a dictionary to hold the final action for each agent
        agent_actions = {}

        # First pass: Determine high-priority actions (retreat/deliver)
        for i in agent_ids:
            agent = self.env.agents[i]
            if not (agent and agent.get('alive')):
                agent_actions[i] = {"movement": np.zeros(2), "pickup": 0}
                continue

            # Priority 1: Survival or Delivery. If health is low or carrying, retreat to hive.
            if agent.get('health', 100) < self.low_health_threshold or agent.get('has_resource'):
                agent_actions[i] = self._retreat(agent)

        # Second pass: Assign gathering or attack tasks to available agents
        for i in agent_ids:
            if i in agent_actions:  # Skip agents that already have an action
                continue

            agent = self.env.agents[i]
            agent_pos = agent.get('pos')

            # Find the closest available resource
            closest_res, res_dist = self._find_closest(agent_pos, self.env.resources, exclude_ids=targeted_resources)

            # Find the closest enemy
            enemies = [e for e in self.env.agents if e and e.get('alive') and e.get('team') != agent.get('team')]
            closest_enemy, enemy_dist = self._find_closest(agent_pos, enemies)

            # Decide whether to gather or attack based on proximity
            if res_dist <= enemy_dist:
                if closest_res:
                    targeted_resources.add(closest_res['id'])
                    agent_actions[i] = self._gather_resource(agent, closest_res, res_dist)
                else:
                    # No resources, but maybe enemies are far away or non-existent
                    agent_actions[i] = self._explore(agent)
            else:
                if closest_enemy:
                    agent_actions[i] = self._attack_enemy(agent, closest_enemy, enemy_dist)
                else:
                    # No enemies, but maybe resources are far away or non-existent
                    agent_actions[i] = self._explore(agent)

        # Ensure actions are in the correct order
        for i in agent_ids:
            actions.append(agent_actions.get(i, {"movement": np.zeros(2), "pickup": 0}))

        return actions

    def _find_closest(self, agent_pos, entity_list, exclude_ids=None):
        if exclude_ids is None:
            exclude_ids = set()
        closest_entity = None
        min_dist_sq = float('inf')
        for entity in entity_list:
            # Added check for 'id' in exclude_ids
            if not entity or entity.get('pos') is None or entity.get('id') in exclude_ids:
                continue
            dist_sq = np.sum((agent_pos - entity['pos'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_entity = entity
        return closest_entity, np.sqrt(min_dist_sq) if closest_entity else float('inf')

    def _retreat(self, agent):
        agent_pos = agent.get('pos')
        team_id = agent.get('team')
        hive = self.env.hives.get(team_id)

        if hive and hive.get('pos') is not None:
            target_pos = hive['pos']
            direction = target_pos - agent_pos
            norm = np.linalg.norm(direction)
            if norm > self.env.metadata['hive_delivery_radius']:
                return {"movement": direction / (norm + 1e-6), "pickup": 0}
        
        # If no hive or already at hive, do nothing
        return {"movement": np.zeros(2), "pickup": 0}

    def _gather_resource(self, agent, target_res, dist):
        """Generates an action to move towards and pick up a specific resource."""
        agent_pos = agent.get('pos')
        target_pos = target_res['pos']
        direction = target_pos - agent_pos

        # Dynamic pickup distance: agent radius + resource radius + a small buffer
        agent_radius = agent.get('agent_radius', self.env.agent_radius_config)
        resource_radius = target_res.get('radius_pb', 5.0) # Use a fallback default
        
        # MATCH THE ENVIRONMENT'S LOGIC EXACTLY
        pickup_distance = (agent_radius + resource_radius) * 1.1

        if dist < pickup_distance:
            # Close enough to pickup
            print(f"[POLICY DEBUG] Agent {agent['id']} ATTEMPTING PICKUP on Res {target_res['id']}. Dist: {dist:.2f}, Required: <{pickup_distance:.2f}")
            return {"movement": np.zeros(2), "pickup": 1}
        else:
            # Move towards resource
            norm = np.linalg.norm(direction)
            return {"movement": direction / (norm + 1e-6), "pickup": 0}

    def _attack_enemy(self, agent, target_enemy, dist):
        """Generates an action to move towards and grapple an enemy."""
        agent_pos = agent.get('pos')
        target_pos = target_enemy['pos']
        direction = target_pos - agent_pos

        # Use COMBAT_RADIUS from constants for the attack/grapple distance
        grapple_distance = COMBAT_RADIUS

        if dist < grapple_distance:
            # Close enough to grapple
            print(f"[POLICY DEBUG] Agent {agent['id']} ATTEMPTING GRAPPLE on Agent {target_enemy['id']}. Dist: {dist:.2f}, Required: <{grapple_distance:.2f}")
            return {"movement": np.zeros(2), "pickup": 1}  # pickup: 1 is the 'interact' action
        else:
            # Move towards enemy
            norm = np.linalg.norm(direction)
            return {"movement": direction / (norm + 1e-6), "pickup": 0}

    def _explore(self, agent):
        agent_pos = agent.get('pos')
        direction = self.center_pos - agent_pos
        norm = np.linalg.norm(direction)
        if norm > 10:
            return {"movement": direction / (norm + 1e-6), "pickup": 0}
        return {"movement": np.random.randn(2), "pickup": 0}
