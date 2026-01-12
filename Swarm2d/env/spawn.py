import numpy as np
import math
import random
import pybullet as p
import torch
import torch.nn as nn
from typing import Optional

from constants import *
from env.helper import distance


class HiveSpawn (nn.Module):
    """Handles the spawning and creation of hive entities in the simulation."""
    
    def _create_hive_instance(self, team_id, pos, spawned_positions):
        """
        Helper function to create a hive's data dictionary and its PyBullet body.

        This internal method is called by `spawn_hives` to perform the actual
        creation of a single hive, including setting up its physical properties,
        visual appearance, and collision filters in the PyBullet engine.

        Args:
            team_id (int): The team ID this hive belongs to.
            pos (np.ndarray): The [x, y] position to spawn the hive.
            spawned_positions (list): A list of positions where hives have already
                                      been spawned, used for distance checks.
        
        Returns:
            bool: True if the hive was created successfully.
        """
        hive_radius_pb = self.metadata.get('hive_radius_assumed', 25.0)
        spawned_positions.append(pos.copy())
        self.hives[team_id] = {
            "id": self.hive_id_counter, 
            "original_team_id": team_id,
            "pos": pos.copy(),        
            "health": self.metadata.get('hive_max_health', HIVE_MAX_HEALTH),
            "food_store": self.metadata.get('hive_max_food', 100.0),
            "state": "active",
            "respawn_timer": 0,
            "lost_counter": 0,
            "owner": team_id,
            "radius": hive_radius_pb,
            "bleed_damage_accumulator_food": 0.0,
            "bleed_damage_accumulator_health": 0.0,
            "max_food_capacity_at_destruction": self.metadata.get('hive_max_food', 100.0) # Store this for core drop
        }

        team_overrides = self.team_parameter_overrides.get(str(team_id), {}) if self.team_parameter_overrides else {}
        self.hives[team_id]["health"] = team_overrides.get('hive_max_health', self.hives[team_id]["health"])
        self.hives[team_id]["food_store"] = team_overrides.get('hive_max_food', self.hives[team_id]["food_store"])
        self.hives[team_id]["max_food_capacity_at_destruction"] = self.hives[team_id]["food_store"]

        self.hive_id_counter += 1
        collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=hive_radius_pb, physicsClientId=self.physicsClient)
        
        # Use team_configs if available, otherwise fall back to TEAM_COLORS
        if self.team_configs and team_id < len(self.team_configs):
            # PyBullet expects colors normalized to 0-1 range
            rgba = [c / 255.0 for c in self.team_configs[team_id].get("color", [128, 128, 128, 255])]
        else:
            # Fallback to constants, which are already 0-255
            rgba_255 = TEAM_COLORS.get(team_id, [128, 128, 128, 255])
            rgba = [c / 255.0 for c in rgba_255]
            
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=hive_radius_pb, rgbaColor=rgba, physicsClientId=self.physicsClient)
        body_id = p.createMultiBody(
            baseMass=0, 
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[pos[0], pos[1], hive_radius_pb], 
            physicsClientId=self.physicsClient
        )
        self.hive_body_ids[team_id] = body_id
        p.setCollisionFilterGroupMask(body_id, -1, COLLISION_GROUP_HIVE,
                                      COLLISION_GROUP_AGENT | COLLISION_GROUP_RESOURCE,
                                      physicsClientId=self.physicsClient)
        return True

    def spawn_hives(self):
        """
        Spawns `self.num_teams` hives in a circular pattern near the center of the arena.

        This method calculates positions for each team's hive, ensuring a minimum
        distance between them. It adds jitter to the positions to create a less
        uniform layout. It calls `_create_hive_instance` to perform the actual
        PyBullet object creation for each hive.
        """
        self.hives = {} # Clear existing hives
        self.hive_body_ids = {}
        spawned_positions = []
        center_x, center_y = self.width / 2, self.height / 2
        radius = min(self.width, self.height) * self.hive_spawn_radius_factor
        angle_step = 2 * math.pi / self.num_teams if self.num_teams > 0 else 0

        print(f"Spawning {self.num_teams} hives...")
        for team in range(self.num_teams):
            valid_spawn = False
            attempts = 0
            # Calculate target position on polygon
            target_angle = team * angle_step
            target_pos = np.array([
                center_x + radius * math.cos(target_angle),
                center_y + radius * math.sin(target_angle)
            ])

            # Add some jitter and check distance
            while not valid_spawn and attempts < 100:
                # RANDOMIZATION: Use random angle instead of fixed polygon angle
                random_angle = np.random.uniform(0, 2 * math.pi)
                jitter = np.random.uniform(0, radius) # Random distance from center
                
                pos_candidate = np.array([
                    center_x + jitter * math.cos(random_angle),
                    center_y + jitter * math.sin(random_angle)
                ])

                # Clamp position
                pos_candidate[0] = np.clip(pos_candidate[0], 50, self.width - 50)
                pos_candidate[1] = np.clip(pos_candidate[1], 50, self.height - 50)

                if all(distance(pos_candidate, sp) >= self.hive_min_distance for sp in spawned_positions):
                    if self._create_hive_instance(team, pos_candidate, spawned_positions):
                        valid_spawn = True
                attempts += 1

            if not valid_spawn:
                # Fallback: Use target position directly, check distance again
                print(f"Warning: Could not find jittered pos for T{team}, trying target polygon point.")
                if all(distance(target_pos, sp) >= self.hive_min_distance for sp in spawned_positions):
                     if self._create_hive_instance(team, target_pos, spawned_positions):
                         valid_spawn = True
                else:
                    # Critical error if even the target position fails
                    raise ValueError(f"Could not spawn hive for team {team} even at target polygon position after {attempts} attempts.")

        if len(self.hives) != self.num_teams:
            print(f"FATAL WARNING: Expected {self.num_teams} hives, but created {len(self.hives)}. Check spawn_hives logic.")


class SpawnHelpers (nn.Module):
    """Provides utility methods for finding safe spawn locations."""
    def _find_safe_spawn_location_near(self, target_pos: np.ndarray, search_radius: float, num_candidates: int = 25) -> Optional[np.ndarray]:
        """
        Finds a safe, unoccupied 2D position near a target position using a vectorized GPU approach.

        This method generates multiple random candidate positions around a target point. It then
        performs a single, batched distance calculation on the GPU against all existing
        entities in the simulation to find a candidate that does not collide with anything.
        This is significantly more efficient than checking one position at a time.

        Args:
            target_pos (np.ndarray): The desired [x, y] spawn location.
            search_radius (float): The minimum clearance required around the spawn point (i.e., the radius of the object to be spawned).
            num_candidates (int): The number of random positions to evaluate in parallel.

        Returns:
            Optional[np.ndarray]: A safe [x, y] position as a NumPy array, or None if no safe spot is found
                                  after checking all candidates.
        """
        if self.current_step_all_pos_t is None or self.current_step_all_radii_t is None:
            # Fallback to a random position if global tensors aren't ready (should not happen in normal operation)
            return np.array([
                np.random.uniform(search_radius, self.width - search_radius),
                np.random.uniform(search_radius, self.height - search_radius)
            ])

        # 1. Generate N candidates around the target position on the GPU
        angles = torch.rand(num_candidates, device=self.device) * (2 * math.pi)
        # Uniformly sample radius from 0 to a max search distance (e.g., 3x search_radius)
        radii_sample = torch.sqrt(torch.rand(num_candidates, device=self.device)) * (search_radius * 3)
        
        offsets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * radii_sample.unsqueeze(1)
        candidates_pos_t = torch.from_numpy(target_pos).to(self.device, dtype=torch.float32) + offsets

        # 2. Get all current entity positions and radii from the global tensors
        all_entity_pos_t = self.current_step_all_pos_t
        all_entity_radii_t = self.current_step_all_radii_t

        if all_entity_pos_t.numel() == 0:
            # No entities exist, any candidate is safe. Return the first one.
            safe_pos_clamped = torch.clamp(candidates_pos_t[0], search_radius, self.width - search_radius)
            return safe_pos_clamped.cpu().numpy()

        # 3. Perform batched collision check on GPU
        # Calculate pairwise distances between N candidates and M entities -> shape (N, M)
        dist_matrix_sq = torch.cdist(candidates_pos_t, all_entity_pos_t).pow(2)

        # Calculate required separation distance squared for each pair -> shape (N, M)
        required_separation_sq = (search_radius + all_entity_radii_t.unsqueeze(0)).pow(2)

        # A candidate is "colliding" with an entity if dist^2 < required_sep^2
        collision_matrix = dist_matrix_sq < required_separation_sq

        # A candidate is "unsafe" if it collides with ANY entity
        is_candidate_unsafe = torch.any(collision_matrix, dim=1)
        
        # Find the first safe candidate, if one exists
        safe_indices = torch.where(~is_candidate_unsafe)[0]
        if safe_indices.numel() > 0:
            best_candidate_pos = candidates_pos_t[safe_indices[0]]
            # Clamp to be within arena bounds before returning
            best_candidate_pos_clamped = torch.clamp(best_candidate_pos, search_radius, self.width - search_radius)
            return best_candidate_pos_clamped.cpu().numpy()

        return None

class ResourceSpawn (SpawnHelpers):
    """Handles the spawning and creation of resource entities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_id_counter = 0
    
    def _spawn_resource_at_location(self, target_pos: np.ndarray, size: float, cooperative: bool):
        """
        Spawns a single resource of a specific size at a safe position near a target location.

        This is often used for specific scenarios, like dropping a resource when a hive is destroyed.
        It uses `_find_safe_spawn_location_near` to ensure the resource doesn't spawn inside
        another object.

        Args:
            target_pos (np.ndarray): The desired [x, y] location to spawn the resource near.
            size (float): The conceptual size of the resource, which influences its properties.
            cooperative (bool): Whether the resource requires multiple agents to carry.

        Returns:
            Optional[dict]: The created resource dictionary, or None if spawning failed.
        """
        # --- Find a safe spawn position ---
        # Calculate radius first to know how much clearance we need
        min_size = self.metadata.get('resource_min_size', RESOURCE_MIN_SIZE)
        max_size = self.metadata.get('resource_max_size', RESOURCE_MAX_SIZE)
        size_range = max(1e-6, max_size - min_size)
        size_fraction = np.clip((size - min_size) / size_range, 0.0, 1.0)
        dynamic_radius_pb = self.min_resource_radius_pb + size_fraction * (self.max_resource_radius_pb - self.min_resource_radius_pb)
        
        safe_pos = self._find_safe_spawn_location_near(target_pos, search_radius=dynamic_radius_pb + 2.0)
        
        if safe_pos is None:
            # --- DIAGNOSTIC PRINT ---
            print("!!! SPAWN FAILED: Could not find a safe location.")
            return None

        # --- If location is found, proceed with creation ---
        resource_mass = self.resource_base_mass + size_fraction * self.resource_mass_scale_factor
        required_agents = max(1, int(np.ceil(size / 4.0))) if cooperative else 1

        try:
            collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=dynamic_radius_pb, physicsClientId=self.physicsClient)
            resource_color = [1, 0.65, 0, 1] if cooperative else [0.0, 0.8, 0.0, 1.0] # Bright green for drops
            visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=dynamic_radius_pb, rgbaColor=resource_color, physicsClientId=self.physicsClient)
            body_id = p.createMultiBody(
                baseMass=resource_mass,
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=[safe_pos[0], safe_pos[1], dynamic_radius_pb + 0.1],
                physicsClientId=self.physicsClient
            )
            # ... (rest of the pybullet setup for the resource)
            p.changeDynamics(body_id, -1,
                    linearDamping=self.pb_res_damping_static, angularDamping=self.pb_res_damping_static,
                    lateralFriction=self.pb_res_friction_static, restitution=self.pb_res_restitution,
                    physicsClientId=self.physicsClient)
            p.setCollisionFilterGroupMask(body_id, -1, COLLISION_GROUP_RESOURCE,
                                          COLLISION_GROUP_AGENT | COLLISION_GROUP_HIVE | COLLISION_GROUP_GROUND | COLLISION_GROUP_OBSTACLE,
                                          physicsClientId=self.physicsClient)
        except p.error as e:
            print(f"!!! SPAWN FAILED: PyBullet error during body creation: {e}")
            return None

        resource = {
            "id": self.resource_id_counter, "pos": safe_pos, "size": size,
            "radius_pb": dynamic_radius_pb, "cooperative": cooperative, "required_agents": required_agents,
            "body_id": body_id, "delivered": False, "carriers": [], "carrier_constraints": {},
            "attached": False, "pickup_cooldown": 0, "food_value": size * 5.0
        }
    
        self.resource_id_counter += 1
        self.resources.append(resource)
        
        # --- DIAGNOSTIC PRINT ---
        print(f"+++ SPAWN SUCCESS: Created resource {resource['id']} at {safe_pos.round(1)}")
        return resource
    
    
    def spawn_resource(self):
        """
        Spawns a single new resource at a random valid location in the arena.

        This method determines the resource's properties (like size and whether it's
        cooperative) and then iteratively searches for a random position that is not
        too close to hives, obstacles, or other resources. Once a valid spot is found,
        it creates the PyBullet body and the corresponding resource dictionary.

        Returns:
            Optional[dict]: The created resource dictionary, or None if a valid spawn
                            location could not be found.
        """
        # --- Determine Size ---
        cooperative = (np.random.rand() < self.coop_resource_probability)
        # Use metadata to get min/max conceptual size for robustness
        min_size = self.metadata.get('resource_min_size', RESOURCE_MIN_SIZE)
        max_size = self.metadata.get('resource_max_size', RESOURCE_MAX_SIZE)

        if cooperative:
            # Ensure cooperative resources are distinctly larger than min_size
            min_coop_size = max(min_size + 1.0, (min_size + max_size) / 2.5) # e.g., start coop size near midpoint
            size = np.random.uniform(min_coop_size, max_size)
            required_agents = max(1, int(np.ceil(size / 3.0)))
        else:
            size = min_size # Non-coop always min size
            required_agents = 1

        # --- Calculate Dynamic Radius based on Size ---
        # Use instance attributes for min/max physical radius
        min_radius_pb = self.min_resource_radius_pb
        max_radius_pb = self.max_resource_radius_pb

        size_range = max(1e-6, max_size - min_size)
        size_fraction = np.clip((size - min_size) / size_range, 0.0, 1.0)

        if cooperative:
            current_min_radius_pb = self.coop_min_resource_radius_pb
            current_max_radius_pb = self.coop_max_resource_radius_pb
        else:
            current_min_radius_pb = self.min_resource_radius_pb
            current_max_radius_pb = self.max_resource_radius_pb

        dynamic_radius_pb = current_min_radius_pb + size_fraction * (current_max_radius_pb - current_min_radius_pb)
        # Ensure it doesn't exceed the specific max for its type after calculation
        dynamic_radius_pb = min(dynamic_radius_pb, current_max_radius_pb)
        # Ensure it's not smaller than the specific min for its type (should be handled by formula but good for safety)
        dynamic_radius_pb = max(dynamic_radius_pb, current_min_radius_pb)
        valid = False
        attempts = 0
        pos = None
        while not valid and attempts < 100:
            pos_candidate = np.array([
                np.random.uniform(dynamic_radius_pb + 5, self.width - dynamic_radius_pb - 5),
                np.random.uniform(dynamic_radius_pb + 5, self.height - dynamic_radius_pb - 5)
            ])
            # Check distances using dynamic_radius_pb
            too_close_hive = any(distance(pos_candidate, hive["pos"]) < (25 + dynamic_radius_pb + self.resource_hive_buffer)
                                 for hive in self.hives.values()) if self.hives else False
            too_close_obstacle = any(distance(pos_candidate, obs["pos"]) < (obs["radius"] + dynamic_radius_pb + self.resource_obstacle_buffer)
                                     for obs in self.obstacles) if self.obstacles else False
            too_close_other_resource = False
            if self.resources: # Only check if other resources exist
                for other_res in self.resources:
                    # Ensure other_res is valid and has a position before checking
                    if other_res and other_res.get('pos') is not None and other_res.get('body_id') is not None:
                        required_dist_res_sq = (other_res['radius_pb'] + dynamic_radius_pb + 5.0)**2 # 5.0 buffer
                        if np.sum((pos_candidate - other_res['pos'])**2) < required_dist_res_sq:
                            too_close_other_resource = True
                            break

            if not too_close_hive and not too_close_obstacle and not too_close_other_resource: # Add new check
                pos = pos_candidate
                valid = True
            attempts += 1

        if not valid:
            print("Warning: Could not find a valid resource spawn position after 100 attempts; skipping spawn.")
            return None

        # --- Create PyBullet Resource Body (using dynamic_radius_pb) ---
        size_fraction_for_mass = np.clip((size - min_size) / size_range, 0.0, 1.0)
        resource_mass = self.resource_base_mass + size_fraction_for_mass * self.resource_mass_scale_factor
        resource_mass = max(self.resource_base_mass, resource_mass) # Ensure minimum mass
        collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=dynamic_radius_pb, physicsClientId=self.physicsClient)
        resource_color = [1, 0.65, 0, 1] if cooperative else [0, 1, 0, 1]
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=dynamic_radius_pb, rgbaColor=resource_color, physicsClientId=self.physicsClient)
        body_id = p.createMultiBody(
            baseMass=resource_mass,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[pos[0], pos[1], dynamic_radius_pb + 0.05], # Ensure this is +0.01 or similar
            physicsClientId=self.physicsClient
        )
       
        p.changeDynamics(body_id, -1,
                # Mass is set by createMultiBody
                linearDamping=self.pb_res_damping_static,    # This should be relatively HIGH for a free resource
                angularDamping=self.pb_res_damping_static,   # Also HIGH
                lateralFriction=self.pb_res_friction_static, # Ensure this is high enough (e.g., 0.8-1.0)
                spinningFriction=0.1, # Can be higher if they spin too much when free
                rollingFriction=0.1,  # Can be higher if they roll too much when free
                restitution=self.pb_res_restitution,       # Low restitution (e.g., 0.0 or 0.1) to prevent bouncing
                physicsClientId=self.physicsClient)
        # The setCollisionFilterGroupMask line remains the same after this
        p.setCollisionFilterGroupMask(body_id, -1,
                                      COLLISION_GROUP_RESOURCE, # Resource's Group
                                      # Collide with: Agents, Hives, Ground, Obstacles
                                      COLLISION_GROUP_AGENT | COLLISION_GROUP_HIVE | COLLISION_GROUP_GROUND | COLLISION_GROUP_OBSTACLE,
                                      physicsClientId=self.physicsClient)
        # --- Create Resource Dictionary (Store dynamic_radius_pb) ---
        resource = {
            "id": self.resource_id_counter,
            "pos": pos,
            "size": size,
            "radius_pb": dynamic_radius_pb,
            "cooperative": cooperative,
            "required_agents": required_agents,
            "body_id": body_id,
            "delivered": False,
            "carriers": [],
            "carrier_constraints": {}, # Correctly initialized
            "attached": False,
            "pickup_cooldown": 0,
            "food_value": size * 5.0
        }
        self.resource_id_counter += 1
        return resource

    def _cleanup_and_respawn_resources(self):
        """
        Removes delivered resources and spawns new ones to maintain a target density.

        This method is called once per simulation step. It first filters out any
        resources that have been marked as 'delivered'. Then, it calculates how many
        new resources are needed to reach the `ACTIVE_RESOURCE_LIMIT` and calls
        `spawn_resource` to create them.
        """
        self.resources=[r for r in self.resources if not r.get('delivered')]
        num_to_spawn=min(ACTIVE_RESOURCE_LIMIT-len(self.resources), self.num_resources_config-self.resource_id_counter)
        for _ in range(num_to_spawn):
            if new_res:=self.spawn_resource(): self.resources.append(new_res)
        for res in self.resources:
            if res.get("pickup_cooldown",0)>0: res["pickup_cooldown"]-=1


class ObstacleSpawn (nn.Module):
    """Handles the spawning and creation of static obstacle entities."""
    def init_obstacles_pybullet(self):
        """
        Initializes and spawns all obstacles for the environment at the start of an episode.

        It iteratively places `self.num_obstacles_config` obstacles at random positions,
        ensuring they do not overlap with hives or other previously placed obstacles.
        Obstacles are static (mass=0) and are represented as boxes.
        """
        self.obstacles = [] # Clear existing
        # Remove previously created obstacles from simulation if any
        # Note: This requires storing obstacle body IDs from previous resets if not clearing simulation fully.
        # Assuming full reset, so no need to remove old bodies here.

        print(f"Spawning {self.num_obstacles_config} obstacles...")
        
        for i in range(self.num_obstacles_config):
            max_attempts = 100
            attempts = 0
            placed = False
            while attempts < max_attempts:
                radius = np.random.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX)
                # Spawn within arena, away from edges
                pos = np.random.uniform(radius + 10, self.width - radius - 10, size=2)

                # Check distance to ALL hives
                too_close_to_hive = False
                if self.hives: # Only check if hives exist
                    too_close_to_hive = any(distance(pos, hive_data["pos"]) < (self.obstacle_hive_buffer + radius)
                                            for hive_data in self.hives.values())

                # Check distance to previously placed obstacles
                too_close_to_obstacle = False
                if self.obstacles:
                     too_close_to_obstacle = any(distance(pos, obs['pos']) < (obs['radius'] + radius + 10) # Add buffer between obstacles
                                                for obs in self.obstacles)

                if not too_close_to_hive and not too_close_to_obstacle:
                    # Create obstacle body
                    collisionShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius, radius, OBSTACLE_HALF_HEIGHT], physicsClientId=self.physicsClient)
                    visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius, radius, OBSTACLE_HALF_HEIGHT], rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.physicsClient)
                    body_id = p.createMultiBody(baseMass=0, # Static
                                                baseCollisionShapeIndex=collisionShapeId,
                                                baseVisualShapeIndex=visualShapeId,
                                                basePosition=[pos[0], pos[1], OBSTACLE_HALF_HEIGHT], # Place on ground
                                                physicsClientId=self.physicsClient)
                    # Set collision filter (e.g., OBSTACLE group, collides with AGENTs)
                    p.setCollisionFilterGroupMask(body_id, -1,
                                                  COLLISION_GROUP_OBSTACLE, # Obstacle's Group
                                                  # Obstacles collide with Agents and Resources
                                                  COLLISION_GROUP_AGENT | COLLISION_GROUP_RESOURCE,
                                                  physicsClientId=self.physicsClient)
                    # Set friction etc.
                    p.changeDynamics(body_id, -1, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0, restitution=0.1, physicsClientId=self.physicsClient)

                    self.obstacles.append({
                        'id': self.obstacle_id_counter, 
                        'pos': pos, 
                        'radius': radius, 
                        'body_id': body_id
                    })
                    self.obstacle_id_counter += 1 

                    placed = True
                    break # Exit attempt loop
                attempts += 1
            if not placed:
                print(f"Warning: Could not place obstacle {i+1}/{self.num_obstacles_config} without interfering after {max_attempts} attempts.")

        print(f"  Successfully placed {len(self.obstacles)} obstacles.")

    
    def init_boundaries(self):
        """
        Creates static boundary walls around the perimeter of the arena.

        These walls are static PyBullet bodies (mass=0) that prevent agents and
        other dynamic objects from leaving the simulation area.
        """
        # --- Wall Creation ---
        thickness = 10 # Visual thickness
        wall_z = OBSTACLE_HALF_HEIGHT # Height matching obstacles
        wall_color = [0.7, 0.7, 0.7, 1] # Grey color

        # Left wall
        left_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, self.height / 2, wall_z], physicsClientId=self.physicsClient)
        left_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, self.height / 2, wall_z], rgbaColor=wall_color, physicsClientId=self.physicsClient)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=left_collision, baseVisualShapeIndex=left_visual, basePosition=[-thickness, self.height / 2, wall_z], physicsClientId=self.physicsClient)
        # Right wall
        right_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, self.height / 2, wall_z], physicsClientId=self.physicsClient)
        right_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, self.height / 2, wall_z], rgbaColor=wall_color, physicsClientId=self.physicsClient)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=right_collision, baseVisualShapeIndex=right_visual, basePosition=[self.width + thickness, self.height / 2, wall_z], physicsClientId=self.physicsClient)
        # Bottom wall
        bottom_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.width / 2, thickness, wall_z], physicsClientId=self.physicsClient)
        bottom_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.width / 2, thickness, wall_z], rgbaColor=wall_color, physicsClientId=self.physicsClient)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bottom_collision, baseVisualShapeIndex=bottom_visual, basePosition=[self.width / 2, -thickness, wall_z], physicsClientId=self.physicsClient)
        # Top wall
        top_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.width / 2, thickness, wall_z], physicsClientId=self.physicsClient)
        top_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[self.width / 2, thickness, wall_z], rgbaColor=wall_color, physicsClientId=self.physicsClient)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=top_collision, baseVisualShapeIndex=top_visual, basePosition=[self.width / 2, self.height + thickness, wall_z], physicsClientId=self.physicsClient)



class AgentSpawn (nn.Module):    
    """Handles the spawning, creation, and respawning of agent entities."""
    def init_agents_pybullet(self):
        """
        Initializes all agents for all teams at the beginning of an episode.

        This method iterates through each team and spawns the configured number of agents
        near their respective hives. It handles the randomization of agent properties
        (like radius, strength, and mass) based on the environment's configuration,
        creates the PyBullet bodies, and initializes the agent data dictionaries.
        """
        self.agents = [] # Clear existing
        self.agent_body_ids = []
        agent_global_id_counter = 0 # Unique ID across all agents

        print(f"Spawning {self.num_agents} agents for {self.num_teams} teams (Mass influenced by size & strength)...")
        if not self.hives:
            print("FATAL ERROR: Cannot spawn agents, no hives were created.")
            return

        base_max_energy = self.metadata.get('agent_max_energy', AGENT_MAX_ENERGY)
        base_max_health = self.metadata.get('agent_max_health', AGENT_MAX_HEALTH)
        
        # AGENT_BASE_STRENGTH is a global constant, AGENT_RADIUS is a global constant
        # self.pb_agent_mass will now act as the mass for a "standard" agent:
        # i.e., an agent with agent_radius_actual == AGENT_RADIUS
        # and agent_strength_actual == AGENT_BASE_STRENGTH.
        base_mass_standard_agent = self.pb_agent_mass # Fetched from __init__ args

        for team_id in range(self.num_teams):
            if team_id not in self.hives:
                 print(f"Warning: Hive for team {team_id} not found during agent spawning. Skipping agents for this team.")
                 continue

            hive_pos = self.hives[team_id]["pos"]
            for agent_local_idx in range(self.agent_counts_per_team[team_id]):
                hive_data = self.hives.get(team_id)
                if not hive_data: # Should not happen if checks are in place
                    print(f"FATAL: Missing hive data for team {team_id} during agent spawn.")
                    continue # Or raise error

                hive_pos_for_spawn = hive_data["pos"]
                # Get hive's actual physical radius (used in its PyBullet creation)
                hive_pb_radius_for_spawn = hive_data.get('radius', self.metadata.get('hive_radius_assumed', HIVE_RADIUS_ASSUMED))
                                
                pos = random_spawn_near_hive( # Or random_spawn_near_hive_annulus
                    hive_pos_for_spawn,
                    spawn_radius=self.agent_spawn_radius, # Your existing hive_spawn_radius
                    width=self.width,
                    height=self.height,
                    hive_physical_radius=hive_pb_radius_for_spawn,
                    agent_min_radius_spawn=AGENT_RADIUS # Use the global constant as a safe buffer
                )
                
                team_overrides = self.team_parameter_overrides.get(str(team_id), {}) if self.team_parameter_overrides else {}

                def get_team_specific_param(param_name, default_value):
                    return team_overrides.get(param_name, default_value)
                
                def get_randomized_value(param_name, base_value_config):
                    """
                    Gets a randomized value for a parameter, respecting team overrides first,
                    then falling back to global randomization settings.
                    """
                    team_override = get_team_specific_param(param_name, None)
                    
                    if team_override is not None and isinstance(team_override, dict):
                        # Use the team-specific override
                        base_value = team_override.get('base', base_value_config)
                        # Use the team's rand factor, or the global one if not specified in the override
                        rand_factor = team_override.get('rand', self.agent_randomization_factors.get(param_name, {}).get('rand', 0.0))
                    else:
                        # No team override, use global defaults
                        base_value = self.agent_randomization_factors.get(param_name, {}).get('base', base_value_config)
                        # Use the global randomization factor passed to the env
                        rand_factor = self.agent_randomization_factors.get(param_name, {}).get('rand', 0.0)

                    return base_value * np.random.uniform(1 - rand_factor, 1 + rand_factor)

                # Agent's individual randomized physical properties
                agent_radius_actual = get_randomized_value('agent_radius', AGENT_RADIUS)
                agent_strength_actual = get_randomized_value('agent_base_strength', AGENT_BASE_STRENGTH)


                # --- Combined Mass Calculation (Option 3) ---
                # Volume factor based on actual radius vs standard AGENT_RADIUS
                # Assuming AGENT_RADIUS is not zero
                volume_factor = (agent_radius_actual / AGENT_RADIUS)**3 if AGENT_RADIUS > 0 else 1.0

                # Strength factor based on actual strength vs standard AGENT_BASE_STRENGTH
                # Assuming AGENT_BASE_STRENGTH is not zero
                strength_influence_factor = 1.0
                if AGENT_BASE_STRENGTH > 0:
                    # Example: strength has a 50% influence on density on top of volume based mass
                    # A 2x stronger agent (strength_factor=2) would have its mass (already scaled by volume)
                    # further multiplied by (1 + (2-1)*0.5) = 1.5
                    # A 0.5x strength agent (strength_factor=0.5) would be multiplied by (1 + (0.5-1)*0.5) = 0.75
                    strength_factor_raw = agent_strength_actual / AGENT_BASE_STRENGTH
                    strength_influence_on_density = self.agent_mass_strength_influence # How much relative strength affects "density"
                    strength_influence_factor = 1.0 + (strength_factor_raw - 1.0) * strength_influence_on_density
                    strength_influence_factor = max(0.25, strength_influence_factor) # Clamp influence factor

                calculated_mass = base_mass_standard_agent * volume_factor * strength_influence_factor
                # Clamp dynamic mass to 0.8x - 1.2x of the standard base mass
                mass_low = self.agent_mass_min_factor * base_mass_standard_agent
                mass_high = self.agent_mass_max_factor * base_mass_standard_agent
                calculated_mass = float(np.clip(calculated_mass, mass_low, mass_high))
                # --- End Combined Mass Calculation ---

                collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=agent_radius_actual, physicsClientId=self.physicsClient)
                
                # Use team_configs if available, otherwise fall back to TEAM_COLORS
                if self.team_configs and team_id < len(self.team_configs):
                    rgba = [c / 255.0 for c in self.team_configs[team_id].get("color", [128, 128, 128, 255])]
                else:
                    rgba_255 = TEAM_COLORS.get(team_id, [128, 128, 128, 255])
                    rgba = [c / 255.0 for c in rgba_255]

                visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=agent_radius_actual, rgbaColor=rgba, physicsClientId=self.physicsClient)
                
                body_id = p.createMultiBody(baseMass=calculated_mass, # USE THE CALCULATED MASS
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[pos[0], pos[1], agent_radius_actual], # Z position is agent's radius
                                            physicsClientId=self.physicsClient)

                # --- Create Observation Radius Visualization ---
                # We will use p.addUserDebugLine in _update_pybullet_visualizations to draw high-quality circles.
                # We store the intended radius here so the updater knows what to draw.
                obs_radius_viz_active = self.render_mode is not None
                obs_radius_color = [*rgba[:3], 1.0] # Full opacity color for lines
                # --- End Visualization ---

                p.changeDynamics(body_id, -1,
                                linearDamping=self.pb_agent_linear_damping,
                                angularDamping=self.pb_agent_angular_damping,
                                lateralFriction=self.pb_agent_lateral_friction,
                                spinningFriction=self.pb_agent_spinning_friction,
                                rollingFriction=self.pb_agent_rolling_friction,
                                restitution=self.pb_agent_restitution,
                                physicsClientId=self.physicsClient)
                
                # Increase velocity limits to allow higher speeds (default is ~100 units/second)
                # This enables fast-paced gameplay with 30 FPS decision making
                p.changeDynamics(body_id, -1,
                                maxJointVelocity=1000,  # Allow up to 1000 units/second
                                physicsClientId=self.physicsClient)
                p.setCollisionFilterGroupMask(body_id, -1, COLLISION_GROUP_AGENT,
                                              COLLISION_GROUP_RESOURCE | COLLISION_GROUP_OBSTACLE | COLLISION_GROUP_HIVE | COLLISION_GROUP_AGENT | COLLISION_GROUP_GROUND,
                                              physicsClientId=self.physicsClient)

                agent_max_energy_actual = get_randomized_value('agent_max_energy', base_max_energy)
                agent_max_health_actual = get_randomized_value('agent_max_health', base_max_health)
                
                # For non-randomizable params, just check for a direct override
                agent_individual_base_speed = get_randomized_value('bee_speed', self.bee_speed_config)
                
                obs_radius_actual = get_randomized_value('sensing_range_fraction', self.obs_radius / self.d_max) * self.d_max

                if self.debug:
                    print(f"Agent {agent_global_id_counter} (Team {team_id}) params: "
                          f"radius={agent_radius_actual:.2f}, "
                          f"strength={agent_strength_actual:.2f}, "
                          f"speed={agent_individual_base_speed:.2f}, "
                          f"obs_radius={obs_radius_actual:.2f}, "
                          f"max_energy={agent_max_energy_actual:.2f}, "
                          f"max_health={agent_max_health_actual:.2f}")

                agent = {
                    'id': agent_global_id_counter,
                    'team': team_id,
                    'body_id': body_id,
                    'obs_radius_viz_ids': [], # Changed to list for debug lines
                    'obs_radius_viz_active': obs_radius_viz_active, # Added for visualization control
                    'obs_radius_color': obs_radius_color, # Added for visualization control
                    'has_resource': False,
                    'resource_obj': None,
                    'alive': True,
                    'cooldown': 0,
                    'speed': agent_individual_base_speed,
                    'agent_radius': agent_radius_actual,
                    'obs_radius': obs_radius_actual,
                    'strength': agent_strength_actual,
                    'energy': agent_max_energy_actual,
                    'max_energy': agent_max_energy_actual,
                    'health': agent_max_health_actual,
                    'max_health': agent_max_health_actual,
                    'slowed_timer': 0,
                    'pos': pos.copy(),
                    'vel': np.zeros(2),
                    'mass': calculated_mass,
                    'is_grappling': False,
                    'grappled_agent_id': None,
                    'grapple_constraint_id': None,
                    'grapple_last_set_force': 0.0, 
                    'grapple_momentum_bonus': 0.0, 
                    'is_grappled': False,
                    'applied_torque': 0.0 
                }

                self.agents.append(agent)
                self.agent_body_ids.append(body_id)
                agent_global_id_counter += 1

        if len(self.agents) != self.num_agents:
            print(f"FATAL WARNING: Expected {self.num_agents} agents, but created {len(self.agents)}. Check init_agents_pybullet logic.")
        else:
             print(f"  Successfully spawned {len(self.agents)} agents with dynamic mass calculation.")


    def respawn_agent(self, agent):
        """
        Respawns a single agent after it has been neutralized.

        This method resets the agent's state (health, energy, etc.), finds a new safe
        spawn location near its team's hive, and moves its PyBullet body to that
        location with zero velocity. It also handles the cleanup of any constraints
        (like grapples or resource carrying) that were active when the agent was neutralized.

        Args:
            agent (dict): The data dictionary of the agent to be respawned.
        """
        # print(f"Respawning Agent {agent['id']} (Team {agent['team']})")
        # Release resource if carrying
        if agent.get("has_resource", False) and agent.get("resource_obj") is not None:
            res_obj = agent["resource_obj"]
            if res_obj and isinstance(res_obj, dict):
                 if agent["id"] in res_obj.get("carrier_constraints", {}):
                     cid = res_obj["carrier_constraints"].pop(agent["id"])
                     try: p.removeConstraint(cid, physicsClientId=self.physicsClient)
                     except p.error: pass
                 res_obj["carriers"] = [c for c in res_obj.get("carriers", []) if isinstance(c, dict) and c.get('id') != agent['id']]

        agent["has_resource"] = False
        agent.pop("resource_obj", None)
        
        if agent.get('is_grappling') and agent.get('grapple_constraint_id') is not None:
             try: p.removeConstraint(agent['grapple_constraint_id'], physicsClientId=self.physicsClient)
             except p.error: pass
        agent['is_grappling'] = False
        agent['grappled_agent_id'] = None
        agent['grapple_constraint_id'] = None
        agent['grapple_last_set_force'] = 0.0 
        agent['grapple_momentum_bonus'] = 0.0 
        agent['is_grappled'] = False
        agent['alive'] = True
        agent['cooldown'] = 0
        agent['energy'] = agent["max_energy"]
        agent['health'] = agent["max_health"]
        agent['slowed_timer'] = 0

        # --- MODIFIED SPAWN LOGIC ---
        hive_data = self.hives.get(agent['team'])
        if not hive_data or hive_data.get('pos') is None:
            hive_pos_for_spawn = np.array([self.width / 2, self.height / 2]) # Fallback
            hive_pb_radius_for_spawn = self.metadata.get('hive_radius_assumed', HIVE_RADIUS_ASSUMED)
            print(f"Warning: Hive for team {agent['team']} not found for respawn. Using center.")
        else:
            hive_pos_for_spawn = hive_data["pos"]
            hive_pb_radius_for_spawn = hive_data.get('radius', self.metadata.get('hive_radius_assumed', HIVE_RADIUS_ASSUMED))

        # Use the agent's own actual radius for a more precise buffer,
        # or AGENT_RADIUS if 'agent_radius' isn't reliably in the agent dict at this point for respawn.
        # Assuming agent dict still holds its 'agent_radius' from its last life.
        agent_radius_for_spawn_buffer = agent.get('agent_radius', AGENT_RADIUS)

        spawn_pos = random_spawn_near_hive( # Or random_spawn_near_hive_annulus
            hive_pos_for_spawn,
            spawn_radius=self.agent_spawn_radius, # Or a different respawn_radius if desired
            width=self.width,
            height=self.height,
            hive_physical_radius=hive_pb_radius_for_spawn,
            agent_min_radius_spawn=agent_radius_for_spawn_buffer
        )

        agent['pos'] = spawn_pos.copy()
        agent['vel'] = np.zeros(2)

        try:
            _, current_ori = p.getBasePositionAndOrientation(agent['body_id'], physicsClientId=self.physicsClient)
            p.resetBasePositionAndOrientation(
                agent['body_id'],
                [spawn_pos[0], spawn_pos[1], agent['agent_radius']], # agent['agent_radius'] is its actual physical radius
                current_ori,
                physicsClientId=self.physicsClient
            )
            p.resetBaseVelocity(agent['body_id'], linearVelocity=[0, 0, 0], angularVelocity=[0,0,0], physicsClientId=self.physicsClient)
        except p.error as e:
             print(f"Warning: PyBullet error resetting agent {agent['id']} during respawn: {e}")


def random_spawn_near_hive(hive_pos, spawn_radius=50, width=WIDTH, height=HEIGHT,
                        # ADD these parameters for clarity and flexibility
                        hive_physical_radius=HIVE_RADIUS_ASSUMED, # Get default from global or pass from env.metadata
                        agent_min_radius_spawn=AGENT_RADIUS # Smallest possible agent radius for buffer
                        ):
    """
    Calculates a random, safe spawn position for an agent near its hive.

    It generates a random point within a specified radius (`spawn_radius`) of the
    hive's center. It ensures the point is not inside the hive's own physical body
    and is clamped within the arena's boundaries.

    Args:
        hive_pos (np.ndarray): The [x, y] position of the hive.
        spawn_radius (float): The maximum distance from the hive center to spawn.
        width (int): The width of the arena for clamping.
        height (int): The height of the arena for clamping.
        hive_physical_radius (float): The actual radius of the hive's body in PyBullet.
        agent_min_radius_spawn (float): The radius of the agent being spawned, used as a buffer.

    Returns:
        np.ndarray: A valid and safe [x, y] spawn position.
    """
    hive_center = np.array(hive_pos[:2])
    min_dist_from_hive_center_sq = (hive_physical_radius + agent_min_radius_spawn + 1.0)**2 # +1 for a small buffer

    attempts = 0
    max_attempts = 100 # Prevent infinite loop

    while attempts < max_attempts:
        angle = np.random.uniform(0, 2 * math.pi)
        # Generate radius uniformly within the spawn_radius circle
        r_spawn = np.sqrt(np.random.uniform(0, spawn_radius**2))
        offset_x = r_spawn * math.cos(angle)
        offset_y = r_spawn * math.sin(angle)
        pos_candidate = hive_center + np.array([offset_x, offset_y])

        # Check distance from hive center
        dist_to_hive_sq = np.sum((pos_candidate - hive_center)**2)

        if dist_to_hive_sq >= min_dist_from_hive_center_sq:
            # Candidate is outside the exclusion zone, now clamp to arena
            pos_candidate[0] = np.clip(pos_candidate[0], agent_min_radius_spawn + 1, width - agent_min_radius_spawn - 1)
            pos_candidate[1] = np.clip(pos_candidate[1], agent_min_radius_spawn + 1, height - agent_min_radius_spawn - 1)
            return pos_candidate # Valid position found

        attempts += 1

    # Fallback if too many attempts: spawn at the edge of the spawn_radius, clamped
    print(f"Warning (random_spawn_near_hive): Could not find valid non-overlapping spawn after {max_attempts} attempts. Spawning at edge of spawn_radius.")
    angle = np.random.uniform(0, 2 * math.pi)
    offset_x = spawn_radius * math.cos(angle) # Use exact spawn_radius
    offset_y = spawn_radius * math.sin(angle)
    pos = hive_center + np.array([offset_x, offset_y])
    pos[0] = np.clip(pos[0], agent_min_radius_spawn + 1, width - agent_min_radius_spawn - 1)
    pos[1] = np.clip(pos[1], agent_min_radius_spawn + 1, height - agent_min_radius_spawn - 1)
    return pos

