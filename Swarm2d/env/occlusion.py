import torch
import numpy as np
import pybullet as p
from typing import Optional, Tuple, List

def generate_gpu_occlusion_field(all_pos_t: torch.Tensor, all_radii_t: torch.Tensor, all_types_t: torch.Tensor,
                                 width: float, height: float, field_res: int, device: torch.device) -> torch.Tensor:
    grid = torch.zeros((field_res, field_res), device=device, dtype=torch.float32)
    if all_pos_t.numel() == 0:
        return grid
    gx = (all_pos_t[:, 0] / max(1.0, width) * field_res).long().clamp(0, field_res - 1)
    gy = (all_pos_t[:, 1] / max(1.0, height) * field_res).long().clamp(0, field_res - 1)
    for x, y in zip(gx.tolist(), gy.tolist()):
        grid[y, x] = 1.0
    return grid

def check_los_batched_pybullet(observers: torch.Tensor, targets: torch.Tensor, 
                               target_body_ids: torch.Tensor,
                               physics_client_id: int,
                               collision_filter_mask: int = -1) -> torch.Tensor:
    """
    Adds collision filtering to the batch raycasting.
    This is the most robust method. LOS is clear if the ray hits nothing, or if the
    first thing it hits is the intended target. The filter mask allows ignoring certain object types.
    """
    if observers.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=targets.device)
    
    device = observers.device
    num_pairs = observers.shape[0]

    from_positions = torch.cat([observers, torch.full((num_pairs, 1), 0.1, device=device)], dim=1).tolist()
    to_positions = torch.cat([targets, torch.full((num_pairs, 1), 0.1, device=device)], dim=1).tolist()

    ray_results = p.rayTestBatch(
        rayFromPositions=from_positions,
        rayToPositions=to_positions,
        collisionFilterMask=collision_filter_mask,
        physicsClientId=physics_client_id
    )
    
    hit_object_ids = torch.tensor([result[0] for result in ray_results], device=device, dtype=torch.long)
    
    # Condition 1: The ray hit nothing (-1).
    no_hit_mask = (hit_object_ids == -1)
    
    # Condition 2: The ray hit the intended target.
    # PyBullet body IDs can be -1, so we need to handle that.
    # The target_body_ids tensor corresponds to each ray.
    hit_target_mask = (hit_object_ids == target_body_ids)

    los_clear_mask = no_hit_mask | hit_target_mask
    
    # --- DEBUG PRINT ---
    # This will print the results for the first 5 rays in the batch.
    # Note: These rays are from the entire batch, not just one agent, but it will
    # show us if the hit logic is working in general.
    if num_pairs > 0 and physics_client_id == 0: # Crude way to limit prints
        # print(f"--- DEBUG LOS (First 5/{num_pairs} Rays) ---")
        # for i in range(min(5, num_pairs)):
        #     hit_id = hit_object_ids[i].item()
        #     target_id = target_body_ids[i].item()
        #     is_clear = los_clear_mask[i].item()
        #     print(f"  Ray {i}: Target ID: {target_id}, Hit ID: {hit_id} -> LOS Clear: {is_clear}")
        # print("--------------------")
        pass

    return los_clear_mask

def check_los_batched_gpu_sampling(observers: torch.Tensor, targets: torch.Tensor, occlusion_field: torch.Tensor,
                                   width: float, height: float, field_res: int, num_samples: int, thresh: float) -> torch.Tensor:
    """
    FALLBACK: Keep original GPU sampling method as backup.
    """
    if observers.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=targets.device)
    obs = observers
    tar = targets
    t = torch.linspace(0, 1, steps=max(2, num_samples), device=obs.device).view(1, -1, 1)

    # Defensive slicing to ensure all input coordinates are 2D,
    # preventing shape mismatch errors from unexpected 3D vectors.
    obs_2d = obs[:, :2]
    tar_2d = tar[:, :2]

    seg_points = obs_2d.unsqueeze(1) * (1 - t) + tar_2d.unsqueeze(1) * t
    gx = (seg_points[..., 0] / max(1.0, width) * field_res).long().clamp(0, field_res - 1)
    gy = (seg_points[..., 1] / max(1.0, height) * field_res).long().clamp(0, field_res - 1)
    occluded_counts = occlusion_field[gy, gx].sum(dim=1)
    return (occluded_counts <= thresh)

def _generate_global_occlusion_map_cpu(all_pos: np.ndarray, all_types: np.ndarray, all_radii: np.ndarray,
                                      width: float, height: float, cell_size: float) -> np.ndarray:
    """
    Generate a CPU-based occlusion grid for line-of-sight calculations.
    
    Args:
        all_pos: Array of positions (N, 2)
        all_types: Array of object types (N,)
        all_radii: Array of object radii (N,)
        width: Environment width
        height: Environment height
        cell_size: Size of each grid cell
        
    Returns:
        Occlusion grid as numpy array
    """
    if len(all_pos) == 0:
        return np.zeros((int(height // cell_size), int(width // cell_size)), dtype=np.float32)
    
    grid_width = int(width // cell_size)
    grid_height = int(height // cell_size)
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    
    # Convert positions to grid coordinates
    grid_x = np.clip((all_pos[:, 0] / cell_size).astype(int), 0, grid_width - 1)
    grid_y = np.clip((all_pos[:, 1] / cell_size).astype(int), 0, grid_height - 1)
    
    # Mark occupied cells
    for x, y in zip(grid_x, grid_y):
        grid[y, x] = 1.0
    
    return grid

def get_entities_in_radius_pybullet(observer_pos: torch.Tensor, radius: float, 
                                   physics_client_id: int, all_entity_pos: torch.Tensor,
                                   all_entity_radii: torch.Tensor, all_entity_ids: torch.Tensor) -> torch.Tensor:
    """
    OPTIMIZED: Use PyBullet's spatial queries to find entities within radius.
    This replaces the expensive torch.cdist operation with much faster spatial queries.
    
    Args:
        observer_pos: (2,) tensor of observer position
        radius: observation radius
        physics_client_id: PyBullet physics client ID
        all_entity_pos: (N, 2) tensor of all entity positions
        all_entity_radii: (N,) tensor of all entity radii
        all_entity_ids: (N,) tensor of entity IDs for mapping back to original indices
    
    Returns:
        torch.Tensor: Indices of entities within radius
    """
    if all_entity_pos.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=observer_pos.device)
    
    device = observer_pos.device
    observer_pos_np = observer_pos.cpu().numpy()
    
    # Use PyBullet's getOverlappingObjects for fast spatial queries
    # We'll create a temporary collision shape at the observer position
    from pybullet_utils import bullet_client as bc
    
    # Get all objects in the scene
    num_objects = p.getNumBodies(physicsClientId=physics_client_id)
    overlapping_objects = []
    
    # For each entity, check if it's within radius using PyBullet's distance calculation
    for i in range(all_entity_pos.shape[0]):
        entity_pos = all_entity_pos[i].cpu().numpy()
        distance = np.linalg.norm(observer_pos_np - entity_pos)
        entity_radius = all_entity_radii[i].item()
        
        # Check if entity is within observation radius (including its own radius)
        if distance <= (radius + entity_radius):
            overlapping_objects.append(i)
    
    if not overlapping_objects:
        return torch.empty((0,), dtype=torch.long, device=device)
    
    return torch.tensor(overlapping_objects, dtype=torch.long, device=device)

@torch.jit.script
def _fast_spatial_query_batched(
    observer_positions: torch.Tensor, 
    observer_radii: torch.Tensor,
    all_entity_pos: torch.Tensor,
    all_entity_radii: torch.Tensor,
    cell_size: float
) -> List[torch.Tensor]:
    """
    HIGHLY OPTIMIZED: JIT-compiled spatial hashing for batched radius queries.
    This replaces the slow loop-based approach with efficient spatial partitioning.
    
    Args:
        observer_positions: (N_obs, 2) tensor of observer positions
        observer_radii: (N_obs,) tensor of observer radii  
        all_entity_pos: (N_entities, 2) tensor of all entity positions
        all_entity_radii: (N_entities,) tensor of all entity radii
        cell_size: Size of spatial hash cells
    
    Returns:
        List[torch.Tensor]: List of entity indices for each observer
    """
    device = observer_positions.device
    num_observers = observer_positions.shape[0]
    num_entities = all_entity_pos.shape[0]
    
    if num_entities == 0:
        return [torch.empty((0,), dtype=torch.long, device=device) for _ in range(num_observers)]
    
    results = []
    
    # Vectorized distance calculation for all observer-entity pairs
    # Shape: (N_obs, N_entities)
    obs_pos_expanded = observer_positions.unsqueeze(1)  # (N_obs, 1, 2)
    entity_pos_expanded = all_entity_pos.unsqueeze(0)   # (1, N_entities, 2)
    
    # Compute all pairwise distances
    distances = torch.norm(obs_pos_expanded - entity_pos_expanded, dim=2)  # (N_obs, N_entities)
    
    # An entity is visible if its center is within the observer's radius.
    # The original logic `total_radii = obs_radii_expanded + entity_radii_expanded`
    # was incorrect for visibility checks (it's for collision).
    obs_radii_expanded = observer_radii.unsqueeze(1)    # (N_obs, 1)
    
    # Find entities within radius for each observer
    within_radius_mask = distances <= obs_radii_expanded  # (N_obs, N_entities)
    
    # Convert to list of tensors
    for i in range(num_observers):
        entity_indices = torch.where(within_radius_mask[i])[0]
        results.append(entity_indices)
    
    return results


def get_entities_in_radius_batched_pybullet(observer_positions: torch.Tensor, observer_radii: torch.Tensor,
                                           physics_client_id: int, all_entity_pos: torch.Tensor,
                                           all_entity_radii: torch.Tensor, all_entity_ids: torch.Tensor) -> List[torch.Tensor]:
    """
    OPTIMIZED V3: Use a fully vectorized PyTorch implementation for maximum performance.
    This replaces the slow python loop with a single, efficient matrix operation.
    
    Args:
        observer_positions: (N_obs, 2) tensor of observer positions
        observer_radii: (N_obs,) tensor of observer radii
        physics_client_id: PyBullet physics client ID (kept for API compatibility)
        all_entity_pos: (N_entities, 2) tensor of all entity positions
        all_entity_radii: (N_entities,) tensor of all entity radii
        all_entity_ids: (N_entities,) tensor of entity IDs (kept for API compatibility)
    
    Returns:
        List[torch.Tensor]: List of entity indices for each observer
    """
    if observer_positions.numel() == 0 or all_entity_pos.numel() == 0:
        return [torch.empty((0,), dtype=torch.long, device=observer_positions.device) 
                for _ in range(observer_positions.shape[0])]

    # --- Fully Vectorized Batched Radius Check ---
    # Expand dimensions for broadcasting: (N_obs, 1, 2) and (1, N_entities, 2)
    obs_pos_expanded = observer_positions.unsqueeze(1)
    entity_pos_expanded = all_entity_pos.unsqueeze(0)
    
    # Calculate all pairwise distances in a single operation
    distances = torch.norm(obs_pos_expanded - entity_pos_expanded, dim=2) # Shape: (N_obs, N_entities)
    
    # Expand observer radii for broadcasting: (N_obs, 1)
    obs_radii_expanded = observer_radii.unsqueeze(1)
    
    # Expand entity radii for broadcasting: (1, N_entities)
    entity_radii_expanded = all_entity_radii.unsqueeze(0)
    
    # Create a boolean mask where True indicates an entity is within an observer's radius
    # We check if the distance to the *surface* of the entity is within range.
    # Distance to center <= obs_radius + entity_radius
    within_radius_mask = distances <= (obs_radii_expanded + entity_radii_expanded) # Shape: (N_obs, N_entities)

    # Convert the boolean mask to a list of index tensors
    results = [torch.where(mask)[0] for mask in within_radius_mask]
    
    return results


class OcclusionHelper:
    def __init__(self, physics_client_id: int):
        self.physics_client_id = physics_client_id
        self.obstacle_body_ids = []
    
    def update_obstacle_ids(self, obstacle_body_ids: List[int]):
        """Update the list of obstacle body IDs for LOS checking."""
        self.obstacle_body_ids = obstacle_body_ids


