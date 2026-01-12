
import torch
from torch_geometric.utils import scatter

# --- Scatter Operations Helper ---
def apply_scatter(data, mask, indices, dim_size, reduce_op='mean'):
    """Applies scatter operation safely."""
    if device is None:
        device = data.device if isinstance(data, torch.Tensor) else 'cpu'
    if mask.sum() == 0:
        return torch.zeros(dim_size, device=device)
    data_masked = data[mask].float()
    indices_masked = indices[mask]
    indices_masked = indices_masked.clamp(0, dim_size - 1)
    if data_masked.numel() == 0:
        return torch.zeros(dim_size, device=device)
    return scatter(data_masked, indices_masked, dim=0, dim_size=dim_size, reduce=reduce_op)

        


def safe_scatter_add(data, indices, dim_size, device):
    if data.numel() == 0 or indices.numel() == 0: return torch.zeros(dim_size, device=device, dtype=torch.float32)
    if indices.max() >= dim_size or indices.min() < 0:
        # print(f"Warning: Clamping scatter indices (add). Max idx: {indices.max()}, Min idx: {indices.min()}, Dim size: {dim_size}")
        indices = indices.clamp(0, dim_size - 1)
    # --- Ensure data is float before scattering ---
    return scatter(data.float(), indices, dim=0, dim_size=dim_size, reduce='add')

def safe_scatter_max(data, indices, dim_size, device):
    if data.numel() == 0 or indices.numel() == 0: return torch.zeros(dim_size, device=device, dtype=torch.float32)
    if indices.max() >= dim_size or indices.min() < 0:
        # print(f"Warning: Clamping scatter indices (max). Max idx: {indices.max()}, Min idx: {indices.min()}, Dim size: {dim_size}")
        indices = indices.clamp(0, dim_size - 1)
    # --- Ensure data is float before scattering ---
    result = scatter(data.float(), indices, dim=0, dim_size=dim_size, reduce='max')
    # Replace -inf that occurs when a destination index has no source indices
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) # Safest way to handle potential -inf
    return result

def safe_scatter_mean(data, indices, dim_size, device):
    if data.numel() == 0 or indices.numel() == 0: return torch.zeros(dim_size, device=device, dtype=torch.float32)
    if indices.max() >= dim_size or indices.min() < 0:
        # print(f"Warning: Clamping scatter indices (mean). Max idx: {indices.max()}, Min idx: {indices.min()}, Dim size: {dim_size}")
        indices = indices.clamp(0, dim_size - 1)
    # --- Ensure data is float before scattering ---
    sums = scatter(data.float(), indices, dim=0, dim_size=dim_size, reduce='add')
    counts = scatter(torch.ones_like(data).float(), indices, dim=0, dim_size=dim_size, reduce='add')
    counts = counts.clamp(min=1e-6) # Clamp counts before division
    return sums / counts

def aggregate_grid_data_scatter(all_coords, all_chunks, all_last_updated, all_certainty,
                               critic_grid_size, critic_world_to_map_scale, current_step,
                               env_metadata, team_id):
    """
    Aggregate grid data using scatter operations for critic observations.
    
    Args:
        all_coords: Coordinates of all data points (N, 2)
        all_chunks: Data chunks to aggregate (N, channels)
        all_last_updated: Last update timestamps (N,)
        all_certainty: Certainty values (N,)
        critic_grid_size: Size of the critic grid
        critic_world_to_map_scale: Scale factor from world to map coordinates
        current_step: Current simulation step
        env_metadata: Environment metadata
        team_id: Team identifier
        
    Returns:
        Aggregated grid data (channels, grid_size, grid_size)
    """
    import torch
    from torch_geometric.utils import scatter
    
    device = all_coords.device
    num_channels = all_chunks.shape[1]
    
    # Convert world coordinates to grid coordinates
    grid_coords = (all_coords * critic_world_to_map_scale).long()
    grid_coords = grid_coords.clamp(0, critic_grid_size - 1)
    
    # Flatten 2D grid coordinates to 1D indices
    grid_indices = grid_coords[:, 1] * critic_grid_size + grid_coords[:, 0]
    
    # Initialize output grid
    output_grid = torch.zeros((num_channels, critic_grid_size * critic_grid_size), device=device)
    
    # Aggregate data using scatter operations
    for ch in range(num_channels):
        channel_data = all_chunks[:, ch]
        # Use scatter_add to aggregate values at each grid cell
        aggregated = scatter(channel_data, grid_indices, dim=0, 
                           dim_size=critic_grid_size * critic_grid_size, reduce='add')
        output_grid[ch] = aggregated
    
    # Reshape to 2D grid
    output_grid = output_grid.view(num_channels, critic_grid_size, critic_grid_size)
    
    return output_grid

def calculate_density_scatter(positions, grid_size, world_to_map_scale, device):
    """
    Calculate density map using scatter operations.
    
    Args:
        positions: Agent positions (N, 2)
        grid_size: Size of the density grid
        world_to_map_scale: Scale factor from world to map coordinates
        device: Device to use for computation
        
    Returns:
        Density map (grid_size, grid_size)
    """
    if positions.numel() == 0:
        return torch.zeros((grid_size, grid_size), device=device)
    
    # Convert world coordinates to grid coordinates
    grid_coords = (positions * world_to_map_scale).long()
    grid_coords = grid_coords.clamp(0, grid_size - 1)
    
    # Flatten 2D grid coordinates to 1D indices
    grid_indices = grid_coords[:, 1] * grid_size + grid_coords[:, 0]
    
    # Count occurrences at each grid cell
    density = scatter(torch.ones(positions.shape[0], device=device), grid_indices, 
                     dim=0, dim_size=grid_size * grid_size, reduce='add')
    
    # Reshape to 2D grid
    density = density.view(grid_size, grid_size)
    
    return density

def calculate_vec_to_target_scatter(positions, targets, grid_size, world_to_map_scale, device):
    """
    Calculate vector field to targets using scatter operations.
    
    Args:
        positions: Agent positions (N, 2)
        targets: Target positions (N, 2)
        grid_size: Size of the vector field grid
        world_to_map_scale: Scale factor from world to map coordinates
        device: Device to use for computation
        
    Returns:
        Vector field (2, grid_size, grid_size)
    """
    if positions.numel() == 0:
        return torch.zeros((2, grid_size, grid_size), device=device)
    
    # Calculate vectors to targets
    vectors = targets - positions
    
    # Convert world coordinates to grid coordinates
    grid_coords = (positions * world_to_map_scale).long()
    grid_coords = grid_coords.clamp(0, grid_size - 1)
    
    # Flatten 2D grid coordinates to 1D indices
    grid_indices = grid_coords[:, 1] * grid_size + grid_coords[:, 0]
    
    # Initialize output vector field
    vector_field = torch.zeros((2, grid_size * grid_size), device=device)
    
    # Aggregate vectors using scatter operations
    for dim in range(2):
        vector_field[dim] = scatter(vectors[:, dim], grid_indices, 
                                  dim=0, dim_size=grid_size * grid_size, reduce='mean')
    
    # Reshape to 2D grid
    vector_field = vector_field.view(2, grid_size, grid_size)
    
    return vector_field