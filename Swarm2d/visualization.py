import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import imageio
import pygame
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Calculate the project root directory (Swarm2dSimple) and add it to the Python path
# __file__ is Swarm2d/visualize_obs.py, so we go up two levels.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.constants import AGENT_RADIUS, OBS_RADIUS, RAW_CH, OCC_CH, NODE_TYPE, MEM_NODE_FEAT_IDX, TEAM_COLORS, RAW_CH_IDX_TO_NAME

# --- V3: Unified Color Map for PIL and Matplotlib ---
from PIL import Image, ImageDraw, ImageFont

# Define a unified color map.
# PIL functions will use the RGBA tuples.
# Matplotlib functions will use the float lists.
ENTITY_COLOR_MAP = {
    # PIL-specific (RGBA tuples)
    'ally_presence': {'pil': (80, 80, 255, 255), 'mpl': [0.3, 0.3, 1]}, # Lighter blue for visibility
    'enemy_presence': {'pil': (255, 0, 0, 255), 'mpl': [1, 0, 0]},
    'resource_presence': {'pil': (0, 255, 0, 255), 'mpl': [0, 1, 0]},
    'coop_resource_presence': {'pil': (255, 165, 0, 255), 'mpl': [1.0, 0.647, 0.0]}, # Orange
    'hive_ally_presence': {'pil': (0, 255, 255, 255), 'mpl': [0, 1, 1]},
    'hive_enemy_presence': {'pil': (255, 0, 255, 255), 'mpl': [1, 0, 1]},
    'obstacle_presence': {'pil': (220, 220, 220, 255), 'mpl': [0.86, 0.86, 0.86]}, # Brighter gray for visibility
    'self_presence': {'pil': (255, 255, 0, 255), 'mpl': [1, 1, 0]},
    
    'last_seen_resource': {'pil': (0, 255, 0, 255), 'mpl': [0, 1, 0]},
    'last_seen_coop_resource': {'pil': (255, 165, 0, 255), 'mpl': [1.0, 0.647, 0.0]}, # Orange
    'last_seen_hive_ally': {'pil': (0, 255, 255, 255), 'mpl': [0, 1, 1]},
    'last_seen_hive_enemy': {'pil': (255, 0, 255, 255), 'mpl': [1, 0, 1]},
    'last_seen_ally': {'pil': (0, 0, 255, 255), 'mpl': [0, 0, 1]},
    'last_seen_enemy': {'pil': (255, 0, 0, 255), 'mpl': [1, 0, 0]},
    'last_seen_self': {'pil': (255, 255, 0, 255), 'mpl': [1, 1, 0]},
    
    'you_are_here': {'pil': (255, 255, 255, 255), 'mpl': [1, 1, 1]},
    'explored_ground': {'pil': (60, 55, 50, 255), 'mpl': [0.23, 0.21, 0.19]}, # Darker for contrast
    'fog_of_war': {'pil': (30, 30, 35, 255), 'mpl': [0.12, 0.12, 0.14]},
    'background': {'pil': (20, 20, 20, 255), 'mpl': [0.08, 0.08, 0.08]},

    # Graph-specific (Matplotlib floats)
    'agent': {'pil': (0, 0, 255, 255), 'mpl': [0, 0, 1]},
    'self_agent': {'pil': (255, 255, 0, 255), 'mpl': [1, 1, 0]},
    'enemy_agent': {'pil': (255, 0, 0, 255), 'mpl': [1, 0, 0]},
    'resource': {'pil': (0, 255, 0, 255), 'mpl': [0, 1, 0]},
    'hive': {'pil': (0, 255, 255, 255), 'mpl': [0, 1, 1]},
    'enemy_hive': {'pil': (255, 0, 255, 255), 'mpl': [1, 0, 1]},
    'obstacle': {'pil': (220, 220, 220, 255), 'mpl': [0.86, 0.86, 0.86]}, # Match obstacle_presence
}
# --- Updated Strategic Definitions ---
CRITIC_CH_DEFS = {
    'ally_presence': 0,
    'enemy_presence': 1,
    'resource_presence': 2,
    'hive_ally_presence': 3,
    'obstacle_presence': 4,
    'coop_resource_presence': 5 
}
def get_strategic_critic_obs(env, grid_size=64):
    grid = torch.zeros((6, grid_size, grid_size), dtype=torch.float32)
    scale = grid_size / max(env.width, env.height)
    y_coords, x_coords = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    
    # DEBUG COUNTERS
    counts = {'res': 0, 'coop': 0, 'hives': 0}

    def add_entity(channel, pos, radius, is_cube=False):
        gx, gy = pos[0] * scale, pos[1] * scale
        r_grid = radius * scale
        if is_cube:
            mask = (torch.abs(x_coords - gx) <= r_grid) & (torch.abs(y_coords - gy) <= r_grid)
            grid[channel][mask] = 1.0
        else:
            dist_sq = (x_coords - gx)**2 + (y_coords - gy)**2
            sigma = max(r_grid * 0.3, 0.5) 
            splat = torch.exp(-dist_sq / (2 * (sigma**2 + 1e-8)))
            grid[channel] = torch.maximum(grid[channel], splat)

    # 1. Hives (Using env.hives.items() from your env.py)
    for h_id, h_data in env.hives.items():
        counts['hives'] += 1
        add_entity(3, h_data['pos'], 40.0)

    # 2. Resources (Using actual varying radius)
    for r in env.resources:
        if r.get('delivered', False): continue
        is_coop = r.get('cooperative', False) 
        ch = 5 if is_coop else 2
        # Use radius_pb (physical radius) to show varying sizes
        rad = r.get('radius_pb', r.get('radius', 3.0))
        add_entity(ch, r['pos'], rad)

    # 3. Hives (Increased Splat Power)
    for h_id, h_data in env.hives.items():
        gx, gy = h_data['pos'][0] * scale, h_data['pos'][1] * scale
        r_grid = 42.0 * scale
        dist_sq = (x_coords - gx)**2 + (y_coords - gy)**2
        # Use sigma 0.6 for a massive, high-intensity circular landmark
        splat = torch.exp(-dist_sq / (2 * (r_grid * 0.5)**2))
        grid[3] = torch.maximum(grid[3], splat)

    # 4. Obstacles & Agents
    for o in env.obstacles:
        add_entity(4, o['pos'], o.get('radius', 20.0), is_cube=True)
    for a in env.agents:
        if a.get('alive', True):
            add_entity(0 if a['team'] == 0 else 1, a['pos'], 3.0)
            
    print(f"[VERIFICATION DEBUG] Detected: {counts['res']} Res, {counts['coop']} Coop, {counts['hives']} Hives")
    return grid


    

def visualize_verification_proof(env, agent_obs, step, output_dir="observation_visuals"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    critic_tensor = get_strategic_critic_obs(env, grid_size=64)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor='#000000')
    plt.subplots_adjust(top=0.9, wspace=0.1)

    # --- PANEL 1: Ground Truth (Team Colors & Accurate Sizing) ---
    ax = axes[0] # FIX: Assigned here before the loop
    ax.set_facecolor('#080808')
    
    # V25: Define Distinct Visualization Team Colors
    # Override the physical simulation colors for visualization purposes only
    # to avoid confusion with entity types (Red=Enemy, Blue=Ally, Green=Resource, etc.)
    VIS_TEAM_COLORS = {
        0: [0, 150, 136, 255],   # Teal (Distinct from Cyan/Blue)
        1: [121, 85, 72, 255],   # Brown (Distinct from Red)
        2: [233, 30, 99, 255],   # Pink/Magenta (Distinct from Purple/Red)
        3: [156, 39, 176, 255],  # Purple (Distinct from Blue/Pink)
        4: [205, 220, 57, 255],  # Lime (Distinct from Green/Yellow)
        5: [63, 81, 181, 255]    # Indigo (Distinct from Blue)
    }

    # Render Hives with Unique Team Colors
    for h_id, h in env.hives.items():
        rgba = VIS_TEAM_COLORS.get(h_id, TEAM_COLORS.get(h_id, [255, 255, 255, 255]))
        color = [c/255.0 for c in rgba[:3]]
        ax.add_patch(patches.Circle((h['pos'][0], h['pos'][1]), 42.0, color=color, alpha=0.15, zorder=1))

    # Render Obstacles
    for o in env.obstacles:
        r = o.get('radius', 20.0)
        ax.add_patch(patches.Rectangle((o['pos'][0]-r, o['pos'][1]-r), r*2, r*2, color='#333', alpha=0.9, zorder=2))

    # Render Resources (Sized correctly)
    for r in env.resources:
        if not r.get('delivered', False):
            is_coop = r.get('cooperative', False)
            color = ENTITY_COLOR_MAP['coop_resource_presence']['mpl'] if is_coop else ENTITY_COLOR_MAP['resource_presence']['mpl']
            rad = r.get('radius_pb', r.get('radius', 3.0)) # Varying size
            ax.add_patch(patches.Circle((r['pos'][0], r['pos'][1]), rad, color=color, zorder=3))

    # Render Agents with Unique Team Colors
    for a in env.agents:
        if a.get('alive', True):
            team = a['team']
            rgba = VIS_TEAM_COLORS.get(team, TEAM_COLORS.get(team, [128, 128, 128, 255]))
            color = [c/255.0 for c in rgba[:3]]
            ax.add_patch(patches.Circle((a['pos'][0], a['pos'][1]), 3.0, color=color, zorder=4))

    ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
    ax.set_aspect('equal', adjustable='box'); ax.invert_yaxis(); ax.axis('off')
    ax.set_title("1. Environment State", color='white', fontsize=12)

    # --- PANEL 2: Actor (DIRECT RAW_MAP) ---
    ax = axes[1]
    raw_map = agent_obs['map'].cpu().numpy()
    actor_rgb = np.zeros((32, 32, 3))
    for name, idx in RAW_CH.items():
        if idx < raw_map.shape[0]:
            color = np.array(ENTITY_COLOR_MAP.get(name, {'mpl': [1,1,1]})['mpl'])
            actor_rgb += np.expand_dims(raw_map[idx], -1) * color
    ax.imshow(np.clip(actor_rgb, 0, 1), interpolation='nearest')
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    ax.set_title("2. Actor Local View", color='white', fontsize=12)

    # --- PANEL 3: Critic (64x64 Abstraction) ---
    ax = axes[2]
    critic_rgb = np.zeros((64, 64, 3))
    for name, idx in CRITIC_CH_DEFS.items():
        color = np.array(ENTITY_COLOR_MAP[name]['mpl'])
        critic_rgb += np.expand_dims(critic_tensor[idx].cpu().numpy(), -1) * color
    ax.imshow(np.clip(critic_rgb, 0, 1), interpolation='nearest', origin='upper')
    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
    ax.set_title("3. Strategic Critic Map", color='white', fontsize=12)

    plt.savefig(os.path.join(output_dir, f"verification_triple_step_{step}.png"), dpi=300, facecolor='#000000', bbox_inches='tight')
    plt.close()






def visualize_observation_map(obs_map_tensor, channel_definitions, agent_id, step_num, 
                              map_type='raw', output_dir="observation_visuals", 
                              obs_radius=None, # ADDED: Allow passing the specific obs_radius
                              recency_normalization_period=250.0,
                              coverage_map=None,  # If None and map_type='memory', will extract from obs_map_tensor
                              coverage_intensity_exponent=0.5,
                              visualization_gamma=1.0, # ADDED: Gamma for visualization
                              suppress_title=False, # ADDED
                              suppress_legend=False, # ADDED
                              filename_suffix=""): # ADDED
    """
    Visualizes a 2D observation map tensor using PIL for advanced blending and saves it as an image.
    """
    if not isinstance(obs_map_tensor, torch.Tensor):
        print(f"Cannot visualize {map_type} map, as it is not a tensor (type: {type(obs_map_tensor)}).")
        return

    C, H, W = obs_map_tensor.shape
    
    # --- V14: Enhanced Debug Print for ALL central channel activity ---
    if map_type == 'raw' and step_num < 2: # Print for a couple of steps
        center_y, center_x = H // 2, W // 2
        patch_radius = 2 # 2 pixels around the center for a 5x5 patch
        
        print("\n--- Raw Map Center Data (5x5) ---")
        print("Shows raw data for ANY active channel at the map's center.")
        
        found_activity = False
        # Iterate over all channels
        for channel_name, channel_idx in channel_definitions.items():
            if H > (patch_radius*2) and W > (patch_radius*2):
                center_patch = obs_map_tensor[channel_idx, 
                                              center_y-patch_radius:center_y+patch_radius+1, 
                                              center_x-patch_radius:center_x+patch_radius+1].cpu().numpy()
                
                # If there's any significant value in the patch, print it
                if np.any(center_patch > 0.01):
                    found_activity = True
                    print(f"\n[Channel: {channel_name}]")
                    print(np.array2string(center_patch, precision=3, floatmode='fixed'))
        
        if not found_activity:
            print("No significant activity detected in the central 5x5 area.")
        print("---------------------------------\n")

    # --- Debug for Explored Channel in Memory Map ---
    if map_type == 'memory' and 'explored' in channel_definitions and step_num == 0:
        explored_idx = channel_definitions['explored']
        explored_data = obs_map_tensor[explored_idx].cpu().numpy()
        print(f"\n[Memory Map Debug Step {step_num}] 'explored' channel stats:")
        print(f"  Min: {explored_data.min():.4f}, Max: {explored_data.max():.4f}, Mean: {explored_data.mean():.4f}")
        print(f"  Unique values: {np.unique(explored_data)[:10]}")
        print("---------------------------------\n")


    # --- V3: Use PIL for Image Creation and Blending ---
    # Start with a dark background for unexplored areas
    base_image = Image.new("RGBA", (W, H), ENTITY_COLOR_MAP['background']['pil'])
    
    # --- V8: Simplified Logic ---
    # We no longer use a separate fog layer. Instead, the ground layer's age will
    # directly modulate the transparency of the entities drawn on top of it.
    ground_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    entity_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    idx_to_name = {v: k for k, v in channel_definitions.items()}

    # --- 1. Render Explored Ground and Get Age Data ---
    explored_age_data = None
    if map_type == 'memory' and 'explored' in channel_definitions:
        explored_idx = channel_definitions['explored']
        explored_age_data = obs_map_tensor[explored_idx].cpu().numpy()
        
        # DEBUG: Print age data stats
        if step_num % 50 == 0:
            print(f"[DEBUG] Explored Age Data Stats: Min={explored_age_data.min():.3f}, Max={explored_age_data.max():.3f}, Mean={explored_age_data.mean():.3f}")

        # FIX: 'explored' channel is age-based (<= 1.0 = explored, 1.1 = unexplored)
        # 1.1 means "never seen". [0.0, 1.0] means "seen".
        has_been_explored_mask = explored_age_data <= 1.05
        
        # Draw the "observed ground" color on all cells that have ever been seen
        ground_color_np = np.array(ENTITY_COLOR_MAP['explored_ground']['pil']).astype(np.uint8)
        
        # V21: Initialize with dark gray for UNEXPLORED instead of transparent black
        # This helps distinguish the memory map from the raw map's black background
        ground_pixels = np.full((H, W, 4), [15, 15, 15, 255], dtype=np.uint8)
        
        # V20: Apply age-based fading to the ground
        # 0.0 = Recent (Bright), 1.0 = Old (Dark)
        if has_been_explored_mask.any():
            # Get ages for explored cells
            ages = explored_age_data[has_been_explored_mask]
            
            # Create brightness factors: 1.0 -> 0.4 (don't go fully black)
            # age 0 -> 1.0, age 1 -> 0.4
            brightness = 1.0 - (0.6 * ages)
            brightness = np.clip(brightness, 0.4, 1.0)
            
            # Expand dimensions for broadcasting (N, 1)
            brightness_factor = brightness[:, np.newaxis]
            
            # Apply to RGB channels of the ground color
            base_rgb = ground_color_np[:3]
            faded_rgb = (base_rgb * brightness_factor).astype(np.uint8)
            
            # Reconstruct RGBA
            faded_pixels = np.zeros((len(ages), 4), dtype=np.uint8)
            faded_pixels[:, :3] = faded_rgb
            faded_pixels[:, 3] = ground_color_np[3] # Keep original alpha
            
            ground_pixels[has_been_explored_mask] = faded_pixels

        ground_layer = Image.fromarray(ground_pixels, 'RGBA')
    elif map_type == 'memory':
        # If no explored data is available (e.g. step 0 init), fill with default unexplored color
        # This prevents transparent background which shows up differently in viewers
        ground_pixels = np.full((H, W, 4), [15, 15, 15, 255], dtype=np.uint8)
        ground_layer = Image.fromarray(ground_pixels, 'RGBA')

    # --- 2. Render Entities with Additive Blending ---
    entity_pixels = np.zeros((H, W, 4), dtype=np.float32) # Use float for blending

    for i in range(C):
        channel_name = idx_to_name.get(i)
        if channel_name is None or channel_name in ['explored', 'vec_hive_x', 'vec_hive_y', 'step_norm', 'you_are_here', 'coverage']:
            continue

        if channel_name in ENTITY_COLOR_MAP:
            color = np.array(ENTITY_COLOR_MAP[channel_name]['pil'])
            channel_data = obs_map_tensor[i].cpu().numpy()

            # --- V9: Corrected Masking and Intensity Logic ---
            final_intensity = np.array([])
            mask = np.array([])
 
            if map_type == 'memory':
                # --- Step 1: Establish the correct mask for the current channel ---
                if 'obstacle' in channel_name:
                    # Obstacles have a binary presence (1.0 if present, 0.0 otherwise)
                    # V18: HARD EDGES for Memory Map Obstacles
                    # Use a higher threshold to cut off Gaussian tails and render as solid blocks
                    # FIX: For memory map, obstacle channel is an AGE (0.0=recent, 1.0=old, 1.1=unseen).
                    # So we want values <= 1.0.
                    mask = channel_data <= 1.0
                else:
                    # Other entities have an age (0.0 to 1.0). A value of 1.1 means "never seen".
                    mask = channel_data <= 1.0

                if np.any(mask):
                    # --- Step 2: Calculate the final intensity for the masked area ---
                    entity_age = np.clip(channel_data[mask], 0.0, 1.0)
                    
                    # Calculate visibility: 1.0 at age 0, fading to 0.3 at age 1.0 (70% decay)
                    # User request: "decaying up to like 30% of the starting brightness"
                    visibility = 1.0 - (entity_age * 0.7) 

                    # Obstacles do not fade. They are always 100% visible and SOLID.
                    if 'obstacle' in channel_name:
                        # Restore fading for obstacles but keep them brighter than background
                        # Fade from 1.0 (recent) to 0.5 (old). 
                        visibility = 1.0 - (entity_age * 0.5) 
                        final_intensity = visibility * 1.2 # Slight boost to keep them distinct
                    else:
                        # Modulate by the coverage map for partial cell occupation.
                        # If coverage_map is not provided separately, extract it from memory_map
                        if coverage_map is None and map_type == 'memory' and 'coverage' in channel_definitions:
                            coverage_map = obs_map_tensor[channel_definitions['coverage']]
                        
                    if coverage_map is not None:
                        coverage_values = coverage_map.cpu().numpy()[mask]
                        scaled_coverage = np.power(coverage_values, coverage_intensity_exponent)
                        # V27: Refined Coverage Modulation (Less Hot)
                        # - Use lower base (0.4) so single cells aren't too bright
                        # - Use lower scaler (0.5 instead of 0.8) to prevent saturation to white too quickly
                        # - Overlaps will still add up, but require more density to become white hot.
                        coverage_modulation = 0.4 + 0.5 * scaled_coverage
                        final_intensity = visibility * coverage_modulation
                    else:
                        # If no coverage map, use base 0.6 (visible but definitely not white)
                        final_intensity = visibility * 0.6

            elif map_type == 'raw': # Raw map
                mask = channel_data > 0.0
                if np.any(mask):
                    final_intensity = np.clip(channel_data[mask], 0.0, 1.0)
                    
                    # V23: Color Correction for Raw Map
                    # Boost specific channels if needed to match legend perception
                    if 'coop_resource' in channel_name:
                         # Boost intensity for coop resources to make the orange pop more
                         final_intensity = np.power(final_intensity, 0.7) # Gamma boost specifically for this channel

            if final_intensity.size > 0:
                # V26: "Foggy" Transparency Logic (Alpha Modulation instead of Darkening)
                # We want base colors to remain saturated/bright, but become transparent as they fade.
                # Density (intensity > 1.0) should desaturate towards white (hot).
                
                # 1. Base Color (RGB)
                r, g, b = color[:3]
                
                # 2. Calculate Per-Pixel RGB and Alpha
                # Expand dimensions for broadcasting
                # final_intensity shape: (N,) where N is number of masked pixels
                
                # Handle "Hot" Intensity (>1.0): Desaturate towards white
                # T = (I - 1.0) clamped to [0, 1]
                hotness = np.maximum(final_intensity - 1.0, 0.0)
                # Scale RGB towards 255 based on hotness
                # New_C = C + (255 - C) * hotness
                # But we need to do this efficiently on vectors
                
                target_r = r + (255 - r) * hotness
                target_g = g + (255 - g) * hotness
                target_b = b + (255 - b) * hotness
                
                # Clamp to 255
                target_r = np.clip(target_r, 0, 255)
                target_g = np.clip(target_g, 0, 255)
                target_b = np.clip(target_b, 0, 255)
                
                # 3. Calculate Alpha
                # For intensity <= 1.0, Alpha = 255 * intensity
                # For intensity > 1.0, Alpha = 255 (fully opaque)
                target_alpha = np.clip(final_intensity, 0.0, 1.0) * 255
                
                # 4. Apply to Pixel Buffer
                # We need to overwrite/blend into the entity_pixels buffer.
                # Since entity_pixels is float32 accumulator for additive blending,
                # we just add the (Pre-Multiplied?) No, we accumulate light.
                
                # WAIT: The previous logic was "Additive Blending" on a black background.
                # entity_pixels[mask, :3] += color[:3] * final_intensity
                # Then normalized.
                
                # The User wants "Base Brightness = Original Color".
                # If we use simple addition, 1.0 * Color = Color.
                # If we have 2.0 * Color, it becomes 2x Color.
                # The normalization step was killing the "Foggy" look by dimming the 1.0s.
                
                # NEW STRATEGY:
                # Don't normalize RGB globally. Allow it to blow out (clip).
                # But to prevent simple clipping (Yellow -> Yellow), we desaturate towards white manually first.
                
                # Apply the hotness logic calculated above
                entity_pixels[mask, 0] += target_r
                entity_pixels[mask, 1] += target_g
                entity_pixels[mask, 2] += target_b
                
                # For Alpha channel in the accumulator (index 3), we store the MAX alpha seen so far
                # This ensures that if we have a faint trail over a faint trail, it doesn't just sum to opaque.
                # But if we have a dense cluster, it is opaque.
                entity_pixels[mask, 3] = np.maximum(entity_pixels[mask, 3], target_alpha)

    # V26: Skip Global Normalization to preserve brightness of sparse entities
    # Instead, we just clip the accumulated values.
    # Because we already handled "hotness" by shifting towards white, 
    # exceeding 255 just means "very bright white", which is fine.
    
    # Clip and convert to uint8 image
    entity_pixels = np.clip(entity_pixels, 0, 255).astype(np.uint8)
    entity_layer = Image.fromarray(entity_pixels, 'RGBA')

    # --- 3. Composite the Layers ---
    # Start with base, composite ground, then entities on top. No more fog layer.
    final_image = Image.alpha_composite(base_image, ground_layer)
    final_image = Image.alpha_composite(final_image, entity_layer)
    
    # --- 4. Draw "You Are Here" Marker and Waypoints on top ---
    draw = ImageDraw.Draw(final_image)
    if 'you_are_here' in channel_definitions:
        you_are_here_idx = channel_definitions['you_are_here']
        # The new filled yellow circle is the single, authoritative representation
        # of the agent's physical body and position. The white pixel is misleading
        # due to grid snapping and is no longer needed.
        # you_are_mask = obs_map_tensor[you_are_here_idx].cpu().numpy() > 0.1
        # if np.any(you_are_mask):
        #     y, x = np.where(you_are_mask)
        #     if len(x) > 0 and len(y) > 0:
        #         draw.point((x[0], y[0]), fill='white')

    # --- V5: Call dedicated function to render the coverage map ---
    if map_type == 'memory' and coverage_map is not None:
        explored_map = None
        if 'explored' in channel_definitions:
            explored_map = obs_map_tensor[channel_definitions['explored']]
        visualize_coverage_map(coverage_map, explored_map, agent_id, step_num, output_dir)

    # --- 5. Save the Final Image with Matplotlib for Titles/Legends ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Increase figure size for better legend spacing if needed, or keep standard
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(final_image, interpolation='nearest')
    
    # --- NEW: Render high-resolution observation radius for raw map ---
    if map_type == 'raw':
        # V13: Use the provided obs_radius for accurate visualization scaling
        if obs_radius is None:
            # Fallback for safety, but the goal is to always provide it.
            print(f"Warning: obs_radius not provided for raw map visualization. Falling back to constant {OBS_RADIUS}.")
            obs_radius = OBS_RADIUS
            
        W, H = final_image.size
        center_x, center_y = (W - 1) / 2.0, (H - 1) / 2.0
        
        # 1. Draw the outer observation radius (the edge of vision)
        obs_radius_pixels = min(W, H) / 2.0
        obs_circle = patches.Circle((center_x, center_y), obs_radius_pixels,
                                    linewidth=1.2, edgecolor=(0.9, 0.9, 0.9, 0.6), facecolor='none', zorder=10)
        ax.add_patch(obs_circle)

        # 2. Draw the inner physical collision radius (the agent's body)
        # V15: The old circle was misleading. The agent's "self" data is a Gaussian splat
        # centered on one of the four center pixels. The best visualization is to
        # highlight the 2x2 area that represents the "true center" of the even-sized grid.
        # world_units_per_pixel = (obs_radius * 2) / W 
        # agent_radius_in_pixels = AGENT_RADIUS / world_units_per_pixel
        
        # --- NEW: Debug Printout ---
        if step_num == 0 and agent_id == 0: # Print once for clarity
            world_units_per_pixel = (obs_radius * 2) / W
            agent_radius_in_pixels = AGENT_RADIUS / world_units_per_pixel
            print("\n--- Raw Map Scaling Debug ---")
            print(f"Agent Observation Radius (World Units): {obs_radius:.2f}") # V13: Use dynamic radius
            print(f"Agent Physical Radius (World Units):    {AGENT_RADIUS:.2f}")
            print(f"Raw Map Resolution (Pixels):            {W}x{H}")
            print(f"World Units per Pixel:                  ({obs_radius * 2:.2f} / {W}) = {world_units_per_pixel:.2f}") # V13: Use dynamic radius
            print(f"Agent Radius in Pixels:                 ({AGENT_RADIUS:.2f} / {world_units_per_pixel:.2f}) = {agent_radius_in_pixels:.2f}")
            print("---------------------------------\n")

        # V16: Revert to a single-pixel marker for aesthetics, but make it more truthful.
        # Instead of a circle at the geometric center (15.5), we now draw a 1x1 square
        # to highlight the specific pixel (16,16) that the observation generation code
        # uses as its anchor for the "self" splat.
        center_pixel_marker = patches.Rectangle(
            (center_x, center_y),  # Bottom-left corner is (15.5, 15.5) for pixel (16,16)
            1, 1,                  # Width and height of 1 pixel
            linewidth=0, facecolor='yellow', alpha=0.7, zorder=11
        )
        ax.add_patch(center_pixel_marker)

    if not suppress_title:
        title = f"Agent {agent_id} - {map_type.capitalize()} Map - Step {step_num}"
        if map_type == 'memory':
            title += "\n(Persistent Memory with Fog of War)"
        # Use suptitle for better control over positioning relative to fig.text
        fig.suptitle(title, fontsize=16, y=0.99)
    
    # --- V7: Add a grid to emphasize the cellular nature of the map ---
    if map_type == 'memory':
        h, w = final_image.size
        # Set up grid lines for each cell
        ax.set_xticks(np.arange(-.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-.5, h, 1), minor=True)
        
        # Customize the grid appearance
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Remove the major tick labels
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.axis('off')

    # Create legend from PIL color map
    if not suppress_legend:
        legend_elements = []
        # Use a set to avoid duplicate labels (e.g. resource_presence and last_seen_resource)
        legend_labels_done = set()
        for channel_name in channel_definitions.keys():
            label = channel_name.replace('last_seen_', '').replace('_presence', '').replace('_', ' ').title()
            if label in legend_labels_done or 'Vec' in label or 'Step' in label or 'Explored' in label or 'Coverage' in label:
                continue
                
            color_key = channel_name
            if color_key in ENTITY_COLOR_MAP:
                color_rgba_255 = ENTITY_COLOR_MAP[color_key]['pil']
                color_rgba_float = [c/255.0 for c in color_rgba_255]
                legend_elements.append(Line2D([0], [0], color=color_rgba_float[:3], lw=4, label=label))
                legend_labels_done.add(label)

        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))

    # Adjust layout
    if suppress_title and suppress_legend:
        # Tighter layout if no chrome
        fig.tight_layout(pad=0)
    else:
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    filename = os.path.join(output_dir, f"agent_{agent_id}_{map_type}_map_step_{step_num}{filename_suffix}.png")
    plt.savefig(filename, bbox_inches='tight' if suppress_legend else None, pad_inches=0 if suppress_legend else 0.1)
    plt.close()
    print(f"Saved map visualization to {filename}")



def visualize_coverage_map(coverage_map_tensor, explored_map_tensor, agent_id, step_num, output_dir="observation_visuals"):
    """
    Visualizes the coverage map with a title and description and saves it as an image.
    Black = Unexplored, Dark Gray = Explored but empty, White = High entity density.
    """
    if not isinstance(coverage_map_tensor, torch.Tensor):
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    
    H, W = coverage_map_tensor.shape
    
    # --- V6: Create a composite image to show explored but empty areas ---
    # 1. Start with a black background (unexplored)
    display_map = np.zeros((H, W), dtype=np.float32)

    # 2. Add a dark gray base for all explored areas
    if explored_map_tensor is not None:
        # FIX: Explored is age-based (<= 1.0)
        explored_mask = explored_map_tensor.cpu().numpy() <= 1.05
        display_map[explored_mask] = 0.15 # Dark gray

    # 3. Add entity density on top of the explored base
    # This scales coverage from 0->1 to an additional brightness of 0->0.85
    coverage_data = coverage_map_tensor.cpu().numpy()
    display_map += coverage_data * 0.85
    
    # 4. Clip the final result to ensure it's in the [0, 1] range
    display_map = np.clip(display_map, 0, 1)

    # Use a grayscale colormap, vmin/vmax to ensure consistent brightness across frames
    ax.imshow(display_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    
    # --- Title ---
    title = f"Agent {agent_id} - Coverage Map (Entity Density) - Step {step_num}"
    ax.set_title(title, fontsize=16)

    # --- Grid and Ticks ---
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.axis('off')

    fig.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout to make space for text

    # --- Save ---
    filename = os.path.join(output_dir, f"agent_{agent_id}_coverage_map_step_{step_num}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved coverage map visualization to {filename}")


def visualize_graph_observation(graph_obs, agent_id, agent_team_id, step_num, world_bounds, max_steps, recency_normalization_period, output_dir="observation_visuals", path_history=None, suppress_legend=False, suppress_title=False, filename_suffix=""):
    """
    Visualizes a graph observation in 3D, with a debug summary, and saves it as an image.
    X, Y are spatial coordinates, Z is the certainty of the node (combining recency and cluster count).
    Node transparency encodes recency (older nodes become more transparent).
    """
    if not hasattr(graph_obs, 'x') or graph_obs.num_nodes == 0:
        print(f"Cannot visualize graph for agent {agent_id} at step {step_num}, it has no nodes.")
        return

    # --- Extract data from graph ---
    pos = graph_obs.pos.cpu().numpy()
    node_features = graph_obs.x.cpu().numpy()
    
    node_type_idx = MEM_NODE_FEAT_IDX.get('node_type_encoded')
    team_id_idx = MEM_NODE_FEAT_IDX.get('team_id')
    is_ego_idx = MEM_NODE_FEAT_IDX.get('is_ego')
    is_cooperative_idx = MEM_NODE_FEAT_IDX.get('is_cooperative') # New

    # Calculate recency: 1.0 = current step, 0.0 = ancient
    last_seen_idx = MEM_NODE_FEAT_IDX.get('last_observed_step')
    if last_seen_idx is not None and last_seen_idx < node_features.shape[1]:
        # Convert absolute step to recency: 1.0 = current step, 0.0 = ancient
        last_seen_abs = node_features[:, last_seen_idx]
        steps_ago = step_num - last_seen_abs
        normalized_steps_ago = np.clip(steps_ago / recency_normalization_period, 0.0, 1.0)
        recency = 1.0 - normalized_steps_ago
    else:
        recency = np.ones(graph_obs.num_nodes)
        
    # Get cluster counts to determine node sizes and certainty
    if node_features.shape[1] > max(MEM_NODE_FEAT_IDX.values()):
         cluster_counts = node_features[:, -1]
    else:
         cluster_counts = np.ones(graph_obs.num_nodes)
    
    # Calculate certainty metric for Z-axis: combines recency and cluster persistence
    # Certainty = recency * (1 + log(cluster_count + 1) / log(max_cluster_count + 1))
    # This gives higher certainty to: (1) recent observations, and (2) observations that have been clustered (persistent)
    max_cluster_count = np.max(cluster_counts) if len(cluster_counts) > 0 else 1.0
    if max_cluster_count > 1:
        # Normalize cluster contribution to [0, 1] range
        cluster_contribution = np.log(cluster_counts + 1) / np.log(max_cluster_count + 1)
    else:
        cluster_contribution = np.zeros_like(cluster_counts)
    
    # Combine recency and cluster contribution
    # Formula: certainty = recency * (0.5 + 0.5 * cluster_contribution)
    # Explanation: The 0.5 + 0.5 * cluster_contribution creates a multiplier range from [0.5, 1.0]
    # - Base weight of 0.5 ensures recency always has at least 50% influence
    # - Additional 0.5 * cluster_contribution adds up to 50% more weight based on clustering
    # - This means: certainty ranges from 0.5*recency (unclustered, old) to 1.0*recency (clustered)
    # - Clustered observations get a boost because clustering indicates persistent/persistent patterns
    # NOTE: This is a VISUALIZATION METRIC ONLY - the agent does NOT receive this as an observation
    certainty = recency * (0.5 + 0.5 * cluster_contribution)

    # --- NEW: Make hives persistent for visualization ---
    # This ensures that once a hive is discovered, it remains as a fixed, certain
    # landmark in the agent's memory graph visualization.
    node_type_map_for_persistence = {v: k for k,v in NODE_TYPE.items()}
    for i in range(graph_obs.num_nodes):
        node_type_val = int(node_features[i, node_type_idx])
        node_type_name = node_type_map_for_persistence.get(node_type_val, 'unknown')
        if 'hive' in node_type_name:
            recency[i] = 1.0
            certainty[i] = 1.0
            
    # Scale certainty to [0.05, 1.0] range for Z-axis visualization (prevent floor sitting)
    certainty_scaled = certainty * 0.95 + 0.05
        
    # Get node colors and types
    node_type_map = {v: k for k,v in NODE_TYPE.items()}
    node_colors = []
    node_types_list = []
    node_markers = [] # New list for markers

    # New Sizing Logic: Make clustered nodes significantly larger
    node_sizes = np.zeros(graph_obs.num_nodes)
    for i in range(graph_obs.num_nodes):
        if cluster_counts[i] > 1:
            node_sizes[i] = 100 + cluster_counts[i] * 40 # Increased multiplier
        else:
            node_sizes[i] = 35 # Standard size for individual nodes


    for i in range(graph_obs.num_nodes):
        node_type_val = int(node_features[i, node_type_idx])
        node_type_name = node_type_map.get(node_type_val, 'obstacle') # Default to obstacle color
        
        # Distinguish self, ally, enemy agents
        if node_type_name == 'agent':
            if is_ego_idx is not None and node_features[i, is_ego_idx] > 0.5:
                node_type_name = 'self_agent'
                # Use a special color from the map for self_agent
                node_colors.append(ENTITY_COLOR_MAP.get('self_agent', {}).get('mpl', [1, 1, 0]))
                node_markers.append('o') # Circle for self-agent
            elif team_id_idx is not None:
                node_team_id = int(node_features[i, team_id_idx])
                # Use the TEAM_COLORS constant directly
                team_color_rgba = TEAM_COLORS.get(node_team_id, [128, 128, 128, 255])
                node_colors.append([c / 255.0 for c in team_color_rgba[:3]])
                node_markers.append('o') # Circle for other agents
        
        # --- NEW: Distinguish hive types and resource types ---
        elif node_type_name == 'hive':
            if team_id_idx is not None:
                node_team_id = int(node_features[i, team_id_idx])
                # Use TEAM_COLORS for hives as well
                team_color_rgba = TEAM_COLORS.get(node_team_id, [128, 128, 128, 255])
                node_colors.append([c / 255.0 for c in team_color_rgba[:3]])
                node_markers.append('s') # Square for hives
            else:
                node_colors.append(ENTITY_COLOR_MAP.get('hive', {}).get('mpl', [0, 1, 1]))
                node_markers.append('s')

        elif node_type_name == 'resource':
            if is_cooperative_idx is not None and node_features[i, is_cooperative_idx] > 0.5:
                # Use the specific color for coop resources
                node_colors.append(ENTITY_COLOR_MAP.get('coop_resource_presence', {}).get('mpl', [1.0, 0.647, 0.0]))
            else:
                node_colors.append(ENTITY_COLOR_MAP.get('resource', {}).get('mpl', [0, 1, 0]))
            node_markers.append('o') # Circle for resources
        # --- END NEW ---
        else:
            # Fallback for other types like obstacles
            node_colors.append(ENTITY_COLOR_MAP.get(node_type_name, {}).get('mpl', [0.5, 0.5, 0.5]))
            node_markers.append('o') # Default to circle
        
        node_types_list.append(node_type_name)


    # --- DEBUG PRINTOUT ---
    print("\n" + "="*40)
    print(f"DEBUG SUMMARY for Agent {agent_id} Unified Graph at Step {step_num}")
    print(f"  - Total Nodes: {graph_obs.num_nodes}")
    
    # Node type breakdown
    unique_types, counts = np.unique(node_types_list, return_counts=True)
    print("  - Node Types:")
    for t, c in zip(unique_types, counts):
        print(f"    - {t.title()}: {c}")

    # Certainty breakdown
    if hasattr(graph_obs, 'is_live') and graph_obs.is_live is not None and graph_obs.is_live.numel() == graph_obs.num_nodes:
        live_nodes = torch.sum(graph_obs.is_live).item()
    else:
        live_nodes = np.sum(recency > 0.999) # Fallback
    memory_nodes = len(recency) - live_nodes
    print("  - Memory Status:")
    print(f"    - Live Nodes (is_live=True): {live_nodes}")
    print(f"    - Memory Nodes (is_live=False): {memory_nodes}")

    # Clustering breakdown
    clustered_nodes_mask = cluster_counts > 1
    num_clustered_nodes = np.sum(clustered_nodes_mask)
    if num_clustered_nodes > 0:
        total_obs_in_clusters = np.sum(cluster_counts[clustered_nodes_mask])
        print("  - Clustering:")
        print(f"    - Clustered Nodes (Groups): {num_clustered_nodes}")
        print(f"    - Total observations represented by clusters: {int(total_obs_in_clusters)}")

    # Edge breakdown
    total_edges = graph_obs.num_edges
    print("  - Edges:")
    print(f"    - Total Edges in Graph: {total_edges}")
    print("="*40 + "\n")


    # --- Calculate transparency based on recency ---
    # For memory nodes: transparency ranges from 1.0 (fully opaque) to 0.45 (45% opaque) based on recency
    # The range from 100% to 40-50% corresponds to the recency_normalization_period
    # Formula: alpha = 0.45 + 0.55 * recency
    # This means: recency=1.0 -> alpha=1.0, recency=0.0 -> alpha=0.45
    node_alpha = 0.45 + 0.55 * recency
    
    # --- Plotting ---
    fig = plt.figure(figsize=(12, 12)) # Square figure for consistent panel width
    ax = fig.add_subplot(111, projection='3d')

    # --- UPDATED FOVEA/MEMORY SPLIT ---
    # Prefer explicit 'is_live' tag if present, otherwise fall back to recency.
    if hasattr(graph_obs, 'is_live') and graph_obs.is_live is not None and graph_obs.is_live.numel() == graph_obs.num_nodes:
        fovea_mask = graph_obs.is_live.cpu().numpy().astype(bool)
    else:
        # Fallback for older data or different graph types: use recency threshold
        fovea_mask = recency > 0.999
    
    memory_mask = ~fovea_mask
    node_colors_arr = np.array(node_colors)

    # Helper function to classify edge type (defined before use)
    def classify_edge_type(start_node, end_node, node_features, node_types_list, agent_team_id):
        """Classify edge type based on node properties and connection patterns."""
        start_type_val = int(node_features[start_node, node_type_idx])
        end_type_val = int(node_features[end_node, node_type_idx])
        start_team = node_features[start_node, team_id_idx] if team_id_idx is not None else -1
        end_team = node_features[end_node, team_id_idx] if team_id_idx is not None else -1
        
        agent_type_val = NODE_TYPE.get('agent')
        resource_type_val = NODE_TYPE.get('resource')
        hive_type_val = NODE_TYPE.get('hive')
        is_grappling_idx = MEM_NODE_FEAT_IDX.get('is_grappling')
        is_grappled_idx = MEM_NODE_FEAT_IDX.get('is_grappled')
        vel_x_idx = MEM_NODE_FEAT_IDX.get('vel_x_norm')
        vel_y_idx = MEM_NODE_FEAT_IDX.get('vel_y_norm')
        
        # Combat State Edge: Both nodes are grappling
        if (is_grappling_idx is not None and is_grappled_idx is not None and 
            node_features.shape[1] > max(is_grappling_idx, is_grappled_idx)):
            start_grappling = (node_features[start_node, is_grappling_idx] > 0.5) or \
                             (node_features[start_node, is_grappled_idx] > 0.5)
            end_grappling = (node_features[end_node, is_grappling_idx] > 0.5) or \
                           (node_features[end_node, is_grappled_idx] > 0.5)
            if start_grappling and end_grappling and start_type_val == agent_type_val and end_type_val == agent_type_val:
                return 'combat'
        
        # Affiliation Edge: Agent to Hive
        if ((start_type_val == agent_type_val and end_type_val == hive_type_val) or
            (start_type_val == hive_type_val and end_type_val == agent_type_val)):
            return 'affiliation'
        
        # Kinematic Edge: Agents with significant relative velocity
        if (start_type_val == agent_type_val and end_type_val == agent_type_val and
            vel_x_idx is not None and vel_y_idx is not None and 
            node_features.shape[1] > max(vel_x_idx, vel_y_idx)):
            # Calculate relative velocity
            start_vel = np.array([node_features[start_node, vel_x_idx], node_features[start_node, vel_y_idx]])
            end_vel = np.array([node_features[end_node, vel_x_idx], node_features[end_node, vel_y_idx]])
            rel_vel_mag = np.linalg.norm(end_vel - start_vel)
            if rel_vel_mag > 0.15:  # Significant relative velocity threshold
                return 'kinematic'
        
        # Shared Intent Edge: Allied agents (would need target information to be fully accurate,
        # but we can infer from velocity alignment patterns)
        if (start_type_val == agent_type_val and end_type_val == agent_type_val and
            start_team == agent_team_id and end_team == agent_team_id and
            start_team == end_team and vel_x_idx is not None and vel_y_idx is not None and
            node_features.shape[1] > max(vel_x_idx, vel_y_idx)):
            # Check if velocities are somewhat aligned (simplified shared intent detection)
            start_vel = np.array([node_features[start_node, vel_x_idx], node_features[start_node, vel_y_idx]])
            end_vel = np.array([node_features[end_node, vel_x_idx], node_features[end_node, vel_y_idx]])
            vel_alignment = np.dot(start_vel, end_vel) / (np.linalg.norm(start_vel) * np.linalg.norm(end_vel) + 1e-6)
            if vel_alignment > 0.5 and np.linalg.norm(start_vel) > 0.1 and np.linalg.norm(end_vel) > 0.1:
                return 'shared_intent'
        
        return None
    
    # Helper function to plot individual edges (defined before use)
    def _plot_edge(ax, edge_index, i, start_node, end_node, pos, certainty_scaled, node_features,
                   edge_types, fovea_mask, ego_node_idx, agent_team_id, edge_attr, recency, node_type_idx,
                   team_id_idx, NODE_TYPE, is_live_edge=False):
        """Plot a single edge with appropriate styling."""
        start_recency = recency[start_node]
        end_recency = recency[end_node]
        
        # --- NEW: V2 Edge Visualization (Color by Content, Style by Recency) ---
        is_ego_connection = (start_node in ego_node_idx) or (end_node in ego_node_idx)
        start_is_live = fovea_mask[start_node]
        end_is_live = fovea_mask[end_node]
        linestyle = 'solid' if (start_is_live or end_is_live) else 'dashed'

        # Get edge type classification if available
        edge_type = edge_types[i] if i < len(edge_types) else None
        
        # Determine linewidth: live edges are thicker, others are thinner
        # Made thinner to prevent edges from covering nodes visually
        if is_live_edge:
            base_linewidth = 0.5  # Thinner for live edges
        else:
            base_linewidth = 0.2  # Much thinner for memory edges
        
        if is_ego_connection:
            edge_color = 'gold'
            edge_alpha = 0.85
            linewidth = base_linewidth * 1.1 if is_live_edge else base_linewidth * 0.8
            linestyle = 'solid' # Ego connections are always with a live node
        elif edge_type == 'combat':
            edge_color = '#FF4757'  # Bright red for combat
            edge_alpha = 0.8
            linewidth = base_linewidth * 1.2 if is_live_edge else base_linewidth * 0.7
            linestyle = 'solid'
        elif edge_type == 'kinematic':
            edge_color = '#FF6B6B'  # Coral red for kinematic
            edge_alpha = 0.65
            linewidth = base_linewidth * 1.0 if is_live_edge else base_linewidth * 0.6
            linestyle = 'dashed'
        elif edge_type == 'affiliation':
            edge_color = '#4ECDC4'  # Teal for affiliation
            edge_alpha = 0.7
            linewidth = base_linewidth * 1.0 if is_live_edge else base_linewidth * 0.6
            linestyle = 'solid'
        elif edge_type == 'shared_intent':
            edge_color = '#95E1D3'  # Mint green for shared intent
            edge_alpha = 0.65
            linewidth = base_linewidth * 1.0 if is_live_edge else base_linewidth * 0.6
            linestyle = 'dotted'
        else:
            # Get node types and teams for content-aware coloring
            start_type_val = int(node_features[start_node, node_type_idx])
            end_type_val = int(node_features[end_node, node_type_idx])
            start_team = node_features[start_node, team_id_idx] if team_id_idx is not None else -1
            end_team = node_features[end_node, team_id_idx] if team_id_idx is not None else -1

            agent_type_val = NODE_TYPE.get('agent')
            resource_type_val = NODE_TYPE.get('resource')
            hive_type_val = NODE_TYPE.get('hive')

            # Default color
            edge_color = '#FFCC80' # Light Orange for other (e.g., res-res)

            # Agent-agent connection
            if start_type_val == agent_type_val and end_type_val == agent_type_val:
                edge_color = 'red' if start_team != end_team else 'blue'
            # Agent-resource connection
            elif (start_type_val == agent_type_val and end_type_val == resource_type_val) or \
                 (start_type_val == resource_type_val and end_type_val == agent_type_val):
                edge_color = 'green'
            # Agent-hive connection
            elif (start_type_val == agent_type_val and end_type_val == hive_type_val) or \
                 (start_type_val == hive_type_val and end_type_val == agent_type_val):
                edge_color = 'cyan'

            edge_alpha = np.clip((start_recency + end_recency) / 2.0, 0.25, 0.55)  # Reduced alpha
            
            # Linewidth based on distance for non-special edges - made thinner
            if edge_attr is not None and i < edge_attr.shape[0]:
                normalized_distance = edge_attr[i, 2] 
                linewidth = np.clip((1.0 - normalized_distance) * 0.5, 0.15, 0.35) if not is_live_edge else \
                           np.clip((1.0 - normalized_distance) * 0.8, 0.3, 0.6)
            else:
                linewidth = base_linewidth if is_live_edge else base_linewidth * 0.6

        p1 = pos[start_node]
        p2 = pos[end_node]
        z1 = certainty_scaled[start_node]
        z2 = certainty_scaled[end_node]
        
        # Use lower zorder for edges so nodes appear on top when at same Z depth
        # In 3D plots, Z coordinate primarily determines rendering, but zorder helps when Z is similar
        edge_zorder = 3 if is_live_edge else 1  # Edges below nodes (nodes use zorder 10-20)
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z1, z2], color=edge_color, alpha=edge_alpha, 
                linewidth=linewidth, linestyle=linestyle, zorder=edge_zorder)

    # IMPORTANT: Plot projection lines FIRST, then edges, then nodes on top
    
    # Add projection lines from nodes to their certainty values on the Z-axis
    # Project each node vertically down to z=0 to visualize certainty height
    # Plot these BEFORE nodes so they appear as background guides
    for i in range(graph_obs.num_nodes):
        x, y = pos[i, 0], pos[i, 1]
        z = certainty_scaled[i]
        
        # Project vertically from node to ground (z=0) at the same (x, y) position
        ax.plot([x, x], [y, y], [0, z], color='lightgray', linestyle=':', linewidth=0.6, alpha=0.5, zorder=2)
    
    # IMPORTANT: Plot edges AFTER projection lines, then nodes on top (so nodes aren't covered by edges)
    
    # Plot edges with transparency based on recency
    # Separate edges into live and non-live for proper layering
    if graph_obs.num_edges > 0:
        edge_index = graph_obs.edge_index.cpu().numpy()
        edge_attr = graph_obs.edge_attr.cpu().numpy() if hasattr(graph_obs, 'edge_attr') and graph_obs.edge_attr is not None else None
        num_nodes = pos.shape[0]
        ego_node_idx = np.where(node_features[:, is_ego_idx] > 0.5)[0] if is_ego_idx is not None else []
        
        # Classify all edges first (before plotting loop)
        edge_types = []
        live_edges = []  # Store indices of live edges
        non_live_edges = []  # Store indices of non-live edges
        
        for i in range(edge_index.shape[1]):
            start_node = edge_index[0, i]
            end_node = edge_index[1, i]
            if start_node < num_nodes and end_node < num_nodes:
                edge_type = classify_edge_type(start_node, end_node, node_features, node_types_list, agent_team_id)
                edge_types.append(edge_type)
                # Check if this is a live edge (connects to live nodes)
                if fovea_mask[start_node] or fovea_mask[end_node]:
                    live_edges.append(i)
                else:
                    non_live_edges.append(i)
            else:
                edge_types.append(None)
                non_live_edges.append(i)

        # First plot non-live edges (darker, lower layer, thinner)
        for i in non_live_edges:
            start_node = edge_index[0, i]
            end_node = edge_index[1, i]
            if start_node >= num_nodes or end_node >= num_nodes:
                continue
            _plot_edge(ax, edge_index, i, start_node, end_node, pos, certainty_scaled, node_features, 
                       edge_types, fovea_mask, ego_node_idx, agent_team_id, edge_attr, recency, node_type_idx, 
                       team_id_idx, NODE_TYPE, is_live_edge=False)
        
        # Then plot live edges (brighter, top layer, thicker)
        for i in live_edges:
            start_node = edge_index[0, i]
            end_node = edge_index[1, i]
            if start_node >= num_nodes or end_node >= num_nodes:
                continue
            _plot_edge(ax, edge_index, i, start_node, end_node, pos, certainty_scaled, node_features,
                       edge_types, fovea_mask, ego_node_idx, agent_team_id, edge_attr, recency, node_type_idx,
                       team_id_idx, NODE_TYPE, is_live_edge=True)

    # 1. Plot memory nodes (spheres) - AFTER edges so they appear on top
    if np.any(memory_mask):
        mem_pos = pos[memory_mask]
        mem_certainty_scaled = certainty_scaled[memory_mask]
        mem_alpha = node_alpha[memory_mask]
        mem_colors = node_colors_arr[memory_mask]
        mem_markers = np.array(node_markers)[memory_mask]
        mem_sizes = node_sizes[memory_mask]

        # Unique markers in memory
        unique_mem_markers = np.unique(mem_markers)
        for marker_shape in unique_mem_markers:
            marker_mask = (mem_markers == marker_shape)
            ax.scatter(mem_pos[marker_mask, 0], mem_pos[marker_mask, 1], mem_certainty_scaled[marker_mask],
                       c=mem_colors[marker_mask], s=mem_sizes[marker_mask],
                       alpha=mem_alpha[marker_mask], marker=marker_shape, zorder=15)
        
        # Add vertical stems for memory nodes
        for i in range(len(mem_pos)):
            ax.plot([mem_pos[i, 0], mem_pos[i, 0]],
                    [mem_pos[i, 1], mem_pos[i, 1]],
                    [0, mem_certainty_scaled[i]],
                    color=node_colors_arr[memory_mask][i], alpha=0.4, linewidth=1.2, linestyle='--')


    # 2. Plot foveal nodes (diamonds with stems)
    if np.any(fovea_mask):
        fovea_pos = pos[fovea_mask]
        fovea_certainty_scaled = certainty_scaled[fovea_mask]
        fovea_colors = node_colors_arr[fovea_mask]
        fovea_markers = np.array(node_markers)[fovea_mask]
        fovea_sizes = node_sizes[fovea_mask] * 1.5  # Make fovea nodes larger

        # Plot vertical stems for foveal nodes
        for i in range(len(fovea_pos)):
            ax.plot([fovea_pos[i, 0], fovea_pos[i, 0]],
                    [fovea_pos[i, 1], fovea_pos[i, 1]],
                    [0, fovea_certainty_scaled[i]],
                    color='yellow', alpha=0.7, linewidth=1.5)

        # Plot the foveal nodes themselves (always fully opaque since they're live)
        # We need to iterate through unique markers to plot them correctly
        unique_fovea_markers = np.unique(fovea_markers)
        for marker_shape in unique_fovea_markers:
            marker_mask = (fovea_markers == marker_shape)
            ax.scatter(fovea_pos[marker_mask, 0], fovea_pos[marker_mask, 1], fovea_certainty_scaled[marker_mask],
                       c=fovea_colors[marker_mask], s=fovea_sizes[marker_mask],
                       alpha=1.0, marker=marker_shape, edgecolor='black', linewidth=1.0, zorder=20)

    # --- NEW: Draw Agent's Path ---
    if path_history and len(path_history) > 1:
        path = np.array(path_history)
        # Draw the path on the ground plane (z=0)
        ax.plot(path[:, 0], path[:, 1], 0, color='yellow', linewidth=1.5, alpha=0.6, zorder=1, label="Agent Path")

    # --- DYNAMIC ZOOMING WITH IMPROVED PADDING ---
    # Set limits first so we can use them for projection lines
    if pos.shape[0] > 0:
        min_coords = pos.min(axis=0)
        max_coords = pos.max(axis=0)
        center = (min_coords + max_coords) / 2
        span = (max_coords - min_coords)
        
        # Improved padding: use adaptive padding based on span size for smoother transitions
        # Use exponential decay for smoother pan-out effects as the view expands
        span_max = span.max()
        if span_max > 0:
            # Smooth exponential decay: padding starts at 30% for small spans, decays to ~15% for large spans
            # Using exp(-span/600) provides smooth transitions between different view sizes
            decay_factor = np.exp(-span_max / 600.0)  # Smooth exponential decay
            padding_factor = 0.20 + 0.20 * decay_factor  # Range: 0.20 to 0.40 - Increased base padding
            padding = max(span_max * padding_factor, 100)  # Minimum padding of 100 units for very tight views
        else:
            padding = 100
        
        ax.set_xlim(center[0] - span[0]/2 - padding, center[0] + span[0]/2 + padding)
        ax.set_ylim(center[1] - span[1]/2 - padding, center[1] + span[1]/2 + padding)
    else:
        # Fallback to world bounds if there are no nodes
        ax.set_xlim(0, world_bounds['width'])
        ax.set_ylim(0, world_bounds['height'])

    ax.set_zlim(0, 1.1)

    # --- ORGANIZED TITLES ---
    if not suppress_title:
        main_title = f"Agent {agent_id} - Foveated Graph (Global & Local State) - Step {step_num}"
        fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.97)

    # Legend target positions (axes coordinates)
    node_legend_loc = 'upper left'
    node_legend_anchor = (0.01, 0.98)
    edge_legend_loc = 'upper right'
    edge_legend_anchor = (0.99, 0.98)
    
    # --- V15: Explicitly set ticks and labels to ensure they are visible ---
    ax.set_xlabel("X Coordinate", fontsize=11, labelpad=10)
    ax.set_ylabel("Y Coordinate", fontsize=11, labelpad=10)
    ax.set_zlabel("Certainty", fontsize=11, fontweight='bold', labelpad=10)
    
    # Force ticks to be shown
    ax.tick_params(axis='x', which='major', labelsize=9, pad=5)
    ax.tick_params(axis='y', which='major', labelsize=9, pad=5)
    ax.tick_params(axis='z', which='major', labelsize=9, pad=5)
    
    # --- V16: Improve spacing to prevent cutoff ---
    # Increase subplot margins significantly
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)

    # --- V5: Adjust aspect ratio to make Z-axis less tall ---
    # This prevents the Z-axis (recency) from visually dominating the X/Y spatial layout.
    # We set the Z-axis display height to be a fraction of the average of the X and Y spatial ranges.
    x_range = np.abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    z_height_ratio = 0.4  # Adjust this to make the plot taller or flatter
    if x_range > 0 and y_range > 0:
        ax.set_box_aspect((x_range, y_range, (x_range + y_range) * 0.5 * z_height_ratio))

    ax.grid(True, color='gray', linestyle=':', linewidth=0.2, alpha=0.1)
    # Zoom out slightly to prevent clipping of axis labels during movement
    ax.view_init(elev=30, azim=60)
    ax.dist = 13 # Default is usually 10. Increase to zoom out and fit everything.
    
    # V25: Define Distinct Visualization Team Colors (Deprecated, now in Constants)
    # The constants file has been updated with these new colors.
    # VIS_TEAM_COLORS = { ... } # Removed to enforce consistency
    
    # --- IMPROVED LEGEND PLACEMENT ---
    if not suppress_legend:
        # Place legend in bottom-left to avoid overlap with text box
        node_legend_map = {
            'Self Agent': {'marker': 'o', 'color': ENTITY_COLOR_MAP['self_agent']['mpl']},
            'Hive': {'marker': 's', 'color': 'grey'}, # New entry for Hives
            'Resource': {'marker': 'o', 'color': ENTITY_COLOR_MAP['resource']['mpl']},
            'Co-op Resource': {'marker': 'o', 'color': ENTITY_COLOR_MAP['coop_resource_presence']['mpl']},
            'Obstacle': {'marker': 'o', 'color': ENTITY_COLOR_MAP['obstacle']['mpl']},
            'Live Fovea': {'marker': 'o', 'color': 'grey', 'edgecolor': 'black'}  # Changed from 'D' to 'o' to match actual rendering
        }
        node_elements = [Line2D([0], [0], marker=val['marker'], color='w', label=key, 
                               markerfacecolor=val['color'], markersize=8, 
                               markeredgecolor=val.get('edgecolor', 'none')) 
                         for key, val in node_legend_map.items()]

        edge_legend_map = {
            'Ego Connection': 'gold',
            'Kinematic': '#FF6B6B',  # Coral red
            'Affiliation': '#4ECDC4',  # Teal
            'Shared Intent': '#95E1D3',  # Mint green
            'Combat': '#FF4757',  # Bright red
            'Ally': 'blue',
            'Enemy': 'red',
            'Resource': 'green',
            'Hive': 'cyan',
            'Memory-to-Memory': '#FFCC80'
        }
        edge_elements = [Line2D([0], [0], color=val, lw=2, label=key) for key, val in edge_legend_map.items()]

        # Position legends toward the upper-left for better visibility
        leg1 = ax.legend(handles=node_elements, title='Node Types', 
                        loc=node_legend_loc, bbox_to_anchor=node_legend_anchor, 
                        fontsize=8, framealpha=0.9, edgecolor='gray', fancybox=True)
        leg1.get_title().set_fontweight('bold')
        leg1.get_title().set_fontsize(9)
        ax.add_artist(leg1)

        leg2 = ax.legend(handles=edge_elements, title='Edge Types', 
                        loc=edge_legend_loc, bbox_to_anchor=edge_legend_anchor, 
                        fontsize=8, framealpha=0.9, edgecolor='gray', fancybox=True)
        leg2.get_title().set_fontweight('bold')
        leg2.get_title().set_fontsize(9)
        ax.add_artist(leg2)

        # --- NEW: Team Color Legend (Static) ---
        team_color_elements = []
        # Iterate through all possible teams from the TEAM_COLORS constant (now updated)
        # to create a complete, static legend.
        for team_id, color_rgba in TEAM_COLORS.items():
            color = [c / 255.0 for c in color_rgba[:3]]
            label = f"Team {team_id}"
            if team_id == agent_team_id:
                label += " (Ally)"
            team_color_elements.append(Line2D([0], [0], color=color, lw=4, label=label))

        if team_color_elements:
            leg3 = ax.legend(handles=team_color_elements, title='Team Colors (Agents & Hives)',
                             loc='lower left', bbox_to_anchor=(0.01, 0.15),
                             fontsize=8, framealpha=0.9, edgecolor='gray', fancybox=True)


    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"agent_{agent_id}_graph_step_{step_num}{filename_suffix}.png")
    
    # Adjust tight_layout based on suppression to minimize whitespace
    # V21: Removed tight_layout as it conflicts with manual subplots_adjust and can cause cutoff
    # if suppress_legend and suppress_title:
    #    plt.tight_layout(pad=0.5)
    
    plt.savefig(filename)
    plt.close()
    print(f"Saved graph visualization to {filename}")


class MetricsTracker:
    """A simple class to collect metrics during the simulation."""
    def __init__(self):
        self.data = []

    def record(self, step, metrics):
        record = {'step': step}
        record.update(metrics)
        self.data.append(record)

    def get_dataframe(self):
        return pd.DataFrame(self.data)

def plot_and_save_metrics(df, output_dir="metrics_visuals"):
    """Generates and saves plots for the collected metrics."""
    if df.empty:
        print("Metrics DataFrame is empty. Skipping plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- 1. Exploration & World Awareness Plot ---
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y='exploration_pct', label='Exploration %')
    plt.title('Agent Exploration Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Map Explored (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "metric_exploration.png"))
    plt.close()
    print(f"Saved exploration metric plot to {os.path.join(output_dir, 'metric_exploration.png')}")

    # --- 2. Knowledge Growth Plot (Graph Nodes) ---
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y='total_nodes', label='Total Entities Known', color='black')
    sns.lineplot(data=df, x='step', y='ally_nodes', label='Allies Known', color='blue')
    sns.lineplot(data=df, x='step', y='enemy_nodes', label='Enemies Known', color='red')
    sns.lineplot(data=df, x='step', y='resource_nodes', label='Resources Known', color='green')
    plt.title('Agent Knowledge Growth Over Time (Graph Nodes)')
    plt.xlabel('Simulation Step')
    plt.ylabel('Number of Nodes in Memory Graph')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "metric_knowledge_growth.png"))
    plt.close()
    print(f"Saved knowledge growth metric plot to {os.path.join(output_dir, 'metric_knowledge_growth.png')}")

    # --- 3. Average Certainty Plot ---
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y='avg_certainty')
    plt.title('Agent Average Knowledge Certainty Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Certainty')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "metric_certainty.png"))
    plt.close()
    print(f"Saved certainty metric plot to {os.path.join(output_dir, 'metric_certainty.png')}")

    # --- 4. Task Performance Plot ---
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y='resources_delivered', label='Cumulative Resources Delivered', color='purple')
    plt.title('Team Task Performance: Resource Delivery')
    plt.xlabel('Simulation Step')
    plt.ylabel('Total Resources Delivered')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "metric_resource_delivery.png"))
    plt.close()
    print(f"Saved task performance metric plot to {os.path.join(output_dir, 'metric_resource_delivery.png')}")

    # --- 5. Team Survival Rate Plot ---
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='step', y='team_survival_rate')
    plt.title('Team Survival Rate Over Time')
    plt.xlabel('Simulation Step')
    plt.ylabel('Survival Rate (%)')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, "metric_survival_rate.png"))
    plt.close()
    print(f"Saved survival rate metric plot to {os.path.join(output_dir, 'metric_survival_rate.png')}")


def save_graph_legend_only(agent_team_id, output_dir="observation_visuals"):
    """
    Saves a standalone image containing only the legends used in graph visualizations.
    This is useful for creating combined sequence images with a shared legend.
    """
    fig, ax = plt.subplots(figsize=(6, 8)) # Vertical layout for legend
    ax.axis('off')
    
    # --- 1. Node Legend ---
    node_legend_map = {
        'Self Agent': {'marker': 'o', 'color': ENTITY_COLOR_MAP['self_agent']['mpl']},
        'Hive': {'marker': 's', 'color': 'grey'}, 
        'Resource': {'marker': 'o', 'color': ENTITY_COLOR_MAP['resource']['mpl']},
        'Co-op Resource': {'marker': 'o', 'color': ENTITY_COLOR_MAP['coop_resource_presence']['mpl']},
        'Obstacle': {'marker': 'o', 'color': ENTITY_COLOR_MAP['obstacle']['mpl']},
        'Live Fovea': {'marker': 'o', 'color': 'grey', 'edgecolor': 'black'}
    }
    node_elements = [Line2D([0], [0], marker=val['marker'], color='w', label=key, 
                           markerfacecolor=val['color'], markersize=10, 
                           markeredgecolor=val.get('edgecolor', 'none')) 
                     for key, val in node_legend_map.items()]

    # --- 2. Edge Legend ---
    edge_legend_map = {
        'Ego Connection': 'gold',
        'Kinematic': '#FF6B6B',
        'Affiliation': '#4ECDC4',
        'Shared Intent': '#95E1D3',
        'Combat': '#FF4757',
        'Ally': 'blue',
        'Enemy': 'red',
        'Resource': 'green',
        'Hive': 'cyan',
        'Memory-to-Memory': '#FFCC80'
    }
    edge_elements = [Line2D([0], [0], color=val, lw=3, label=key) for key, val in edge_legend_map.items()]

    # --- 3. Team Color Legend ---
    team_color_elements = []
    for team_id, color_rgba in TEAM_COLORS.items():
        color = [c / 255.0 for c in color_rgba[:3]]
        label = f"Team {team_id}"
        if team_id == agent_team_id:
            label += " (Ally)"
        team_color_elements.append(Line2D([0], [0], color=color, lw=4, label=label))

    # Add legends to the empty figure
    # We place them vertically
    leg1 = ax.legend(handles=node_elements, title='Node Types', 
                    loc='upper center', bbox_to_anchor=(0.5, 0.98),
                    fontsize=10, framealpha=1.0, edgecolor='gray', fancybox=True)
    leg1.get_title().set_fontweight('bold')
    leg1.get_title().set_fontsize(12)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=edge_elements, title='Edge Types', 
                    loc='upper center', bbox_to_anchor=(0.5, 0.65),
                    fontsize=10, framealpha=1.0, edgecolor='gray', fancybox=True)
    leg2.get_title().set_fontweight('bold')
    leg2.get_title().set_fontsize(12)
    ax.add_artist(leg2)
    
    if team_color_elements:
        leg3 = ax.legend(handles=team_color_elements, title='Team Colors',
                         loc='upper center', bbox_to_anchor=(0.5, 0.30),
                         fontsize=10, framealpha=1.0, edgecolor='gray', fancybox=True)
        leg3.get_title().set_fontweight('bold')
        leg3.get_title().set_fontsize(12)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "graph_legend_only.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved graph legend visualization to {filename}")
    return filename


def create_sequence_composition(image_paths, legend_path, output_path):
    """
    Combines a sequence of graph images into a single image with a shared legend.
    """
    images = []
    for path in image_paths:
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"Warning: Image not found for sequence: {path}")
    
    if not images:
        print("No images found for sequence composition.")
        return

    if os.path.exists(legend_path):
        legend_img = Image.open(legend_path)
    else:
        print(f"Warning: Legend image not found: {legend_path}")
        legend_img = None

    # Assume all graph images are same size
    w, h = images[0].size
    
    # Calculate dimensions for the combined image
    # We'll arrange graph images in a row, and put the legend on the right.
    # If the row is too long, we might split it, but user asked for "sequence image", usually a row or strip.
    # Let's do a single row.
    
    combined_width = w * len(images)
    combined_height = h
    
    if legend_img:
        # Resize legend to match height of graph images (or reasonable width)
        # Let's keep legend original aspect ratio but fit it to the height of the graphs
        # or maybe just attach it to the right.
        # Legends are vertical (see save_graph_legend_only), so they are tall.
        # Graph images are 16x12 inches -> high resolution.
        
        # We'll append legend to the right.
        leg_w, leg_h = legend_img.size
        
        # If legend is taller than graph, we might need to scale everything or crop.
        # Usually graph images are quite large.
        
        combined_width += leg_w
        combined_height = max(h, leg_h)
    
    final_img = Image.new("RGBA", (combined_width, combined_height), (255, 255, 255, 255))
    
    current_x = 0
    for img in images:
        final_img.paste(img, (current_x, 0))
        current_x += w
        
    if legend_img:
        # Center legend vertically if it's shorter, or align top
        y_offset = (combined_height - leg_h) // 2
        final_img.paste(legend_img, (current_x, max(0, y_offset)))
        
    final_img.save(output_path)
    print(f"Saved combined sequence image to {output_path}")


def create_legend_footer(image_widths, height=350):
    """
    Creates a dedicated legend footer image to be appended to the comparison panel.
    Returns a PIL Image.
    Args:
        image_widths: List of widths for the 4 images (Ground Truth, Raw, Memory, Graph)
        height: Height of the footer
    """
    total_width = sum(image_widths)
    img = Image.new("RGB", (total_width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 36) # Reduced section header
        font_title = ImageFont.truetype("arial.ttf", 28) # Reduced item title
        font_small = ImageFont.truetype("arial.ttf", 22) # Reduced description
    except:
        font_large = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Define Legend Sections matching the columns
    
    # --- Column 1: Ground Truth & Team IDs ---
    col1_x = 0
    draw.text((col1_x + 20, 20), "Simulation & Teams", fill="white", font=font_large)
    draw.text((col1_x + 20, 70), "Map: 1000x1000 | Teams: 6", fill="#CCCCCC", font=font_title)
    
    # Team Swatches (Compact 2-row layout)
    team_swatch_size = 25
    start_y = 120
    for team_id in range(6): # Assume 6 teams
        if team_id in TEAM_COLORS:
            rgba = TEAM_COLORS[team_id]
            color = tuple(int(c) for c in rgba)
            
            # Layout: 2 rows of 3
            col_offset = (team_id % 3) * 140
            row_offset = (team_id // 3) * 40
            
            x_pos = col1_x + 20 + col_offset
            y_pos = start_y + row_offset
            
            draw.rectangle([x_pos, y_pos, x_pos + team_swatch_size, y_pos + team_swatch_size], fill=color, outline="white")
            draw.text((x_pos + 35, y_pos), f"Team {team_id}", fill="white", font=font_small)

    # --- Column 2: Entity Key (Concise) ---
    col2_x = image_widths[0]
    draw.text((col2_x + 20, 20), "Entity Key", fill="white", font=font_large)
    
    entities = [
        ('Self', ENTITY_COLOR_MAP['self_presence']['pil']),
        ('Ally', ENTITY_COLOR_MAP['ally_presence']['pil']),
        ('Enemy', ENTITY_COLOR_MAP['enemy_presence']['pil']),
        ('Resource', ENTITY_COLOR_MAP['resource_presence']['pil']),
        ('Coop Res', ENTITY_COLOR_MAP['coop_resource_presence']['pil']),
        ('Hive (Ally)', ENTITY_COLOR_MAP['hive_ally_presence']['pil']),
        ('Hive (Enemy)', ENTITY_COLOR_MAP['hive_enemy_presence']['pil']),
        ('Obstacle', ENTITY_COLOR_MAP['obstacle_presence']['pil']),
    ]
    
    curr_y = 70
    # Split entities into two sub-columns
    for i, (name, color) in enumerate(entities):
        x_pos = col2_x + 20 + (220 if i >= 4 else 0) # Tighter column spacing
        y_pos = curr_y + ((i % 4) * 40) # Tighter vertical spacing
        
        draw.rectangle([x_pos, y_pos, x_pos + 25, y_pos + 25], fill=color, outline="white")
        draw.text((x_pos + 35, y_pos), name, fill="white", font=font_title)

    # --- Column 3: Memory Properties ---
    col3_x = image_widths[0] + image_widths[1]
    draw.text((col3_x + 20, 20), "Memory Properties", fill="white", font=font_large)
    
    draw.rectangle([col3_x + 20, 70, col3_x + 45, 95], fill=ENTITY_COLOR_MAP['explored_ground']['pil'], outline="gray")
    draw.text((col3_x + 55, 70), "Explored Ground", fill="white", font=font_title)
    
    draw.rectangle([col3_x + 20, 110, col3_x + 45, 135], fill=(0, 0, 0), outline="gray")
    draw.text((col3_x + 55, 110), "Unexplored (Void)", fill="white", font=font_title)
    
    draw.text((col3_x + 20, 160), "Brightness = Certainty", fill="#AAAAAA", font=font_small)


    # --- Column 4: Graph Representation (Concise Tables) ---
    col4_x = image_widths[0] + image_widths[1] + image_widths[2]
    draw.text((col4_x + 20, 20), "Graph Representation", fill="white", font=font_large)
    
    node_items = [
        ('Agent', 'circle', 'white'),
        ('Hive', 'square', 'white'),
        ('Fovea', 'circle', 'yellow'), 
    ]
    
    # Concise edge list - V29: Removed edges from footer as per user request
    # "check the edges from the graph representation, not sure if we should include them in the full_panel, since these are not visible anyway"
    edge_items = []
    
    curr_y = 70
    
    # Nodes Header
    draw.text((col4_x + 20, curr_y), "Nodes:", fill="white", font=font_title)
    
    # Render Nodes
    node_y = curr_y + 40
    for i, (name, shape, color) in enumerate(node_items):
        y_pos = node_y + (i * 35)
        
        # Draw Shape
        shape_x = col4_x + 20
        shape_y = y_pos + 5
        if shape == 'circle':
            draw.ellipse([shape_x, shape_y, shape_x + 15, shape_y + 15], outline="white", width=2)
        elif shape == 'square':
            draw.rectangle([shape_x, shape_y, shape_x + 15, shape_y + 15], outline="white", width=2)
            
        draw.text((col4_x + 45, y_pos), name, fill="white", font=font_small)

    # Z-Axis Annotation (Moved up since edges are gone)
    draw.text((col4_x + 20, node_y + 120), "Z-Axis: Knowledge Certainty", fill="#AAAAAA", font=font_small)

    return img

def create_comparison_image(env, agent_id, step, output_dir):
    """
    Creates a side-by-side comparison image of:
    1. Ground Truth (Pygame Render)
    2. Raw Map (Agent Vision)
    3. Memory Map (Agent Internal Map)
    4. Graph Representation (Agent Knowledge)
    """
    # 1. Get Ground Truth from Pygame Render Manager
    gt_img = None
    try:
        # Robust Ground Truth Capture
        if hasattr(env, 'render_manager') and env.render_manager is not None:
             # Ensure the screen is active and updated
             env.render_manager.render(mode="human", suppress_overlay=True) # Suppress text overlay
             if env.render_manager.screen is not None:
                 # Pygame surface to array (W, H, 3)
                 view = pygame.surfarray.array3d(env.render_manager.screen)
                 # Transpose to (H, W, 3) for PIL/Matplotlib
                 ground_truth_arr = np.transpose(view, (1, 0, 2))
                 gt_img = Image.fromarray(ground_truth_arr.astype('uint8'), 'RGB')
             else:
                 print("Warning: env.render_manager.screen is None.")
        else:
             print("Warning: env has no render_manager.")

    except Exception as e:
        print(f"Failed to capture ground truth: {e}")
        # import traceback
        # traceback.print_exc()

    if gt_img is None:
        gt_img = Image.new("RGB", (600, 600), (40, 40, 40))
        d = ImageDraw.Draw(gt_img)
        d.text((50, 280), "Ground Truth Unavailable", fill="white")
    
    # 2. Load the other generated images
    # Filenames are standard from generate_visuals
    raw_path = os.path.join(output_dir, f"agent_{agent_id}_raw_map_step_{step}_clean.png")
    mem_path = os.path.join(output_dir, f"agent_{agent_id}_memory_map_step_{step}_clean.png")
    graph_path = os.path.join(output_dir, f"agent_{agent_id}_graph_step_{step}_clean.png")
    
    images = []
    titles = [
        "1. Ground Truth (Global)", 
        "2. Raw Sensory Map (Local 32x32)", 
        "3. Memory Map (Global Model 64x64)", 
        "4. Foveated Graph (Global & Local State)"
    ]
    
    # Add Ground Truth
    images.append(gt_img)
    
    # Add others if they exist
    for p in [raw_path, mem_path, graph_path]:
        if os.path.exists(p):
            images.append(Image.open(p))
        else:
            print(f"Warning: Missing visualization file: {p}")
            images.append(Image.new("RGB", (500, 500), (0, 0, 0)))

    # Resize all to match the height of the graph image (usually the largest/cleanest aspect ratio)
    target_height = 800
    resized_images = []
    
    # Reference width from the Raw Map (index 1) which is typically square
    reference_width = target_height 
    
    for i, img in enumerate(images):
        # V22: Force Graph (Index 3) to be square to match Raw Map layout
        if i == 3: # Graph
             resized_images.append(img.resize((reference_width, target_height), Image.Resampling.LANCZOS))
        else:
            aspect = img.width / img.height
            new_w = int(target_height * aspect)
            # Capture width of Raw Map (index 1) to use as reference if needed
            if i == 1:
                reference_width = new_w
            resized_images.append(img.resize((new_w, target_height), Image.Resampling.LANCZOS))
        
    # Create combined image
    image_widths = [img.width for img in resized_images]
    total_width = sum(image_widths)
    header_height = 80
    footer_height = 400 # Slightly reduced from 450 to be more concise but still safe
    
    combined = Image.new("RGB", (total_width, target_height + header_height + footer_height), (20, 20, 20)) 
    
    x_offset = 0
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 36) # Increased title font size (was 28)
    except:
        font = ImageFont.load_default()
        
    # Draw Images and Titles
    for i, img in enumerate(resized_images):
        combined.paste(img, (x_offset, header_height))
        
        # Draw Title
        title = titles[i]
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (img.width - text_w) / 2
        draw.text((text_x, 20), title, fill="white", font=font)
        
        x_offset += img.width

    # Generate and Paste Footer Legend - PASSING COLUMN WIDTHS
    legend_img = create_legend_footer(image_widths, footer_height)
    combined.paste(legend_img, (0, target_height + header_height))
        
    save_path = os.path.join(output_dir, f"comparison_step_{step}_full_panel.png")
    combined.save(save_path)
    print(f"Saved comparison panel to {save_path}")
    return save_path

def main():
    """
    Main function to run the environment, generate visualizations, and collect metrics.
    """
    # --- Control GIF generation ---
    generate_gif_frames = False # Disabled for speed as requested
    gif_output_dir = "observation_visuals_gif_frames"
    if generate_gif_frames:
        os.makedirs(gif_output_dir, exist_ok=True)
        print(f"GIF frame generation is ON. Frames will be saved to '{gif_output_dir}'.")
        
    metrics_tracker = MetricsTracker()
    cumulative_resources_delivered = 0

    env_config = {
        'num_teams': 6,
        'width': 1000,
        'height': 1000,
        'sensing_range_fraction': 0.05,
        'num_agents_per_team': 15,
        'num_resources': 200, # Increased for larger map
        'num_obstacles': 15,  # Increased for larger map
        'max_steps': 2000, # Extended simulation steps
        'render_mode': "human", # Enable Pygame for ground truth
        'debug': False,
        'graph_connection_radius_factor': 0.75, 
        'cluster_aggressiveness': 4.0,
        'mem_skeleton_connection_factor': 2.0,
        'cluster_exclusion_radius_factor': 1.05,
        'recency_normalization_period': 250.0
    }

    # Suppress window if needed (optional, but good for scripts)
    os.environ["SDL_VIDEODRIVER"] = "dummy" 

    env = Swarm2DEnv(**env_config)
    agent_to_visualize = 0
    initial_team_size = env_config['num_agents_per_team']
    print("Environment created.")

    obs_list, info = env.reset()
    print("Environment reset.")
    
    visualized_agent_team_id = env.agents[agent_to_visualize]['team']
    env.single_agent_obs_idx = agent_to_visualize
    
    # --- Updated intervals for milestone PNGs (step 0, 1, then every 50 steps) ---
    steps_to_visualize = [0, 1] + list(range(50, 2001, 50)) # Include step 1 explicitly
    sequence_steps = list(range(0, 601, 100)) # Keep existing sequence steps if needed, or adjust
    world_bounds = {'width': env.width, 'height': env.height}
    
    comparison_image_paths = [] # Store paths for final sequence
    foveated_vis_output_dir = "foveated_visualization_output"
    os.makedirs(foveated_vis_output_dir, exist_ok=True)
    
    # --- NEW: Track agent path ---
    agent_path_history = [env.agents[agent_to_visualize]['pos']]

    # --- V6: Tunable visualization parameter ---
    # V27: Lower exponent to boost low-coverage visibility (non-linear scaling)
    coverage_exponent = 0.5 # Adjust this to make entities more/less bright (0.5=bright, 1.0=linear, >1.0=faint)
    visualization_gamma = 0.45 # V14: Gamma boost for raw map visuals. <1.0 boosts shadows.


    def generate_visuals(obs, agent_idx, step, output_dir="observation_visuals", path_history=None):
        if obs and len(obs) > agent_idx:
            agent_obs = obs[agent_idx]
            
            # --- ADD THIS LINE HERE ---
            if step % 1 == 0: # Only create panel on milestones to save time
                visualize_verification_proof(env, agent_obs, step, output_dir)
  
            agent_id = env.agents[agent_idx].get('id', agent_idx)
            agent_team_id = env.agents[agent_idx].get('team', -1) # Get agent's team_id
            # V13: Get the specific agent's observation radius for accurate visualization
            agent_obs_radius = env.agents[agent_idx].get('obs_radius', OBS_RADIUS)

            if 'map' in agent_obs:
                raw_map_tensor = agent_obs['map']
                C, H, W = raw_map_tensor.shape
                # Standard
                visualize_observation_map(raw_map_tensor, RAW_CH, agent_id, step, 'raw', 
                                          output_dir=output_dir, 
                                          obs_radius=agent_obs_radius, 
                                          recency_normalization_period=env_config.get('recency_normalization_period', 250.0),
                                          visualization_gamma=visualization_gamma)
                # Clean for comparison
                if step in steps_to_visualize:
                    visualize_observation_map(raw_map_tensor, RAW_CH, agent_id, step, 'raw', 
                                              output_dir=output_dir, 
                                              obs_radius=agent_obs_radius,
                                              recency_normalization_period=env_config.get('recency_normalization_period', 250.0),
                                              visualization_gamma=visualization_gamma,
                                              suppress_title=True, suppress_legend=True, filename_suffix="_clean")

            if 'memory_map' in agent_obs:
                coverage = agent_obs.get('coverage_map')
                # Standard
                visualize_observation_map(agent_obs['memory_map'], OCC_CH, agent_id, step, 'memory', 
                                          output_dir=output_dir, 
                                          recency_normalization_period=env_config.get('recency_normalization_period', 250.0), 
                                          coverage_map=coverage,
                                          coverage_intensity_exponent=coverage_exponent)
                # Clean for comparison
                if step in steps_to_visualize:
                    visualize_observation_map(agent_obs['memory_map'], OCC_CH, agent_id, step, 'memory', 
                                              output_dir=output_dir, 
                                              recency_normalization_period=env_config.get('recency_normalization_period', 250.0), 
                                              coverage_map=coverage,
                                              coverage_intensity_exponent=coverage_exponent,
                                              suppress_title=True, suppress_legend=True, filename_suffix="_clean")

            if 'graph' in agent_obs:
                # Standard
                visualize_graph_observation(agent_obs['graph'], agent_id, agent_team_id, step, world_bounds, env_config['max_steps'], env_config['recency_normalization_period'], output_dir=output_dir, path_history=path_history)
                
                # Clean for comparison
                if step in steps_to_visualize:
                    visualize_graph_observation(agent_obs['graph'], agent_id, agent_team_id, step, world_bounds, env_config['max_steps'], env_config['recency_normalization_period'], 
                                                output_dir=output_dir, path_history=path_history, 
                                                suppress_legend=True, suppress_title=True, filename_suffix="_clean")
                
                # Special visualization for sequence (no legend)
                if step in sequence_steps:
                    visualize_graph_observation(agent_obs['graph'], agent_id, agent_team_id, step, world_bounds, env_config['max_steps'], env_config['recency_normalization_period'], 
                                                output_dir=output_dir, path_history=path_history, 
                                                suppress_legend=True, suppress_title=True, filename_suffix="_no_legend")

    # Visualize at reset for both milestone and GIF
    # Ensure this runs after reset but before the main loop
    # And specifically for step 0
    if 0 in steps_to_visualize:
        # Create a dummy path history for step 0 if needed (it should be initialized above)
        print("Generating visualizations for Step 0...")
        generate_visuals(obs_list, agent_to_visualize, 0, output_dir=foveated_vis_output_dir, path_history=agent_path_history)
        # Create comparison panel for Step 0
        comp_path = create_comparison_image(env, agent_to_visualize, 0, foveated_vis_output_dir)
        comparison_image_paths.append(comp_path)

    if generate_gif_frames:
        generate_visuals(obs_list, agent_to_visualize, 0, output_dir=gif_output_dir, path_history=agent_path_history)
            
    # --- V2: Waypoints for Anarchy Symbol (Updated for 1000x1000) ---
    center_x, center_y = env.width / 2, env.height / 2
    # Scale radius based on map size
    radius = min(center_x, center_y) * 0.7

    # Circle points (smoother with more points)
    circle_waypoints = []
    for i in range(61): # Increased points for smoother large circle
        angle = (i / 60) * 2 * np.pi
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        circle_waypoints.append(np.array([x, y]))
        
    # 'A' points - made more distinct
    a_height = radius * 1.5
    a_width = radius * 1.2
    a_waypoints = [
        np.array([center_x - a_width / 2, center_y - a_height / 2]), # bottom-left
        np.array([center_x, center_y + a_height / 2]),               # top-center
        np.array([center_x + a_width / 2, center_y - a_height / 2]), # bottom-right
        # Lift pen to move to cross-bar start
        np.array([center_x + a_width / 2.5, center_y - a_height / 4]),
        # Draw cross-bar
        np.array([center_x - a_width / 2.5, center_y - a_height / 4]),
    ]
    
    # Final flourish - a small loop at the end
    flourish_waypoints = []
    start_pos = a_waypoints[-1]
    for i in range(11):
        angle = (i/10) * 2 * np.pi
        x = start_pos[0] + 30 * np.cos(angle)
        y = start_pos[1] + 30 * np.sin(angle)
        flourish_waypoints.append(np.array([x,y]))

    all_waypoints = circle_waypoints + a_waypoints + flourish_waypoints
    current_waypoint_idx = 0
    waypoint_tolerance = 100.0 # Increased tolerance
    path_completed = False
    
    # Stuck detection variables
    stuck_counter = 0
    last_positions_buffer = []
    STUCK_THRESHOLD_DIST = 10.0 # Minimal movement required
    STUCK_CHECK_INTERVAL = 20   # Steps to check over
    
    # V18: Advanced Unstuck State
    recovering_from_stuck = False
    recovery_timer = 0
    recovery_direction = np.zeros(2)

    for step in range(1, env_config['max_steps']):
        actions = []
        for i in range(env.num_agents):
            agent_pos = np.array(env.agents[i]['pos'])
            base_movement_action = np.zeros(2)

            if i == agent_to_visualize:
                # --- STUCK DETECTION ---
                last_positions_buffer.append(agent_pos.copy())
                if len(last_positions_buffer) > STUCK_CHECK_INTERVAL:
                    old_pos = last_positions_buffer.pop(0)
                    dist_moved = np.linalg.norm(agent_pos - old_pos)
                    if dist_moved < STUCK_THRESHOLD_DIST and not recovering_from_stuck:
                        stuck_counter += 1
                    else:
                        stuck_counter = max(0, stuck_counter - 1)
                
                # Activate recovery mode if stuck for too long
                if stuck_counter > 10 and not recovering_from_stuck:
                     recovering_from_stuck = True
                     recovery_timer = 30 # Spend 30 steps recovering
                     # Pick a random robust direction
                     angle = np.random.uniform(0, 2 * np.pi)
                     recovery_direction = np.array([np.cos(angle), np.sin(angle)])
                     print(f"Agent {i} IS STUCK. Initiating recovery maneuver.")
                     stuck_counter = 0

                # --- MOVEMENT LOGIC ---
                
                # 1. Recovery Mode takes precedence
                if recovering_from_stuck:
                    base_movement_action = recovery_direction
                    recovery_timer -= 1
                    if recovery_timer <= 0:
                        recovering_from_stuck = False
                        # Clear buffer to prevent immediate re-trigger
                        last_positions_buffer = []
                        print(f"Agent {i} recovery complete. Resuming path.")
                
                # 2. Roaming Mode (Spiral)
                elif path_completed:
                    # More active exploration: spiral outward pattern that continues moving
                    center = np.array([center_x, center_y])
                    
                    # Create a continuous spiral exploration pattern
                    # Angle increases continuously, radius grows slowly
                    spiral_angle = step * 0.1  # Continuous rotation (radians)
                    spiral_radius = 100 + (step % 400) * 0.5  # Slowly growing radius
                    target_pos = center + np.array([np.cos(spiral_angle) * spiral_radius, 
                                                     np.sin(spiral_angle) * spiral_radius])
                    direction = target_pos - agent_pos
                    distance = np.linalg.norm(direction)
                    
                    if distance > 20:  # Move towards spiral point
                        base_movement_action = direction / (distance + 1e-6) * 0.8  # Scale down for smoother movement
                    else:  # If close to spiral point, continue outward/around
                        # Continue in spiral direction
                        tangent = np.array([-np.sin(spiral_angle), np.cos(spiral_angle)])
                        base_movement_action = tangent * 0.6

                # 3. Path Following Mode
                else:
                    target_waypoint = all_waypoints[current_waypoint_idx]
                    direction = target_waypoint - agent_pos
                    distance = np.linalg.norm(direction)
                    
                    # Basic direction
                    base_movement_action = direction / (distance + 1e-6)

                    if distance < waypoint_tolerance:
                        current_waypoint_idx += 1
                        if current_waypoint_idx >= len(all_waypoints):
                            path_completed = True
                            print(f"[MOVEMENT DEBUG] Agent {i} completed Anarchy path, now roaming.")
                        else:
                            print(f"[MOVEMENT DEBUG] Agent {i} reached waypoint, moving to next ({current_waypoint_idx}/{len(all_waypoints)})")
            else:
                # Other agents: simple random movement
                base_movement_action = np.random.uniform(-1, 1, size=2)

            # --- V17: Simple Obstacle Avoidance using a Repulsion Field ---
            avoidance_vector = np.zeros(2)
            avoidance_radius = AGENT_RADIUS * 6  # Check for obstacles within 6 body radii
            
            for obs in env.obstacles:
                obs_pos = np.array(obs['pos'])
                vec_to_obs = obs_pos - agent_pos
                dist_to_obs = np.linalg.norm(vec_to_obs)
                
                # Consider the obstacle's radius in the distance check for more accuracy
                obstacle_radius = obs.get('radius', AGENT_RADIUS)
                effective_distance = dist_to_obs - obstacle_radius
                
                if effective_distance < avoidance_radius:
                    # Repulsion strength is inversely proportional to distance (stronger when closer)
                    # Using a squared relationship for a more pronounced effect up close.
                    repulsion_strength = (1 - (effective_distance / avoidance_radius))**2
                    
                    # Add a vector pointing directly away from the obstacle's center
                    avoidance_vector -= (vec_to_obs / (dist_to_obs + 1e-6)) * repulsion_strength
            
            # --- Combine base movement with avoidance and normalize ---
            # Combine the goal-directed movement with the avoidance vector.
            # Avoidance is given a higher weight to make it a priority.
            
            # If recovering, ignore standard avoidance (we want to force a move)
            if not recovering_from_stuck:
                if np.linalg.norm(avoidance_vector) > 0.1:
                    # Add tangential component to steer around
                    tangent = np.array([-avoidance_vector[1], avoidance_vector[0]])
                    # Steer in direction that aligns better with current goal
                    if np.dot(tangent, base_movement_action) < 0:
                        tangent = -tangent
                    
                    avoidance_vector += tangent * 0.8 # Increase tangent influence
    
                final_movement_vector = base_movement_action + avoidance_vector * 3.0
            else:
                final_movement_vector = base_movement_action
            
            # Normalize the final vector to ensure consistent speed
            final_norm = np.linalg.norm(final_movement_vector)
            if final_norm > 0:
                final_movement_vector /= final_norm

            actions.append({'movement': final_movement_vector, 'pickup': np.random.randint(0, 3)})

        obs_list, rewards, done, _, infos = env.step(actions)
        
        # --- METRICS COLLECTION ---
        metrics = {}
        # Team-level metrics from 'infos'
        delivered_this_step = len(infos.get("delivered_resource_ids_this_step", []))
        cumulative_resources_delivered += delivered_this_step
        metrics['resources_delivered'] = cumulative_resources_delivered

        # Calculate survival rate for the visualized agent's team
        alive_agents_on_team = sum(1 for a in env.agents if a and a.get('team') == visualized_agent_team_id and a.get('alive'))
        metrics['team_survival_rate'] = (alive_agents_on_team / initial_team_size) * 100 if initial_team_size > 0 else 0

        # Agent-specific metrics from observation
        if obs_list and len(obs_list) > agent_to_visualize:
            agent_obs = obs_list[agent_to_visualize]

            # 1. Exploration
            if 'memory_map' in agent_obs and 'explored' in OCC_CH:
                memory_map = agent_obs['memory_map']
                explored_channel = memory_map[OCC_CH['explored']].cpu().numpy()
                explored_mask = explored_channel <= 1.0
                total_cells = explored_channel.size
                if total_cells > 0:
                    metrics['exploration_pct'] = (np.sum(explored_mask) / total_cells) * 100
                else:
                    metrics['exploration_pct'] = 0

            # 2. Graph-based metrics
            if 'graph' in agent_obs and hasattr(agent_obs['graph'], 'num_nodes') and agent_obs['graph'].num_nodes > 0:
                graph = agent_obs['graph']
                metrics['total_nodes'] = graph.num_nodes

                # Node type breakdown
                node_features = graph.x.cpu().numpy()
                node_type_idx = MEM_NODE_FEAT_IDX['node_type_encoded']
                team_id_idx = MEM_NODE_FEAT_IDX['team_id']
                node_types = node_features[:, node_type_idx]
                
                agent_type_val = NODE_TYPE['agent']
                resource_type_val = NODE_TYPE['resource']
                
                agent_mask = node_types == agent_type_val
                
                if np.any(agent_mask):
                    agent_teams = node_features[agent_mask, team_id_idx]
                    metrics['ally_nodes'] = np.sum(agent_teams == visualized_agent_team_id)
                    metrics['enemy_nodes'] = np.sum(agent_teams != visualized_agent_team_id)
                else:
                    metrics['ally_nodes'] = 0
                    metrics['enemy_nodes'] = 0
                
                metrics['resource_nodes'] = np.sum(node_types == resource_type_val)

                # 3. Certainty
                last_seen_idx = MEM_NODE_FEAT_IDX['last_observed_step']
                steps_ago = step - node_features[:, last_seen_idx]
                recency = 1.0 - np.clip(steps_ago / env_config['recency_normalization_period'], 0.0, 1.0)
                metrics['avg_certainty'] = np.mean(recency)
            else:
                metrics['total_nodes'] = 0
                metrics['ally_nodes'] = 0
                metrics['enemy_nodes'] = 0
                metrics['resource_nodes'] = 0
                metrics['avg_certainty'] = 0
        
        metrics_tracker.record(step, metrics)


        # --- NEW: Append current agent position to path history ---
        current_agent_pos = env.agents[agent_to_visualize]['pos']
        agent_path_history.append(current_agent_pos)

        # --- V30: Save Raw Map EVERY STEP for dense sequence ---
        # The user requested specifically "raw_map" to be saved every step
        if obs_list and len(obs_list) > agent_to_visualize:
             agent_obs = obs_list[agent_to_visualize]
             agent_id = env.agents[agent_to_visualize].get('id', agent_to_visualize)
             agent_obs_radius = env.agents[agent_to_visualize].get('obs_radius', OBS_RADIUS)
             
             if 'map' in agent_obs:
                 raw_map_tensor = agent_obs['map']
                 # We use a specific subdirectory for these high-frequency frames to avoid clutter
                 raw_seq_dir = os.path.join(foveated_vis_output_dir, "raw_map_sequence")
                 os.makedirs(raw_seq_dir, exist_ok=True)
                 
                 # Call visualize directly for raw map only, suppressing extra outputs
                 visualize_observation_map(raw_map_tensor, RAW_CH, agent_id, step, 'raw', 
                                           output_dir=raw_seq_dir, 
                                           obs_radius=agent_obs_radius,
                                           recency_normalization_period=env_config.get('recency_normalization_period', 250.0),
                                           visualization_gamma=visualization_gamma,
                                           suppress_title=True, suppress_legend=True, # Minimalist for sequence
                                           filename_suffix="") # Keep standard naming "step_X.png"

        # --- UPDATED: Dual-track visualization ---
        # Milestone PNGs
        if step in steps_to_visualize:
             generate_visuals(obs_list, agent_to_visualize, step, output_dir=foveated_vis_output_dir, path_history=agent_path_history)
             # Create the comparison panel
             comp_path = create_comparison_image(env, agent_to_visualize, step, foveated_vis_output_dir)
             comparison_image_paths.append(comp_path)
        
        # GIF frames (every step)
        if generate_gif_frames:
            generate_visuals(obs_list, agent_to_visualize, step, output_dir=gif_output_dir, path_history=agent_path_history)
        
        if done:
            print(f"Episode finished early at step {step}.")
            break
            
        # Stop early if we have all needed steps to save time
        if step > max(steps_to_visualize) + 10:
            print("Captured all requested steps. Ending simulation.")
            break

    # --- PLOT AND SAVE METRICS AT THE END OF THE SIMULATION ---
    df_metrics = metrics_tracker.get_dataframe()
    plot_and_save_metrics(df_metrics, output_dir="metrics_visuals")

    env.close()
    print("Environment closed.")

    # --- FINAL SEQUENCE GENERATION ---
    print("\nCreating final Foveated Memory Sequence...")
    if comparison_image_paths:
        # Load all comparison panels
        panels = [Image.open(p) for p in comparison_image_paths]
        
        # Stack vertically
        total_w = max(img.width for img in panels)
        total_h = sum(img.height for img in panels)
        
        final_seq = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        y_off = 0
        for img in panels:
            # Center if widths differ (they shouldn't if generated same way, but just in case)
            x_off = (total_w - img.width) // 2
            final_seq.paste(img, (x_off, y_off))
            y_off += img.height
            
        seq_path = os.path.join(foveated_vis_output_dir, "foveated_memory_sequence.png")
        final_seq.save(seq_path)
        print(f"Saved final sequence to {seq_path}")

    # --- SEQUENCE IMAGE GENERATION ---
    print("\nCreating combined sequence image for graph visualization...")
    output_dir = foveated_vis_output_dir
    
    # 1. Generate the standalone legend
    legend_path = save_graph_legend_only(visualized_agent_team_id, output_dir=output_dir)
    
    # 2. Collect the no-legend images
    image_paths = []
    for step in sequence_steps:
        path = os.path.join(output_dir, f"agent_{agent_to_visualize}_graph_step_{step}_no_legend.png")
        if os.path.exists(path):
            image_paths.append(path)
        else:
            print(f"  - Missing image for step {step}: {path}")
            
    # 3. Create the composition
    if image_paths:
        sequence_output_path = os.path.join(output_dir, f"agent_{agent_to_visualize}_graph_sequence_0_to_600.png")
        create_sequence_composition(image_paths, legend_path, sequence_output_path)


    # --- GIF Creation ---
    if generate_gif_frames:
        print("\nCreating GIFs from generated visualizations...")
        
        # Helper function to create GIF from frames
        def create_gif_from_frames(prefix, gif_name, output_dir=foveated_vis_output_dir):
            images = []
            # Filter for PNGs that start with prefix and DO NOT end with 'clean.png'
            frame_files = [f for f in os.listdir(gif_output_dir) 
                          if f.startswith(prefix) and f.endswith('.png') and not f.endswith('clean.png')]
            # Sort files numerically to ensure correct order
            try:
                frame_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
            except ValueError as e:
                print(f"Error sorting frames for {gif_name}: {e}")
                return
            
            if not frame_files:
                print(f"No images found for {gif_name}.")
                return
            
            for filename in frame_files:
                images.append(imageio.imread(os.path.join(gif_output_dir, filename)))
            
            # Add a pause at the end of the GIF by duplicating the last frame
            for _ in range(10):  # Add 10 frames for a 1-second pause at 10 fps
                images.append(images[-1])
            
            gif_path = os.path.join(output_dir, gif_name)
            imageio.mimsave(gif_path, images, fps=10)
            print(f"Successfully created GIF: {gif_path}")
        
        # Create GIFs for graph, memory_map, and raw_map
        create_gif_from_frames('agent_0_graph_step_', 'agent_0_unified_graph.gif')
        create_gif_from_frames('agent_0_memory_map_step_', 'agent_0_memory_map.gif')
        create_gif_from_frames('agent_0_raw_map_step_', 'agent_0_raw_map.gif')
        create_gif_from_frames('agent_0_coverage_map_step_', 'agent_0_coverage_map.gif')


if __name__ == "__main__":
    main()

