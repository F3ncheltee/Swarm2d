import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Re-use color definitions from visualize_obs.py to ensure exact match ---
ENTITY_COLOR_MAP = {
    'self_presence': {'pil': (255, 255, 0, 255), 'mpl': [1, 1, 0]},
    'ally_presence': {'pil': (80, 80, 255, 255), 'mpl': [0.3, 0.3, 1]},
    'enemy_presence': {'pil': (255, 0, 0, 255), 'mpl': [1, 0, 0]},
    'resource_presence': {'pil': (0, 255, 0, 255), 'mpl': [0, 1, 0]},
    'coop_resource_presence': {'pil': (255, 165, 0, 255), 'mpl': [1.0, 0.647, 0.0]},
    'hive_ally_presence': {'pil': (0, 255, 255, 255), 'mpl': [0, 1, 1]},
    'hive_enemy_presence': {'pil': (255, 0, 255, 255), 'mpl': [1, 0, 1]},
    'obstacle_presence': {'pil': (220, 220, 220, 255), 'mpl': [0.86, 0.86, 0.86]},
    'explored_ground': {'pil': (60, 55, 50, 255), 'mpl': [0.23, 0.21, 0.19]},
}

TEAM_COLORS = {
    0: [0, 150, 136, 255],   
    1: [121, 85, 72, 255],   
    2: [233, 30, 99, 255],   
    3: [156, 39, 176, 255],  
    4: [205, 220, 57, 255],  
    5: [63, 81, 181, 255]    
}

def create_exact_legend_footer(width=2000, height=350): # Width matches approx full panel width
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except:
        font_large = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Calculate column starts (roughly based on visual weights)
    col1_x = 0
    col2_x = int(width * 0.25)
    col3_x = int(width * 0.50)
    col4_x = int(width * 0.75)

    # --- Column 1: Ground Truth & Team IDs ---
    draw.text((col1_x + 20, 20), "Simulation & Teams", fill="white", font=font_large)
    draw.text((col1_x + 20, 70), "Map: 1000x1000 | Teams: 6", fill="#CCCCCC", font=font_title)
    
    team_swatch_size = 25
    start_y = 120
    for team_id in range(6):
        if team_id in TEAM_COLORS:
            rgba = TEAM_COLORS[team_id]
            color = tuple(int(c) for c in rgba) # PIL expects RGB tuple
            
            col_offset = (team_id % 3) * 140
            row_offset = (team_id // 3) * 40
            
            x_pos = col1_x + 20 + col_offset
            y_pos = start_y + row_offset
            
            draw.rectangle([x_pos, y_pos, x_pos + team_swatch_size, y_pos + team_swatch_size], fill=color[:3], outline="white")
            draw.text((x_pos + 35, y_pos), f"Team {team_id}", fill="white", font=font_small)

    # --- Column 2: Entity Key ---
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
    for i, (name, color) in enumerate(entities):
        # 2-column layout for entities
        x_pos = col2_x + 20 + (220 if i >= 4 else 0)
        y_pos = curr_y + ((i % 4) * 40)
        
        draw.rectangle([x_pos, y_pos, x_pos + 25, y_pos + 25], fill=color[:3], outline="white")
        draw.text((x_pos + 35, y_pos), name, fill="white", font=font_title)

    # --- Column 3: Memory Properties ---
    draw.text((col3_x + 20, 20), "Memory Properties", fill="white", font=font_large)
    
    draw.rectangle([col3_x + 20, 70, col3_x + 45, 95], fill=ENTITY_COLOR_MAP['explored_ground']['pil'][:3], outline="gray")
    draw.text((col3_x + 55, 70), "Explored Ground", fill="white", font=font_title)
    
    draw.rectangle([col3_x + 20, 110, col3_x + 45, 135], fill=(0, 0, 0), outline="gray")
    draw.text((col3_x + 55, 110), "Unexplored (Void)", fill="white", font=font_title)
    
    draw.text((col3_x + 20, 160), "Brightness = Certainty", fill="#AAAAAA", font=font_small)

    # --- Column 4: Graph Representation ---
    draw.text((col4_x + 20, 20), "Graph Representation", fill="white", font=font_large)
    
    curr_y = 70
    draw.text((col4_x + 20, curr_y), "Nodes:", fill="white", font=font_title)
    
    node_items = [
        ('Agent', 'circle', 'white'),
        ('Hive', 'square', 'white'),
        ('Fovea', 'circle', 'yellow'), 
    ]
    
    node_y = curr_y + 40
    for i, (name, shape, color) in enumerate(node_items):
        y_pos = node_y + (i * 35)
        shape_x = col4_x + 20
        shape_y = y_pos + 5
        
        if shape == 'circle':
            draw.ellipse([shape_x, shape_y, shape_x + 15, shape_y + 15], outline="white", width=2)
        elif shape == 'square':
            draw.rectangle([shape_x, shape_y, shape_x + 15, shape_y + 15], outline="white", width=2)
            
        draw.text((col4_x + 45, y_pos), name, fill="white", font=font_small)

    draw.text((col4_x + 20, node_y + 120), "Z-Axis: Knowledge Certainty", fill="#AAAAAA", font=font_small)

    return img

def create_title_strip(width=2000, height=80):
    # Match the background color of the comparison panels exactly
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        # Reduce font size slightly to prevent overflow
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
        
    titles = [
        "1. Ground Truth (Global)", 
        "2. Raw Sensory Map (Local 32x32)", 
        "3. Memory Map (Global Model 64x64)", 
        "4. Foveated Graph (Global & Local State)"
    ]
    
    # Precise column calculation matching `create_comparison_image` resizing logic
    # In `create_comparison_image`, each image is resized to roughly `target_height` aspect ratio
    # but for simplicity, let's assume equal width distribution as a good approximation for the title strip.
    # However, to avoid overlapping, we should ensure the text is centered within its quadrant.
    
    col_width = width // 4
    for i, title in enumerate(titles):
        # Calculate text size using textbbox
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        
        # Calculate center position for this column
        center_x = (i * col_width) + (col_width // 2)
        
        # Position text centered on the column
        text_x = center_x - (text_w // 2)
        
        # Draw text
        draw.text((text_x, 20), title, fill="white", font=font)
        
    return img

if __name__ == "__main__":
    # Generate Legend Only
    legend_img = create_exact_legend_footer()
    legend_img.save("comparison_legend_only.png")
    print("Saved comparison_legend_only.png")
    
    # Generate Title Strip Only
    title_img = create_title_strip()
    title_img.save("comparison_titles_only.png")
    print("Saved comparison_titles_only.png")
