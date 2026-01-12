import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_comparison_legend():
    # --- 1. Define Styles (Matching your comparison panel) ---
    
    # Simulation & Teams
    simulation_teams = [
        {'label': 'Team 0', 'color': '#006464'}, # Teal
        {'label': 'Team 1', 'color': '#5a3c1e'}, # Brown
        {'label': 'Team 2', 'color': '#ff69b4'}, # Hot Pink
        {'label': 'Team 3', 'color': '#4b0082'}, # Indigo
        {'label': 'Team 4', 'color': '#808000'}, # Olive
        {'label': 'Team 5', 'color': '#4682b4'}, # Steel Blue
    ]

    # Entity Key
    entity_key = [
        {'label': 'Self',         'color': 'yellow'},
        {'label': 'Ally',         'color': 'blue'},
        {'label': 'Enemy',        'color': 'red'},
        {'label': 'Resource',     'color': 'lime'},
        {'label': 'Coop Res',     'color': 'orange'},
        {'label': 'Hive (Ally)',  'color': 'cyan'},
        {'label': 'Hive (Enemy)', 'color': 'magenta'},
        {'label': 'Obstacle',     'color': 'lightgray'},
    ]

    # Memory Properties
    memory_properties = [
        {'label': 'Explored Ground',  'color': 'dimgray'}, # Dark Gray
        {'label': 'Unexplored (Void)','color': 'black'},
        # Note: "Brightness = Certainty" is usually text, but we can make a dummy patch or just text in the legend title/label
    ]

    # --- 2. Create Figure ---
    # Adjust size to fit 3-4 columns comfortably
    fig = plt.figure(figsize=(18, 3)) 
    ax = fig.add_subplot(111)
    ax.axis('off') # Hide axis

    # --- 3. Construct Handles ---
    
    sim_handles = []
    for t in simulation_teams:
        h = mpatches.Patch(color=t['color'], label=t['label'])
        sim_handles.append(h)

    entity_handles = []
    for e in entity_key:
        h = mpatches.Patch(color=e['color'], label=e['label'])
        entity_handles.append(h)

    mem_handles = []
    for m in memory_properties:
        h = mpatches.Patch(color=m['color'], label=m['label'])
        mem_handles.append(h)
    
    # Adding a dummy handle for the text annotation "Brightness = Certainty"
    # Or we can just add it as text to the plot, but let's try to put it in the legend for consistency
    # A common trick is a white patch with the label
    # mem_handles.append(mpatches.Patch(color='none', label='Brightness = Certainty'))


    # --- 4. Place Legends Horizontally ---
    
    # Legend 1: Teams (Left)
    l1 = ax.legend(handles=sim_handles, title="Simulation & Teams", 
                   loc='center left', bbox_to_anchor=(0.02, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.add_artist(l1)

    # Legend 2: Entity Key (Center)
    l2 = ax.legend(handles=entity_handles, title="Entity Key", 
                   loc='center', bbox_to_anchor=(0.40, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=4) # 4 columns to be wider
    ax.add_artist(l2)

    # Legend 3: Memory Properties (Right)
    # We add the text annotation manually below the legend or as a label
    l3 = ax.legend(handles=mem_handles, title="Memory Properties\n(Brightness = Certainty)", 
                   loc='center right', bbox_to_anchor=(0.90, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=1)
    
    # Note: "Graph Representation" part from your image is just text about Nodes/Z-Axis. 
    # If you want that included as a 4th legend column, we can add it.
    # Assuming you want the visual color keys primarily.

    # --- 5. Save ---
    plt.tight_layout()
    output_filename = 'comparison_legend_horizontal.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='black') # Dark background to match
    print(f"Legend saved as {output_filename}")

if __name__ == "__main__":
    # Set dark background style to match the provided image
    plt.style.use('dark_background')
    create_comparison_legend()


