import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def create_horizontal_legend():
    # --- 1. Define Styles (Matching your foveated graph) ---
    
    # Node Types
    node_types = [
        {'label': 'Self Agent', 'color': 'yellow', 'marker': 'o'},
        {'label': 'Hive',       'color': 'gray',   'marker': 's'}, # Square for hive
        {'label': 'Resource',   'color': 'lime',   'marker': 'o'},
        {'label': 'Co-op Res',  'color': 'orange', 'marker': 'o'},
        {'label': 'Obstacle',   'color': 'lightgray', 'marker': 'o'},
        {'label': 'Live Fovea', 'color': 'none',   'marker': 'o', 'edgecolor': 'black', 'linewidth': 1.5} # Outline only
    ]

    # Team Colors
    team_colors = [
        {'label': 'Team 0 (Ally)', 'color': '#006464'}, # Teal
        {'label': 'Team 1',        'color': '#5a3c1e'}, # Brown
        {'label': 'Team 2',        'color': '#ff69b4'}, # Hot Pink
        {'label': 'Team 3',        'color': '#4b0082'}, # Indigo
        {'label': 'Team 4',        'color': '#808000'}, # Olive
        {'label': 'Team 5',        'color': '#4682b4'}, # Steel Blue
    ]

    # Edge Types
    edge_types = [
        {'label': 'Ego Connection',    'color': '#ffd700'}, # Gold
        {'label': 'Kinematic',         'color': '#ff6347'}, # Tomato
        {'label': 'Affiliation',       'color': '#40e0d0'}, # Turquoise
        {'label': 'Shared Intent',     'color': '#afeeee'}, # Pale Turquoise
        {'label': 'Combat',            'color': '#ff4500'}, # Orange Red
        {'label': 'Ally',              'color': 'blue'},
        {'label': 'Enemy',             'color': 'red'},
        {'label': 'Resource',          'color': 'green'},
        {'label': 'Hive',              'color': 'cyan'},
        {'label': 'Memory-to-Memory',  'color': '#f5deb3'}, # Wheat
    ]

    # --- 2. Create Figure ---
    fig = plt.figure(figsize=(16, 4)) # Wide and short
    ax = fig.add_subplot(111)
    ax.axis('off') # Hide axis

    # --- 3. Construct Handles ---
    
    # Node Handles
    node_handles = []
    for n in node_types:
        if n['label'] == 'Live Fovea':
            # Create a circle with an edge but no face
            h = mlines.Line2D([], [], color='white', marker='o', 
                              markerfacecolor='none', markeredgecolor='black', 
                              markersize=10, markeredgewidth=1.5, label=n['label'], linestyle='None')
        else:
            h = mlines.Line2D([], [], color='white', marker=n['marker'], 
                              markerfacecolor=n['color'], markersize=10, 
                              label=n['label'], linestyle='None')
        node_handles.append(h)

    # Team Handles
    team_handles = []
    for t in team_colors:
        # Using patches (rectangles) for team colors often looks better/standard for "Team" legends
        h = mpatches.Patch(color=t['color'], label=t['label'])
        team_handles.append(h)

    # Edge Handles
    edge_handles = []
    for e in edge_types:
        h = mlines.Line2D([], [], color=e['color'], linewidth=2, label=e['label'])
        edge_handles.append(h)

    # --- 4. Place Legends Horizontally ---
    
    # Legend 1: Node Types (Left)
    l1 = ax.legend(handles=node_handles, title="Node Types", 
                   loc='center left', bbox_to_anchor=(0.02, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.add_artist(l1)

    # Legend 2: Team Colors (Center)
    l2 = ax.legend(handles=team_handles, title="Team Colors", 
                   loc='center', bbox_to_anchor=(0.42, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=2)
    ax.add_artist(l2)

    # Legend 3: Edge Types (Right)
    l3 = ax.legend(handles=edge_handles, title="Edge Types", 
                   loc='center right', bbox_to_anchor=(0.98, 0.5), 
                   frameon=True, fancybox=True, shadow=True, ncol=2)

    # --- 5. Save ---
    plt.tight_layout()
    output_filename = 'graph_legend_horizontal.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Legend saved as {output_filename}")

if __name__ == "__main__":
    create_horizontal_legend()


