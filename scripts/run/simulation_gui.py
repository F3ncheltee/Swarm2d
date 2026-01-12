"""
Main graphical user interface (GUI) for running and configuring Swarm2d simulations.

This module provides a comprehensive PyQt6-based application that allows for
real-time configuration of all environment parameters, policy selection for
each team, and visualization of key performance metrics.
"""
import sys
import os
import time
import json
import traceback
import threading

# Add project root to the Python path to allow running from the Swarm2d folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, QFileDialog, QComboBox, QStatusBar, QTabWidget, QLineEdit, QColorDialog, QListWidget, QStackedWidget, QCheckBox, QSpinBox, QGroupBox, QDoubleSpinBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

import inspect
import importlib
import pyqtgraph as pg
import numpy as np
import Swarm2d.constants as constants_module
import os
import imageio

from env.env import Swarm2DEnv
from gui_config import get_default_config
from gui_widgets import CollapsibleBox, create_parameter_widget

# Import visualization functions
from Swarm2d.visualize_obs import visualize_observation_map, visualize_graph_observation, ENTITY_COLOR_MAP
from Swarm2d.constants import RAW_CH, OCC_CH, NODE_TYPE, MEM_NODE_FEAT_IDX
# The direct import below is no longer needed as policies are loaded dynamically
# from Swarm2d.policies.heuristicPolicy.simple_heuristic_policy import SimpleHeuristicPolicy
# from Swarm2d.scenarios.tuning_scenarios import SCENARIOS


# --- DEPRECATED (Superceded by new REWARD_CONFIG system) ---
# TUNABLE_REWARDS = [
#     "r_progress", "r_delivery", "r_exploration_intrinsic", "r_resource_found",
#     "r_obstacle_found", "r_enemy_found", "r_hive_found", "r_grapple_control",
#     "r_grapple_controlled", "r_hive_total"
# ]
# 
# BASE_REWARD_CONSTANTS = {
#     'r_delivery': 'DELIVERY_REWARD',
#     'r_death': 'AGENT_DEATH_PENALTY',
#     'r_teammate_lost_nearby': 'TEAMMATE_DEATH_VICINITY_PENALTY'
# }
# 
# # Inconsistent names require a manual mapping for non-tunable rewards.
# FIXED_REWARD_CONSTANTS = {
#     "r_attachment": "REWARD_SCALE_ATTACHMENT",
#     "r_combat_win": "REWARD_SCALE_COMBAT_WIN",
#     "r_combat_lose": "REWARD_SCALE_COMBAT_LOSE",
#     "r_combat_continuous": "NEW_REWARD_SCALE_COMBAT_CONTINUOUS",
#     "r_hive_win": "REWARD_SCALE_HIVE_WIN",
#     "r_hive_lose": "REWARD_SCALE_HIVE_LOSE",
#     "r_grapple_break": "REWARD_SCALE_GRAPPLE_BREAK",
#     "r_torque_win": "REWARD_SCALE_TORQUE_WIN",
#     "r_death": "AGENT_DEATH_PENALTY",
#     "r_teammate_lost_nearby": "TEAMMATE_DEATH_VICINITY_PENALTY"
# }
# --- END DEPRECATED ---

class SimulationRunner(QThread):
    """
    Runs the Swarm2D simulation in a separate thread to keep the GUI responsive.

    This thread handles the simulation loop, including resetting the environment,
    getting actions from policies, stepping the environment, and collecting metrics.

    Signals:
        status_changed (pyqtSignal): Emits status messages (e.g., "Running", "Stopped").
        step_info (pyqtSignal): Emits the current step number.
        metrics_updated (pyqtSignal): Emits a dictionary of collected metrics each step.
    """
    status_changed = pyqtSignal(str)
    step_info = pyqtSignal(int)
    metrics_updated = pyqtSignal(dict)
    observation_updated = pyqtSignal(list, int)  # Emits (observations_list, current_step)

    def __init__(self, env, config, team_policies):
        super().__init__()
        self.env = env
        self.config = config
        self.team_policies = team_policies
        self.policies = {} # Will hold instantiated policy objects
        self.running = False
        self.current_observations = None
        self.current_step = 0

    def run(self):
        """The main simulation loop."""
        self.running = True
        try:
            obs, _ = self.env.reset()

            # --- Instantiate policies dynamically ---
            # This allows for selecting different behaviors for each team from the GUI.
            for team_id_str, policy_name in self.team_policies.items():
                team_id = int(team_id_str)
                try:
                    # --- NEW: Handle 'random' policy as a special case ---
                    if policy_name == 'random':
                        class RandomPolicy:
                            def __init__(self, env, team_id):
                                self.env = env
                                self.team_id = team_id
                            def get_actions(self, obs, agent_ids):
                                return [self.env.action_space.sample() for _ in agent_ids]
                        self.policies[team_id] = RandomPolicy(self.env, team_id=team_id)
                        print(f"Loaded RandomPolicy for team {team_id}.")
                        continue

                    # Construct the module path based on the directory structure.
                    policy_module_path = f"Swarm2d.policies.working_policies.{policy_name}"
                    
                    # If the policy name points to a directory, assume the actual module is inside
                    # with the same name + "_policy.py". e.g., 'randomPolicy' -> 'randomPolicy/random_policy.py'
                    potential_module_path = os.path.join(os.path.dirname(__file__), '..', 'policies', 'working_policies', policy_name)
                    if os.path.isdir(potential_module_path):
                        inferred_module_name = f"{policy_name.lower().replace('policy', '')}_policy"
                        policy_module_path = f"Swarm2d.policies.working_policies.{policy_name}.{inferred_module_name}"
                        class_name_base = inferred_module_name
                    else:
                        class_name_base = policy_name

                    # Import the module dynamically.
                    module = importlib.import_module(policy_module_path)
                    importlib.reload(module) # Reload to pick up any code changes without restarting the GUI.
                    
                    # Find the policy class within the module by convention (e.g., 'simple_heuristic' -> 'SimpleHeuristicPolicy').
                    class_name = ''.join(word.capitalize() for word in class_name_base.split('_'))
                    if not class_name.endswith("Policy"):
                         class_name += "Policy"

                    PolicyClass = getattr(module, class_name)
                    self.policies[team_id] = PolicyClass(self.env, team_id=team_id)

                except (ImportError, AttributeError, ModuleNotFoundError) as e:
                    print(f"Warning: Could not load policy '{policy_name}' for team {team_id}. Error: {e}. Using fallback.")
                    # Define a fallback policy that does nothing if the specified policy fails to load.
                    class DoNothingPolicy:
                        def get_actions(self, obs, agent_ids):
                            return [{"movement": np.zeros(2), "pickup": 0} for _ in agent_ids]
                    self.policies[team_id] = DoNothingPolicy()
            
            step = 0
            while self.running:
                all_actions = [None] * self.env.num_agents
                
                # Group agent IDs by their respective teams to query the correct policy.
                agent_ids_by_team = {}
                for i in range(self.env.num_agents):
                    team_id = self.env.agents[i]['team']
                    if team_id not in agent_ids_by_team:
                        agent_ids_by_team[team_id] = []
                    agent_ids_by_team[team_id].append(i)

                # Get actions from each team's assigned policy.
                for team_id, agent_ids in agent_ids_by_team.items():
                    policy = self.policies.get(team_id)
                    if policy:
                        # Pass the global observation and the specific agent_ids this policy is responsible for.
                        team_actions = policy.get_actions(obs, agent_ids)
                        for i, agent_id in enumerate(agent_ids):
                            all_actions[agent_id] = team_actions[i]
                    else: # Fallback for unimplemented or failed-to-load policies.
                        for agent_id in agent_ids:
                            all_actions[agent_id] = self.env.action_space.sample()

                obs, rewards, terminated, truncated, infos = self.env.step(all_actions)
                
                # Store observations for visualization
                self.current_observations = obs
                self.current_step = step
                
                if self.config.get('render_pygame_window', False):
                    self.env.render()

                self.step_info.emit(step)

                # Collect and emit metrics for the GUI to display.
                metrics = self.collect_metrics(rewards, infos)
                self.metrics_updated.emit(metrics)
                
                # Emit observations for visualization
                self.observation_updated.emit(obs, step)

                step += 1
                time.sleep(1 / self.config.get('FPS', 30))
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    step = 0
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in simulation thread: {e}\n{tb}")
            self.status_changed.emit(f"Error: {e}")
        finally:
            # The environment is owned by the main GUI window and should not be closed by the thread.
            # The main window will handle closing the environment when it's stopped or the app closes.
            self.status_changed.emit("Simulation Stopped")

    def stop(self):
        """Stops the simulation loop."""
        self.running = False

    def collect_metrics(self, rewards, infos):
        """
        Aggregates and calculates metrics from the environment's outputs.

        Args:
            rewards (list): A list of reward dictionaries from the environment.
            infos (dict): An info dictionary from the environment.

        Returns:
            dict: A dictionary of aggregated metrics.
        """
        metrics = {}
        num_teams = self.env.num_teams
        
        # Initialize metrics
        for i in range(num_teams):
            metrics[f'team_{i}_reward'] = 0
            metrics[f'team_{i}_agents_alive'] = 0
            metrics[f'team_{i}_hive_health'] = 0

        # --- Calculate Per-Team Metrics ---
        # Agent-based metrics
        for i, agent in enumerate(self.env.agents):
            if agent and agent.get('alive'):
                team_id = agent['team']
                # Accumulate rewards
                total_reward = sum(rewards[i].values())
                metrics[f'team_{team_id}_reward'] += total_reward
                # Count agents alive
                metrics[f'team_{team_id}_agents_alive'] += 1

        # Hive-based metrics
        for i in range(num_teams):
            hive = self.env.hives.get(i)
            if hive and hive.get('state') == 'active':
                metrics[f'team_{i}_hive_health'] = hive.get('health', 0)

        # --- Global Metrics ---
        metrics['resources_delivered'] = len(infos.get("delivered_resource_ids_this_step", []))
        metrics['resources_picked'] = self.env.resources_picked_count

        # --- NEW Detailed Combat/Interaction Metrics from infos dict ---
        kills_by_team = infos.get('kills_by_team', [0] * num_teams)
        deaths_by_team = infos.get('deaths_by_team', [0] * num_teams)
        damage_by_team = infos.get('damage_by_team', [0.0] * num_teams)
        grapples_initiated = infos.get('grapples_initiated_by_team', [0] * num_teams)
        grapples_broken = infos.get('grapples_broken_by_team', [0] * num_teams)
        hive_damage_by_team = infos.get('hive_damage_by_team', [0.0] * num_teams)

        for i in range(num_teams):
            metrics[f'team_{i}_kills'] = kills_by_team[i]
            metrics[f'team_{i}_deaths'] = deaths_by_team[i]
            metrics[f'team_{i}_damage_dealt'] = damage_by_team[i]
            metrics[f'team_{i}_grapples_initiated'] = grapples_initiated[i]
            metrics[f'team_{i}_grapples_broken'] = grapples_broken[i]
            metrics[f'team_{i}_hive_damage'] = hive_damage_by_team[i]

        return metrics


class SwarmSimGUI(QMainWindow):
    """
    The main window for the Swarm2d Simulation GUI.

    This class sets up the entire user interface, including configuration tabs,
    team setup panels, and metrics plots. It handles user interactions,
    manages the simulation lifecycle (start, stop), and updates the display
    with data from the simulation thread.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm2d Simulation GUI")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { color: #d3d3d3; background-color: #3c3f41; }
            QLabel { font-size: 14px; }
            QPushButton { background-color: #555555; border: 1px solid #777777; padding: 5px; font-size: 14px; }
            QPushButton:hover { background-color: #666666; }
            QPushButton:pressed { background-color: #777777; }
            QCheckBox { spacing: 5px; font-size: 14px; }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 2px solid #777777;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #d3d3d3;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border: 2px solid #005c99;
            }
        """)

        # --- Initialize Core Components ---
        self.sim_thread = None
        self.env_instance = None
        self.base_config = get_default_config()
        self.team_overrides = {}
        self.available_policies = self._discover_policies()
        self.scenarios = self._load_scenarios()
        self.metric_checkboxes = {}
        
        # --- Visualization State ---
        self.current_observations = None
        self.current_step = 0
        self.visualization_output_dir = "observation_visuals"
        self.gif_frame_dir = "observation_visuals_gif_frames"
        self.gif_frames = []  # Store frames for GIF generation

        # --- Setup the UI ---
        self._init_ui()
        self._connect_signals()

        # --- Load Initial Data ---
        self.load_defaults()

    def _init_ui(self):
        """Initializes the main UI layout and creates all UI elements."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.tabs = QTabWidget()
        self.tabs.setMaximumWidth(600)  # Constrain width to prevent excessive expansion
        main_layout.addWidget(self.tabs) # Stretch factor is not needed with max width

        self._create_config_tabs(self.tabs)
        self._create_teams_tab()
        self._create_visualization_tab()
        
        # The metrics tab is now a panel on the right
        metrics_panel = self._create_metrics_panel()
        main_layout.addWidget(metrics_panel)

    def _load_scenarios(self):
        """Scans the 'presets' directory to find and load available .json scenario files."""
        scenarios = {}
        try:
            presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
            if os.path.exists(presets_dir):
                for item in os.scandir(presets_dir):
                    if item.is_file() and item.name.endswith('.json'):
                        scenario_name = os.path.splitext(item.name)[0]
                        try:
                            with open(item.path, 'r') as f:
                                scenarios[scenario_name] = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse scenario file: {item.name}")
        except Exception as e:
            print(f"Warning: Could not discover scenarios: {e}")
        return scenarios

    def _discover_policies(self):
        """Scans the 'policies/working_policies' directory to find available policies."""
        policies = []
        try:
            policies_path = os.path.join(os.path.dirname(__file__), 'policies', 'working_policies')
            if os.path.exists(policies_path):
                for item in os.scandir(policies_path):
                    # Add directories (like 'randomPolicy') and .py files (like 'simple_heuristic_policy.py')
                    if item.is_dir() and not item.name.startswith('__'):
                        policies.append(item.name)
                    elif item.is_file() and item.name.endswith('.py') and not item.name.startswith('__'):
                        policy_name = item.name[:-3] # Remove .py
                        policies.append(policy_name)
        except Exception as e:
            print(f"Warning: Could not discover policies: {e}")
        
        # Add a default 'random' policy as a fallback
        if "random" not in policies:
            policies.append("random")
            
        return sorted(list(set(policies))) # Sort and remove duplicates

    def _connect_signals(self):
        """Connects all widget signals (e.g., button clicks, slider changes) to their handler methods."""
        self.scenario_selector.currentIndexChanged.connect(self.on_scenario_selected)
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.save_button.clicked.connect(self.save_preset)
        self.load_button.clicked.connect(self.load_preset)
        self.defaults_button.clicked.connect(self.load_defaults)

        # Connect the num_teams widget specifically to update the team panel
        num_teams_widget, _ = self.find_widget_and_details("num_teams")
        if num_teams_widget:
            # This is a simplified connection; assuming 'slider' is the primary input
            num_teams_widget['slider'].valueChanged.connect(self.update_team_setup_panel)


        # We must connect the global parameter widgets after they are created
        def connect_all_widgets(widget_dict):
            for key, widget in widget_dict.items():
                if isinstance(widget, dict) and "label" not in widget: # It's a category dict
                    connect_all_widgets(widget)
                else:
                    self.connect_widget_signal(widget, key)
        connect_all_widgets(self.widgets)

    def _create_config_tabs(self, parent_tab_widget):
        """Creates the main 'Configuration' tab which contains nested tabs for different parameter categories."""
        config_tab_widget = QWidget()
        parent_tab_widget.addTab(config_tab_widget, "Configuration")
        config_layout = QVBoxLayout(config_tab_widget)

        # Create the nested tab widget
        nested_tabs = QTabWidget()
        config_layout.addWidget(nested_tabs)

        # --- Create individual tabs ---
        self._create_general_tab(nested_tabs)
        self._create_physics_tab(nested_tabs)
        self._create_spawning_tab(nested_tabs)
        self._create_constants_tab(nested_tabs)
        self._create_observations_tab(nested_tabs)
        self._create_rewards_tab_unified(nested_tabs) # Use the new unified rewards tab

        # --- Main Control Buttons ---
        button_layout = self._create_main_buttons()
        config_layout.addLayout(button_layout)

    def _create_tab_with_categories(self, tab_widget, tab_name, categories):
        """
        Helper function to create a tab and populate it with parameter widgets
        for a list of specified categories from the default config.
        """
        tab = QWidget()
        tab_widget.addTab(tab, tab_name)
        main_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)
        
        full_config = get_default_config()
        
        # --- Scenario and Policy Selection (only for the first tab) ---
        if tab_name == "General & Rendering":
            top_layout = QHBoxLayout()
            top_layout.addWidget(QLabel("Load Scenario:"))
            self.scenario_selector = QComboBox()
            self.scenario_selector.addItem("Select a scenario...")
            self.scenario_selector.addItems(list(self.scenarios.keys()))
            top_layout.addWidget(self.scenario_selector)
            
            top_layout.addWidget(QLabel("Policy:"))
            self.policy_selector = QComboBox()
            self.policy_selector.addItems(self.available_policies)
            top_layout.addWidget(self.policy_selector)
            layout.addLayout(top_layout)

        # Create widgets for the specified categories
        if not hasattr(self, 'widgets'): self.widgets = {}
        
        for category_name in categories:
            if category_name in full_config:
                category_config = {category_name: full_config[category_name]}
                self._recursively_setup_widgets(layout, category_config, self.widgets)
        
        layout.addStretch()

    def _create_general_tab(self, tab_widget):
        """Creates the 'General & Rendering' configuration tab."""
        self._create_tab_with_categories(tab_widget, "General & Rendering", ["General", "Rendering & Debug"])

    def _create_physics_tab(self, tab_widget):
        """Creates the 'Physics' configuration tab."""
        self._create_tab_with_categories(tab_widget, "Physics", ["Physics"])

    def _create_spawning_tab(self, tab_widget):
        """Creates the 'Spawning' configuration tab."""
        self._create_tab_with_categories(tab_widget, "Spawning", ["Spawning"])

    def _create_constants_tab(self, tab_widget):
        """Creates the 'Constants' configuration tab."""
        self._create_tab_with_categories(tab_widget, "Constants", ["Constants"])

    def _create_observations_tab(self, tab_widget):
        """Creates the 'Observations' configuration tab."""
        self._create_tab_with_categories(tab_widget, "Observations", ["Observations", "Observation Settings"])
        
    def _create_rewards_tab(self, tab_widget):
        """DEPRECATED: Creates the 'Rewards' configuration tab."""
        self._create_tab_with_categories(tab_widget, "Rewards", ["Rewards"])

    def _create_rewards_tab_unified(self, tab_widget):
        """Creates a unified 'Rewards' tab based on the REWARD_CONFIG in the constants module."""
        tab = QWidget()
        tab_widget.addTab(tab, "Rewards")
        main_layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)

        if not hasattr(self, 'widgets'): self.widgets = {}
        if 'Rewards' not in self.widgets: self.widgets['Rewards'] = {}

        # The new system uses a single source of truth for rewards
        reward_config_from_constants = constants_module.REWARD_CONFIG
        
        # We still get the default GUI config structure for 'Rewards'
        gui_reward_config_template = get_default_config().get("Rewards", {})

        # Create a single collapsible box for all tunable rewards
        rewards_box = CollapsibleBox("Reward Multipliers")
        layout.addWidget(rewards_box)

        # Iterate through all rewards defined in the master list
        for key in constants_module.REWARD_COMPONENT_KEYS:
            if key in reward_config_from_constants:
                # Use the template from gui_config if available, otherwise create a default
                params = gui_reward_config_template.get(key, {
                    'type': 'reward', # A sensible default type for multipliers
                    'value': reward_config_from_constants[key]['default_multiplier'],
                    'range': [0.0, 10.0], 
                    'step': 0.1
                })
                
                # Set the 'value' from our new constants config to ensure it's the default
                params['value'] = reward_config_from_constants[key]['default_multiplier']

                # Create a more descriptive label
                base_val = reward_config_from_constants[key]['default_value']
                label_text = f"{key} (Base: {base_val})"
                
                # Pass the base_val to the widget creator
                container_widget, widget_group = create_parameter_widget(label_text, params, base_value=base_val)
                rewards_box.add_widget(container_widget)
                self.widgets['Rewards'][key] = widget_group
        
        layout.addStretch()

    def _create_teams_tab(self):
        """Creates the 'Team Setup' tab with a list of teams and their specific settings."""
        tab = QWidget()
        self.tabs.addTab(tab, "Team Setup")
        main_layout = QHBoxLayout(tab)

        self.team_list_widget = QListWidget()
        self.team_list_widget.setMaximumWidth(200)
        main_layout.addWidget(self.team_list_widget, 1)

        self.team_stacked_widget = QStackedWidget()
        main_layout.addWidget(self.team_stacked_widget, 4)
        
        self.team_parameter_widgets = {}

    def _create_metrics_panel(self):
        """Creates the metrics panel with a plot widget and controls."""
        metrics_panel_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_panel_widget)

        metrics_controls_layout = QVBoxLayout()
        metrics_layout.addLayout(metrics_controls_layout, 1)

        # This layout will hold the checkboxes
        self.metrics_checkbox_layout = QVBoxLayout()
        metrics_controls_layout.addLayout(self.metrics_checkbox_layout)
        metrics_controls_layout.addStretch()
        
        self.plot_widget = pg.PlotWidget()
        metrics_layout.addWidget(self.plot_widget, 4)
        
        self.plot_curves = {}
        self.metric_data = {}

        self.plot_widget.setBackground('#3c3f41')
        self.plot_widget.getAxis('left').setTextPen('w')
        self.plot_widget.getAxis('bottom').setTextPen('w')
        self.plot_widget.setLabel('left', 'Value', color='w')
        self.plot_widget.setLabel('bottom', 'Step', color='w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(textColor='w')
        
        return metrics_panel_widget

    def _update_metrics_panel(self, num_teams):
        """Dynamically updates the checkboxes in the metrics panel based on the number of teams."""
        # Clear existing checkboxes
        for i in reversed(range(self.metrics_checkbox_layout.count())): 
            widgetToRemove = self.metrics_checkbox_layout.itemAt(i).widget()
            if widgetToRemove is not None:
                widgetToRemove.setParent(None)
        self.metric_checkboxes.clear()

        # Define metrics to be displayed per team and globally
        team_metrics = ["Reward", "Agents Alive", "Hive Health", "Kills", "Deaths", "Damage Dealt", "Grapples Initiated", "Grapples Broken", "Hive Damage"]
        global_metrics = ["Resources Delivered", "Resources Picked"]

        # Create new checkboxes for each team
        for i in range(num_teams):
            for metric in team_metrics:
                metric_name = f"Team {i} {metric}"
                cb = QCheckBox(metric_name)
                cb.setChecked(True)
                self.metric_checkboxes[metric_name] = cb
                self.metrics_checkbox_layout.addWidget(cb)
        
        # Create global metric checkboxes
        for metric in global_metrics:
            cb = QCheckBox(metric)
            cb.setChecked(True)
            self.metric_checkboxes[metric] = cb
            self.metrics_checkbox_layout.addWidget(cb)

    def _create_main_buttons(self):
        """Creates the main simulation (Start/Stop) and preset (Save/Load/Defaults) control buttons."""
        button_layout = QHBoxLayout()
        sim_button_layout = QHBoxLayout()
        preset_button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        sim_button_layout.addWidget(self.start_button)
        sim_button_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("Save Preset")
        self.load_button = QPushButton("Load Preset")
        self.defaults_button = QPushButton("Load Defaults")
        preset_button_layout.addWidget(self.save_button)
        preset_button_layout.addWidget(self.load_button)
        preset_button_layout.addWidget(self.defaults_button)

        button_layout.addLayout(sim_button_layout)
        button_layout.addStretch()
        button_layout.addLayout(preset_button_layout)
        
        return button_layout

    def _recursively_setup_widgets(self, layout, config_dict, widget_dict, is_team_specific=False):
        """
        Recursive function to create nested widgets for any config dictionary.
        
        This function walks through the configuration dictionary and creates
        collapsible boxes for categories and parameter widgets for individual settings.
        If `is_team_specific` is True, it only creates widgets for parameters
        marked as 'overridable'.
        """
        for key, params in config_dict.items():
            # Team Settings are handled by the dedicated team tab system.
            if key == "Team Settings":
                continue

            if not isinstance(params, dict):
                continue
            
            # --- Handle Parameters (leaf nodes of the config tree) ---
            if "value" in params:
                if is_team_specific and not params.get("overridable", False):
                    continue # Skip non-overridable parameters in team-specific views.
                
                # --- NEW: Check if this is a reward and get its base value ---
                base_value = None
                label_text = key # Default label
                if params.get("type") == "reward":
                    base_value = constants_module.REWARD_CONFIG.get(key, {}).get("default_value")
                    # For team-specific rewards, create a descriptive label with the base value.
                    if base_value is not None and is_team_specific:
                        label_text = f"{key} (Base: {base_value})"

                # Use the original key for creating the widget
                key_for_widget = key
                container, widget_group = create_parameter_widget(label_text, params, base_value=base_value, is_team_specific=is_team_specific)
                if isinstance(layout, CollapsibleBox):
                    layout.add_widget(container)
                else:
                    layout.addWidget(container)
                widget_dict[key_for_widget] = widget_group
            
            # --- Handle Categories/Sub-categories (branch nodes) ---
            else:
                # In team view, only show categories that contain at least one overridable item.
                if is_team_specific:
                    def can_display_in_team_view(d):
                        """Checks if a category or any of its sub-items are overridable."""
                        if d.get("overridable", False):
                            return True
                        # Check children recursively.
                        for v in d.values():
                            if isinstance(v, dict):
                                if "value" in v: # It's a parameter
                                    if v.get("overridable", False):
                                        return True
                                elif can_display_in_team_view(v): # It's a sub-category
                                    return True
                        return False

                    if not can_display_in_team_view(params):
                        continue
                
                help_text = params.get("help_text", "")
                cat_box = CollapsibleBox(key, help_text=help_text)
                if isinstance(layout, CollapsibleBox):
                    layout.add_widget(cat_box)
                else:
                    layout.addWidget(cat_box)
                
                widget_dict[key] = {}
                self._recursively_setup_widgets(cat_box, params, widget_dict[key], is_team_specific)

    def on_scenario_selected(self, index):
        """Handles the selection of a scenario from the dropdown."""
        if index <= 0: # Ignore the placeholder text "Select a scenario..."
            return

        scenario_name = self.scenario_selector.itemText(index)
        scenario_flat_config = self.scenarios[scenario_name]

        # Start with a fresh default config and apply the scenario's flat overrides.
        new_base_config = get_default_config()

        def update_nested(d, flat_d):
            """Recursively updates the nested config dictionary with values from the flat scenario dictionary."""
            for k, v in d.items():
                if isinstance(v, dict):
                    if "value" in v and k in flat_d:
                        v["value"] = flat_d[k]
                    else:
                        update_nested(v, flat_d)
        
        update_nested(new_base_config, scenario_flat_config)
        
        self.base_config = new_base_config
        self.team_overrides = {}

        # Update all GUI widgets to reflect the new configuration.
        flat_config = self.get_flat_config(self.base_config)
        self.set_widgets_from_config(flat_config)

        self.update_team_setup_panel()
        self.update_status(f"Loaded scenario: {scenario_name}")
        
        self.scenario_selector.setCurrentIndex(0) # Reset selector to placeholder.

    def update_team_setup_panel(self):
        """
        Updates the entire 'Team Setup' tab based on the current number of teams.
        
        Clears and recreates the list of teams and their corresponding settings pages.
        """
        self.team_list_widget.clear()
        # Clear the stacked widget that holds the settings pages for each team.
        while self.team_stacked_widget.count() > 0:
            widget = self.team_stacked_widget.widget(0)
            self.team_stacked_widget.removeWidget(widget)
            if widget:
                widget.deleteLater()
        self.team_parameter_widgets.clear()
        
        num_teams_widget, _ = self.find_widget_and_details("num_teams")
        if not num_teams_widget:
            print("Error: Could not find 'num_teams' widget to determine number of teams.")
            return
        
        num_teams = num_teams_widget['slider'].value()
        
        self._update_metrics_panel(num_teams)

        # Ensure the team config list has enough entries for the number of teams.
        while len(self.base_config["Team Settings"]["teams"]) < num_teams:
            self.base_config["Team Settings"]["teams"].append(
                {"name": f"Team {len(self.base_config['Team Settings']['teams'])}", "color": [128, 128, 128, 255]}
            )

        # Disconnect signal to prevent multiple connections during recreation.
        try:
            self.team_list_widget.currentRowChanged.disconnect()
        except TypeError:
            pass # Signal was not connected.
        
        for i in range(num_teams):
            team_data = self.base_config["Team Settings"]["teams"][i]
            self.team_list_widget.addItem(f"Team {i}: {team_data['name']}")
            team_settings_widget = self._create_team_settings_widget(i)
            self.team_stacked_widget.addWidget(team_settings_widget)
            
        self.team_list_widget.currentRowChanged.connect(self.team_stacked_widget.setCurrentIndex)
        if num_teams > 0:
            self.team_list_widget.setCurrentRow(0)

    def _create_team_settings_widget(self, team_index):
        """Creates the settings panel for a single team."""
        container = QWidget()
        main_layout = QVBoxLayout(container)

        details_layout = QHBoxLayout()
        team_data = self.base_config["Team Settings"]["teams"][team_index]
        
        details_layout.addWidget(QLabel(f"<b>Team {team_index}:</b>"))
        name_edit = QLineEdit(team_data["name"])
        details_layout.addWidget(name_edit)
        
        color_button = QPushButton()
        color_button.setFixedSize(25, 25)
        self.update_button_color(color_button, team_data["color"])
        details_layout.addWidget(color_button)

        # --- Policy Selection for this team ---
        details_layout.addStretch()
        details_layout.addWidget(QLabel("Policy:"))
        policy_selector = QComboBox()
        policy_selector.addItems(self.available_policies)
        
        # --- NEW: Config Button for Policy (e.g. loading models) ---
        policy_config_btn = QPushButton("Load Model")
        policy_config_btn.setFixedSize(80, 25)
        policy_config_btn.setEnabled(False) # Disabled by default
        policy_config_btn.clicked.connect(lambda: self.on_configure_team_policy(team_index))
        # -----------------------------------------------------------

        # Store the policy selector in the config to easily access its value later.
        team_data['policy_widget'] = policy_selector
        team_data['policy_config_btn'] = policy_config_btn # Store button ref
        
        details_layout.addWidget(policy_selector)
        details_layout.addWidget(policy_config_btn) # Add button to layout

        main_layout.addLayout(details_layout)

        reset_button = QPushButton("Reset All Overrides to Global Defaults")
        reset_button.setToolTip(
            "Resets all parameters for this team back to the global default values.\n"
            "Parameters with a yellow label are using a team-specific override."
        )
        main_layout.addWidget(reset_button)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        param_container = QWidget()
        param_layout = QVBoxLayout(param_container)
        scroll.setWidget(param_container)

        team_widgets = {}
        self._recursively_setup_widgets(param_layout, get_default_config(), team_widgets, is_team_specific=True)
        self.team_parameter_widgets[team_index] = team_widgets
        
        self._update_team_widgets_from_config(team_index)
        
        name_edit.editingFinished.connect(lambda: self.on_team_name_changed(team_index, name_edit.text()))
        color_button.clicked.connect(lambda: self.on_change_team_color(color_button, team_index))
        reset_button.clicked.connect(lambda: self.on_reset_team_overrides(team_index))
        policy_selector.currentIndexChanged.connect(
            lambda: self.on_team_policy_changed(team_index, policy_selector.currentText())
        )

        def connect_team_widgets(widget_dict):
            for key, widget in widget_dict.items():
                if isinstance(widget, dict) and "label" not in widget:
                    connect_team_widgets(widget)
                else:
                    self._connect_team_widget_signal(widget, key, team_index)
        connect_team_widgets(team_widgets)

        return container

    def on_team_policy_changed(self, team_index, policy_name):
        """Handles changes to a team's policy selection dropdown."""
        self.base_config["Team Settings"]["teams"][team_index]["policy"] = policy_name
        self.update_status(f"Team {team_index} policy set to {policy_name}")
        
        # Enable "Load Model" button if it's a trainable policy
        team_data = self.base_config["Team Settings"]["teams"][team_index]
        btn = team_data.get('policy_config_btn')
        if btn:
            is_trainable = "trained" in policy_name.lower()
            btn.setEnabled(is_trainable)
            if is_trainable:
                # Optionally auto-open dialog if switching to it for the first time
                # self.on_configure_team_policy(team_index) 
                pass

    def on_configure_team_policy(self, team_index):
        """Opens a file dialog to select a model checkpoint for the team's policy."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select Model for Team {team_index}", 
            os.getcwd(), 
            "PyTorch Models (*.pt *.pth)"
        )
        
        if file_path:
            self.base_config["Team Settings"]["teams"][team_index]["policy_path"] = file_path
            self.update_status(f"Selected model for Team {team_index}: {os.path.basename(file_path)}")
        else:
            self.update_status("Model selection cancelled.")

    def on_team_name_changed(self, team_index, new_name):
        """Handles changes to a team's name."""
        self.base_config["Team Settings"]["teams"][team_index]["name"] = new_name
        self.team_list_widget.item(team_index).setText(f"Team {team_index}: {new_name}")
        self.update_status(f"Team {team_index} name changed to {new_name}")
        
    def on_change_team_color(self, button, team_index):
        """Opens a color dialog to change a team's color."""
        current_color_val = self.base_config["Team Settings"]["teams"][team_index]["color"]
        current_color = QColor(*current_color_val)
        
        new_color = QColorDialog.getColor(current_color, self)
        
        if new_color.isValid():
            new_color_val = [new_color.red(), new_color.green(), new_color.blue(), new_color.alpha()]
            self.base_config["Team Settings"]["teams"][team_index]["color"] = new_color_val
            self.update_button_color(button, new_color_val)
            self.update_status(f"Team {team_index} color changed.")

    def update_button_color(self, button, color_val):
        """Updates a button's background color and text color for contrast."""
        r, g, b, a = color_val
        button.setStyleSheet(f"background-color: rgba({r},{g},{b},{a});")
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "black" if brightness > 125 else "white"
        button.setStyleSheet(button.styleSheet() + f"color: {text_color};")

    def on_reset_team_overrides(self, team_index):
        """Resets all parameter overrides for a specific team."""
        if team_index in self.team_overrides:
            del self.team_overrides[team_index]
        self._update_team_widgets_from_config(team_index)
        self.update_status(f"All overrides for Team {team_index} have been reset.")
        
    def _connect_team_widget_signal(self, widget, key, team_index):
        """Connects signals for a parameter widget within a team's settings panel."""
        details = self.find_param_details(self.base_config, key)
        if not details: return

        def value_changed_handler():
            self.on_team_parameter_changed(key, team_index)

        widget_type = details.get('type')
        if widget_type == 'bool':
            if 'checkbox' in widget: widget['checkbox'].toggled.connect(value_changed_handler)
        elif 'slider' in widget:
            widget['slider'].valueChanged.connect(value_changed_handler)
            widget['line_edit'].editingFinished.connect(value_changed_handler)
        elif 'base_slider' in widget:
            widget['base_slider'].valueChanged.connect(value_changed_handler)
            widget['base_line_edit'].editingFinished.connect(value_changed_handler)
            widget['rand_slider'].valueChanged.connect(value_changed_handler)
            widget['rand_line_edit'].editingFinished.connect(value_changed_handler)

    def on_team_parameter_changed(self, key, team_index):
        """
        Handles a change in a team-specific parameter override.
        
        Stores the new value in `self.team_overrides` and updates the widget's style.
        """
        widget_dict = self.team_parameter_widgets.get(team_index)
        if not widget_dict: return
        
        widget = self._find_widget_in_dict(widget_dict, key)
        details = self.find_param_details(self.base_config, key)
        if not widget or not details: return
        
        new_value = self.get_value_from_widget(widget, details)
        
        if team_index not in self.team_overrides:
            self.team_overrides[team_index] = {}
        self.team_overrides[team_index][key] = new_value
        
        # In the GUI, team overrides are stored with the team index as an integer.
        # The environment expects a string key for the team ID.
        # Let's prepare the data structure the environment expects.
        self.env_team_overrides = {str(k): v for k, v in self.team_overrides.items()}

        # NEW: Pass only the 'Rewards' part of the overrides to the dedicated reward parameter.
        # The main override dict will handle other parameters.
        reward_multipliers = {}
        for team_id_int, overrides in self.team_overrides.items():
            team_id_str = str(team_id_int)
            if 'Rewards' in overrides:
                reward_multipliers[team_id_str] = {k: v for k, v in overrides['Rewards'].items()}

        self.env_reward_multipliers = reward_multipliers


        self._update_widget_style(widget, is_overridden=True)
        
    def _find_widget_in_dict(self, widget_dict, key):
        """Recursively finds a widget group by key in a nested widget dictionary."""
        if key in widget_dict:
            return widget_dict[key]
        for v in widget_dict.values():
            if isinstance(v, dict):
                found = self._find_widget_in_dict(v, key)
                if found: return found
        return None

    def _update_team_widgets_from_config(self, team_index):
        """Updates all widgets in a team's settings panel from the current config."""
        if team_index not in self.team_parameter_widgets: return
        
        flat_base_config = self.get_flat_config(self.base_config)
        team_overrides = self.team_overrides.get(team_index, {})
        merged_config = {**flat_base_config, **team_overrides}
        
        self.set_widgets_from_config(merged_config, target_widgets=self.team_parameter_widgets[team_index])

        def update_styles_recursively(w_dict):
            for key, widget in w_dict.items():
                if isinstance(widget, dict) and "label" not in widget:
                    update_styles_recursively(widget)
                else:
                    is_overridden = key in team_overrides
                    self._update_widget_style(widget, is_overridden)
        update_styles_recursively(self.team_parameter_widgets[team_index])

    def _update_widget_style(self, widget, is_overridden):
        """Updates a widget's label color to indicate if it's using an overridden value."""
        if not isinstance(widget, dict) or "label" not in widget:
            return

        label = widget["label"]
        style = "color: #ffcc00;" if is_overridden else "" # Bright yellow for override
        label.setStyleSheet(style)
    
    def connect_widget_signal(self, widget, key):
        """Connects signals for a global parameter widget."""
        details = self.find_param_details(self.base_config, key)
        if not details: return

        def value_changed_handler(): self.on_parameter_changed(key)

        widget_type = details.get('type')
        if widget_type == 'bool':
            if 'checkbox' in widget: widget['checkbox'].toggled.connect(value_changed_handler)
        elif 'slider' in widget:
            widget['slider'].valueChanged.connect(value_changed_handler)
            widget['line_edit'].editingFinished.connect(value_changed_handler)
        elif 'base_slider' in widget:
            widget['base_slider'].valueChanged.connect(value_changed_handler)
            widget['base_line_edit'].editingFinished.connect(value_changed_handler)
            widget['rand_slider'].valueChanged.connect(value_changed_handler)
            widget['rand_line_edit'].editingFinished.connect(value_changed_handler)
     
    def on_parameter_changed(self, key):
        """Handles a change in a global parameter."""
        widget, config_details = self.find_widget_and_details(key)
        if not widget or not config_details: return

        new_value = self.get_value_from_widget(widget, config_details)
        self.update_nested_dict(self.base_config, key, new_value)
        
        if key == "num_teams": 
            self.update_team_setup_panel()
            
    def get_value_from_widget(self, widget, details):
        """Extracts the correctly typed value from a parameter widget group."""
        try:
            param_type = details.get('type')
            if param_type == 'bool': 
                return widget['checkbox'].isChecked()
            elif param_type == 'int':
                return widget['slider'].value()
            elif param_type == 'float':
                return widget['slider'].value() / 100.0
            elif param_type == 'reward':
                val = widget['slider'].value() / 100.0
                return val if widget['checkbox'].isChecked() else 0.0
            elif param_type == 'randomizable':
                base_val = widget['base_slider'].value()
                if details.get('value_type') == 'float':
                    base_val /= 100.0
                rand_val = widget['rand_slider'].value() / 100.0
                return {"base": base_val, "rand": rand_val}
        except (ValueError, KeyError, AttributeError): return None
        
    def update_nested_dict(self, d, key, value):
        """Recursively searches for a key in the nested config and updates its 'value' field."""
        for k, v in d.items():
            if k == key and 'value' in v:
                v['value'] = value
                return True
            elif isinstance(v, dict) and self.update_nested_dict(v, key, value):
                return True
        return False
    
    def get_flat_config(self, config_dict):
        """
        Flattens a nested configuration dictionary into a simple key-value map.
        
        Example: {"Category": {"param": {"value": 1}}} -> {"param": 1}
        """
        flat_config = {}
        def flatten(d):
            for k, v in d.items():
                if isinstance(v, dict) and "value" in v:
                    flat_config[k] = v['value']
                elif isinstance(v, dict): flatten(v)
        flatten(config_dict)
        return flat_config

    def load_defaults(self):
        """Resets the entire GUI and configuration to the default state."""
        self.base_config = get_default_config()
        self.team_overrides = {}
        
        flat_default_config = self.get_flat_config(self.base_config)
        self.set_widgets_from_config(flat_default_config)
        
        self.update_team_setup_panel()
        self.update_status("All parameters reset to default values.")

    def save_preset(self):
        """Opens a file dialog to save the current configuration as a .json preset file."""
        config_to_save = {"base_config": self.base_config, "team_overrides": self.team_overrides}
        presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
        os.makedirs(presets_dir, exist_ok=True)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Preset", presets_dir, "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f: json.dump(config_to_save, f, indent=4)
            self.update_status(f"Preset saved to {file_path}")

    def load_preset(self):
        """Opens a file dialog to load a .json preset file."""
        presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Preset", presets_dir, "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r') as f: loaded_data = json.load(f)
            
            self.base_config = loaded_data.get("base_config", get_default_config())
            self.team_overrides = loaded_data.get("team_overrides", {})
            
            flat_base_config = self.get_flat_config(self.base_config)
            self.set_widgets_from_config(flat_base_config)

            self.update_team_setup_panel()
            
            self.update_status(f"Preset loaded from {file_path}")

    def set_widgets_from_config(self, loaded_data, target_widgets=None):
        """
        Updates the values of all GUI widgets from a flat configuration dictionary.

        Args:
            loaded_data (dict): A flat dictionary of parameter names to values.
            target_widgets (dict, optional): The widget dictionary to update. 
                                             Defaults to the global widgets.
        """
        if target_widgets is None:
            target_widgets = self.widgets
            
        def recurse_and_update(w_dict):
            for key, widget in w_dict.items():
                if isinstance(widget, dict) and "label" not in widget:
                    recurse_and_update(widget)
                    continue

                if key not in loaded_data: continue
                param_details = self.find_param_details(self.base_config, key)
                if not param_details: continue
                
                loaded_value = loaded_data[key]
                param_type = param_details.get("type")

                try:
                    if param_type == 'randomizable' and isinstance(loaded_value, dict):
                        base_val, rand_val = loaded_value.get('base', 0), loaded_value.get('rand', 0)
                        multiplier = 100 if param_details['value_type'] == 'float' else 1
                        widget["base_slider"].setValue(int(base_val * multiplier))
                        widget["rand_slider"].setValue(int(rand_val * 100))
                    elif param_type == 'reward':
                        is_enabled = float(loaded_value) != 0.0
                        widget["checkbox"].setChecked(is_enabled)
                        widget["slider"].setValue(int(float(loaded_value) * 100))
                    elif param_type == 'float':
                        widget["slider"].setValue(int(float(loaded_value) * 100))
                    elif param_type == 'int':
                        widget["slider"].setValue(int(loaded_value))
                    elif param_type == 'bool':
                        widget['checkbox'].setChecked(bool(loaded_value))
                except (TypeError, ValueError, AttributeError) as e:
                    print(f"Warning: Could not update widget for '{key}'. Invalid data: {loaded_value}. Error: {e}")
        recurse_and_update(target_widgets)

    def find_param_details(self, config, key):
        """Recursively finds the full parameter details dictionary for a given key."""
        if key in config and "value" in config[key]: return config[key]
        for v in config.values():
            if isinstance(v, dict):
                found = self.find_param_details(v, key)
                if found: return found
        return None

    def find_widget_and_details(self, key):
        """Convenience function to find both a widget and its config details by key."""
        return self._find_widget_in_dict(self.widgets, key), self.find_param_details(self.base_config, key)
        
    def setup_simulation_environment(self):
        """
        Prepares and creates the Swarm2DEnv instance in the main thread.
        
        This method gathers all parameters from the GUI, applies them to the
        constants module, reloads necessary modules to ensure changes take effect,
        and finally instantiates the environment.
        """
        sig = inspect.signature(Swarm2DEnv.__init__)
        valid_env_params = {p for p in sig.parameters if p != 'self'}
        
        env_params = {}
        constants_params = {}
        agent_rand_factors = {}
        
        # Use the full nested config to properly distinguish parameter types
        def process_config_recursively(config_dict, parent_key=None):
            for key, params in config_dict.items():
                if isinstance(params, dict):
                    if "value" in params: # This is a parameter
                        value = params['value']
                        param_type = params.get('type')
                        
                        # Randomizable parameters are handled separately and passed in a dedicated dictionary.
                        if param_type == 'randomizable' and isinstance(value, dict):
                            agent_rand_factors[key] = value
                        else:
                            # Normal parameters are either for the environment's __init__ or for the constants module.
                            if key in valid_env_params:
                                env_params[key] = value
                            else:
                                constants_params[key] = value
                    else:
                        # This is a category, so recurse into it.
                        process_config_recursively(params, key)

        process_config_recursively(self.base_config)

        # The 'agent_randomization_factors' dict is the primary way to pass randomization settings.
        env_params['agent_randomization_factors'] = agent_rand_factors
        
        # --- Handle Per-Team Agent Counts ---
        num_teams = env_params.get('num_teams', self.base_config.get("General", {}).get("num_teams", {}).get("value", 1))
        default_agent_count = self.base_config.get("General", {}).get("num_agents_per_team", {}).get("value", 10)
        
        agent_counts_per_team = []
        for i in range(num_teams):
            team_override_params = self.team_overrides.get(i, {})
            agent_count = team_override_params.get('num_agents_per_team', default_agent_count)
            agent_counts_per_team.append(agent_count)
        
        env_params['num_agents_per_team'] = agent_counts_per_team
        # --- End Handle Per-Team Agent Counts ---

        # Convert integer team keys to strings for the environment
        env_params['team_parameter_overrides'] = {str(k): v for k, v in self.team_overrides.items()}
        
        # NEW: The environment now accepts a dedicated parameter for reward multipliers.
        # This keeps reward tuning separate from other parameter overrides.
        reward_multipliers = {}
        for team_id_int, overrides in self.team_overrides.items():
            team_id_str = str(team_id_int)
            # We need to find the 'Rewards' category within the nested override dictionary.
            # The structure is {'Rewards': {'r_delivery': 1.5, ...}}
            reward_overrides_for_team = {}

            def find_rewards_recursively(d):
                if 'Rewards' in d and isinstance(d['Rewards'], dict):
                    reward_overrides_for_team.update(d['Rewards'])
                for v in d.values():
                    if isinstance(v, dict):
                        find_rewards_recursively(v)

            find_rewards_recursively(overrides)
            if reward_overrides_for_team:
                reward_multipliers[team_id_str] = reward_overrides_for_team

        env_params['team_reward_multipliers'] = reward_multipliers


        # Set values in the constants module. This is necessary for global parameters
        # that are not passed directly to the environment's constructor.
        for key, value in constants_params.items():
            key_upper = key.upper()
            if hasattr(constants_module, key_upper):
                setattr(constants_module, key_upper, value)
                if hasattr(constants_module, 'ConstantsMain') and hasattr(constants_module.ConstantsMain, key_upper):
                    setattr(constants_module.ConstantsMain, key_upper, value)

        # Reloading modules is crucial to ensure that changes to the constants module are propagated
        # to all other modules that import it.
        import Swarm2d.env.env as env_module
        import Swarm2d.env.spawn as spawn_module
        import Swarm2d.env.physics as physics_module
        import Swarm2d.env.observations as observations_module
        import Swarm2d.env.rewards as rewards_module
        import Swarm2d.env.managers as managers_module
        importlib.reload(constants_module) # Reload constants first
        importlib.reload(spawn_module)
        importlib.reload(physics_module)
        importlib.reload(observations_module)
        importlib.reload(rewards_module)
        importlib.reload(managers_module)
        importlib.reload(env_module)
        
        render_mode_val = "gui" if self.base_config["Rendering & Debug"]["render_pybullet_gui"]["value"] else "headless"
        
        team_configs = self.base_config.get("Team Settings", {}).get("teams", [])

        # Pass the new 'team_reward_multipliers' parameter to the environment
        env = env_module.Swarm2DEnv(render_mode=render_mode_val, team_configs=team_configs, **env_params)
        return env
        
    def start_simulation(self):
        """Starts the simulation thread."""
        if self.sim_thread and self.sim_thread.isRunning():
            self.update_status("Simulation is already running.")
            return

        self.stop_simulation()

        try:
            self.update_status("Creating environment...")
            QApplication.processEvents()
            self.env_instance = self.setup_simulation_environment()
        except Exception as e:
            self.update_status(f"Error creating environment: {e}")
            traceback.print_exc()
            return
        
        # Use a flattened config for the runner's simple needs (like FPS)
        flat_config = self.get_flat_config(self.base_config)
        # Add pygame rendering flag
        flat_config['render_pygame_window'] = self.base_config["Rendering & Debug"]["render_pygame_window"]["value"]

        # --- Attach Policy Paths to Env ---
        # The SimulationRunner creates policies in a thread, so we must pass paths via the env
        # or another shared object. Attaching to env is the simplest way given current architecture.
        if not hasattr(self.env_instance, 'team_policy_paths'):
            self.env_instance.team_policy_paths = {}
            
        for i in range(len(self.base_config["Team Settings"]["teams"])):
            team_data = self.base_config["Team Settings"]["teams"][i]
            if "policy_path" in team_data:
                self.env_instance.team_policy_paths[i] = team_data["policy_path"]
        # ----------------------------------

        policy_name = self.policy_selector.currentText()

        # --- Get Per-Team Policies ---
        team_policies = {}
        num_teams = self.base_config["General"]["num_teams"]["value"]
        for i in range(num_teams):
            team_data = self.base_config["Team Settings"]["teams"][i]
            # Use the globally selected policy if a team-specific one isn't set
            team_policies[str(i)] = team_data.get("policy", policy_name)

        self.sim_thread = SimulationRunner(self.env_instance, flat_config, team_policies)
        self.sim_thread.status_changed.connect(self.update_status)
        self.sim_thread.step_info.connect(self.update_step_info)
        self.sim_thread.metrics_updated.connect(self.update_metrics)
        self.sim_thread.observation_updated.connect(self.on_observation_updated)
        self.sim_thread.start()
        self.update_status("Starting simulation...")

    def stop_simulation(self):
        """Stops the simulation thread and cleans up the environment."""
        if self.sim_thread and self.sim_thread.isRunning():
            self.sim_thread.stop()
            self.sim_thread.wait()
            self.update_status("Stopping simulation...")
        
        if self.env_instance:
            self.env_instance.close()
            self.env_instance = None

        self.clear_metrics()

    def update_status(self, message):
        """Updates the message in the status bar."""
        self.status_bar.showMessage(message)

    def update_step_info(self, step):
        """Updates the step count in the status bar."""
        self.status_bar.showMessage(f"Running... Step: {step}")

    def update_metrics(self, metrics):
        """
        Receives new metrics from the simulation thread and updates the plot.
        
        Handles both cumulative (e.g., reward) and instantaneous (e.g., agent count) metrics.
        """
        for name, value in metrics.items():
            display_name = name.replace("_", " ").title()
            
            num_teams = self.env_instance.num_teams if self.env_instance else 0

            if display_name not in self.metric_data:
                self.metric_data[display_name] = {'x': [], 'y': []}
                if display_name in self.metric_checkboxes:
                    # Use a color map to get distinct colors for each new metric
                    color = pg.intColor(len(self.plot_curves), hues=num_teams * 3 if num_teams > 0 else 9)
                    self.plot_curves[display_name] = self.plot_widget.plot(pen=color, name=display_name)
            
            # Decide if the metric is cumulative or instantaneous
            if "Reward" in display_name or "Delivered" in display_name or "Kills" in display_name or "Deaths" in display_name or "Damage" in display_name or "Grapples" in display_name:
                # Cumulative metrics
                y_val = self.metric_data[display_name]['y'][-1] + value if self.metric_data[display_name]['y'] else value
            else:
                # Instantaneous metrics (like agent count, health)
                y_val = value
            
            self.metric_data[display_name]['x'].append(len(self.metric_data[display_name]['x']))
            self.metric_data[display_name]['y'].append(y_val)

            if display_name in self.plot_curves and self.metric_checkboxes.get(display_name, QCheckBox()).isChecked():
                self.plot_curves[display_name].setData(self.metric_data[display_name]['x'], self.metric_data[display_name]['y'])

    def clear_metrics(self):
        """Clears all data from the metrics plot."""
        self.plot_widget.clear()
        self.plot_curves = {}
        self.metric_data = {}
        self.plot_widget.addLegend()
    
    def _create_visualization_tab(self):
        """Creates the 'Observation Visualization' tab with controls for exporting observations."""
        tab = QWidget()
        self.tabs.addTab(tab, "Visualization")
        main_layout = QVBoxLayout(tab)
        
        # Instructions
        info_label = QLabel(
            "Export observation visualizations as images or GIFs.\n"
            "Select an agent, observation types, and output format, then click 'Export'."
        )
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        
        # Agent selection
        agent_group = QGroupBox("Agent Selection")
        agent_layout = QHBoxLayout()
        agent_layout.addWidget(QLabel("Agent ID:"))
        self.agent_selector = QSpinBox()
        self.agent_selector.setMinimum(0)
        self.agent_selector.setMaximum(999)
        self.agent_selector.setValue(0)
        agent_layout.addWidget(self.agent_selector)
        agent_layout.addStretch()
        agent_group.setLayout(agent_layout)
        main_layout.addWidget(agent_group)
        
        # Observation type selection
        obs_group = QGroupBox("Observation Types")
        obs_layout = QVBoxLayout()
        self.vis_raw_map_checkbox = QCheckBox("Raw Map")
        self.vis_raw_map_checkbox.setChecked(True)
        obs_layout.addWidget(self.vis_raw_map_checkbox)
        
        self.vis_memory_map_checkbox = QCheckBox("Memory Map")
        self.vis_memory_map_checkbox.setChecked(True)
        obs_layout.addWidget(self.vis_memory_map_checkbox)
        
        self.vis_graph_checkbox = QCheckBox("Graph")
        self.vis_graph_checkbox.setChecked(True)
        obs_layout.addWidget(self.vis_graph_checkbox)
        obs_group.setLayout(obs_layout)
        main_layout.addWidget(obs_group)
        
        # --- NEW: Visualization Parameters ---
        vis_params_group = QGroupBox("Visualization Parameters")
        vis_params_layout = QVBoxLayout()

        # Coverage Intensity Exponent
        coverage_layout = QHBoxLayout()
        coverage_layout.addWidget(QLabel("Coverage Intensity Exponent:"))
        self.vis_coverage_exponent = QDoubleSpinBox()
        self.vis_coverage_exponent.setRange(0.1, 5.0)
        self.vis_coverage_exponent.setSingleStep(0.1)
        self.vis_coverage_exponent.setValue(0.8)
        self.vis_coverage_exponent.setToolTip("Controls entity brightness in memory maps.\nLower values = brighter (e.g., 0.5), Higher values = fainter (e.g., 1.5).")
        coverage_layout.addWidget(self.vis_coverage_exponent)
        vis_params_layout.addLayout(coverage_layout)

        # Visualization Gamma
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Visualization Gamma:"))
        self.vis_gamma = QDoubleSpinBox()
        self.vis_gamma.setRange(0.1, 5.0)
        self.vis_gamma.setSingleStep(0.1)
        self.vis_gamma.setValue(0.45)
        self.vis_gamma.setToolTip("Gamma correction for raw maps to make faint details more visible.\n< 1.0 boosts shadows, > 1.0 darkens shadows.")
        gamma_layout.addWidget(self.vis_gamma)
        vis_params_layout.addLayout(gamma_layout)

        vis_params_group.setLayout(vis_params_layout)
        main_layout.addWidget(vis_params_group)
        # --- END NEW ---

        # Output format selection
        output_group = QGroupBox("Output Format")
        output_layout = QVBoxLayout()
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["Single Image", "GIF (Animated)"])
        output_layout.addWidget(QLabel("Format:"))
        output_layout.addWidget(self.output_format_combo)
        
        # GIF options (only visible when GIF is selected)
        self.gif_start_step_label = QLabel("Start Step:")
        self.gif_start_step = QSpinBox()
        self.gif_start_step.setMinimum(0)
        self.gif_start_step.setMaximum(999999)
        self.gif_start_step.setValue(0)
        
        self.gif_end_step_label = QLabel("End Step:")
        self.gif_end_step = QSpinBox()
        self.gif_end_step.setMinimum(0)
        self.gif_end_step.setMaximum(999999)
        self.gif_end_step.setValue(100)
        
        self.gif_fps_label = QLabel("FPS:")
        self.gif_fps = QSpinBox()
        self.gif_fps.setMinimum(1)
        self.gif_fps.setMaximum(60)
        self.gif_fps.setValue(10)
        
        gif_layout = QHBoxLayout()
        gif_layout.addWidget(self.gif_start_step_label)
        gif_layout.addWidget(self.gif_start_step)
        gif_layout.addWidget(self.gif_end_step_label)
        gif_layout.addWidget(self.gif_end_step)
        gif_layout.addWidget(self.gif_fps_label)
        gif_layout.addWidget(self.gif_fps)
        output_layout.addLayout(gif_layout)
        
        # Show/hide GIF options based on selection
        def toggle_gif_options(index):
            is_gif = index == 1
            self.gif_start_step_label.setVisible(is_gif)
            self.gif_start_step.setVisible(is_gif)
            self.gif_end_step_label.setVisible(is_gif)
            self.gif_end_step.setVisible(is_gif)
            self.gif_fps_label.setVisible(is_gif)
            self.gif_fps.setVisible(is_gif)
        
        self.output_format_combo.currentIndexChanged.connect(toggle_gif_options)
        toggle_gif_options(0)  # Initialize to hide GIF options
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Export directory selection
        dir_group = QGroupBox("Output Directory")
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directory:"))
        self.output_dir_edit = QLineEdit(self.visualization_output_dir)
        dir_layout.addWidget(self.output_dir_edit)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_output_dir)
        dir_layout.addWidget(browse_button)
        dir_group.setLayout(dir_layout)
        main_layout.addWidget(dir_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        self.export_image_button = QPushButton("Export Current Observation")
        self.export_image_button.clicked.connect(self._export_current_observation)
        button_layout.addWidget(self.export_image_button)
        
        self.start_gif_button = QPushButton("Start GIF Recording")
        self.start_gif_button.clicked.connect(self._start_gif_recording)
        button_layout.addWidget(self.start_gif_button)
        
        self.stop_gif_button = QPushButton("Stop & Save GIF")
        self.stop_gif_button.clicked.connect(self._stop_and_save_gif)
        self.stop_gif_button.setEnabled(False)
        button_layout.addWidget(self.stop_gif_button)
        
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        
        # Store references for GIF recording state
        self.is_recording_gif = False
        self.recording_gif_frames = []
    
    def _browse_output_dir(self):
        """Opens a file dialog to select the output directory."""
        current_dir = self.output_dir_edit.text()
        if not os.path.exists(current_dir):
            current_dir = os.path.dirname(__file__)
        
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", current_dir)
        if selected_dir:
            self.output_dir_edit.setText(selected_dir)
            self.visualization_output_dir = selected_dir
    
    def on_observation_updated(self, obs_list, step):
        """Called when new observations are available from the simulation."""
        # obs_list is a list of observation dicts, one per agent
        self.current_observations = obs_list
        self.current_step = step
        
        # If recording GIF, add frames
        if self.is_recording_gif:
            self._add_gif_frame(obs_list, step)
    
    def _export_current_observation(self):
        """Exports the current observation as image(s)."""
        if not self.sim_thread or not self.sim_thread.isRunning():
            self.update_status("Simulation is not running. Start the simulation first.")
            return
        
        if not self.current_observations or len(self.current_observations) == 0:
            self.update_status("No observations available yet. Wait for the simulation to run a few steps.")
            return
        
        agent_id = self.agent_selector.value()
        output_dir = self.output_dir_edit.text()
        os.makedirs(output_dir, exist_ok=True)
        
        if agent_id >= len(self.current_observations):
            self.update_status(f"Agent {agent_id} not found. Available agents: 0-{len(self.current_observations)-1}")
            return
        
        agent_obs = self.current_observations[agent_id]
        
        if not self.env_instance:
            self.update_status("Environment not initialized. Start the simulation first.")
            return
        
        agent_team_id = self.env_instance.agents[agent_id].get('team', -1) if agent_id < len(self.env_instance.agents) else -1
        world_bounds = {'width': self.env_instance.width, 'height': self.env_instance.height}
        max_steps = self.env_instance.max_steps if hasattr(self.env_instance, 'max_steps') else 1000
        recency_period = getattr(self.env_instance, 'recency_normalization_period', 200.0)
        
        try:
            if self.vis_raw_map_checkbox.isChecked() and 'map' in agent_obs:
                visualize_observation_map(
                    agent_obs['map'], RAW_CH, agent_id, self.current_step,
                    'raw', output_dir=output_dir,
                    obs_radius=self.env_instance.agents[agent_id].get('obs_radius', constants_module.OBS_RADIUS),
                    recency_normalization_period=recency_period,
                    visualization_gamma=self.vis_gamma.value()
                )
            
            if self.vis_memory_map_checkbox.isChecked() and 'memory_map' in agent_obs:
                visualize_observation_map(
                    agent_obs['memory_map'], OCC_CH, agent_id, self.current_step,
                    'memory', output_dir=output_dir,
                    coverage_map=agent_obs.get('coverage_map'),
                    recency_normalization_period=recency_period,
                    coverage_intensity_exponent=self.vis_coverage_exponent.value()
                )
            
            if self.vis_graph_checkbox.isChecked() and 'graph' in agent_obs:
                visualize_graph_observation(
                    agent_obs['graph'], agent_id, agent_team_id, self.current_step,
                    world_bounds, max_steps, recency_period, output_dir=output_dir
                )
            
            self.update_status(f"Exported observation visualization for agent {agent_id} at step {self.current_step}")
        except Exception as e:
            self.update_status(f"Error exporting visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_gif_recording(self):
        """Starts recording frames for GIF generation."""
        if not self.sim_thread or not self.sim_thread.isRunning():
            self.update_status("Simulation is not running. Start the simulation first.")
            return
        
        if not self.current_observations or len(self.current_observations) == 0:
            self.update_status("No observations available yet. Wait for the simulation to run a few steps.")
            return
        
        self.is_recording_gif = True
        self.recording_gif_frames = []
        self.start_gif_button.setEnabled(False)
        self.stop_gif_button.setEnabled(True)
        self.update_status("GIF recording started. Click 'Stop & Save GIF' when done.")
    
    def _add_gif_frame(self, obs_list, step):
        """Adds a frame to the GIF recording."""
        agent_id = self.agent_selector.value()
        
        if not obs_list or agent_id >= len(obs_list):
            return
        
        agent_obs = obs_list[agent_id]
        
        # Store the observation and step for later rendering
        frame_data = {
            'obs': agent_obs,
            'step': step,
            'agent_id': agent_id
        }
        self.recording_gif_frames.append(frame_data)
    
    def _stop_and_save_gif(self):
        """Stops GIF recording and generates the GIF file."""
        if not self.recording_gif_frames:
            self.update_status("No frames recorded. Start recording first.")
            return
        
        self.is_recording_gif = False
        self.start_gif_button.setEnabled(True)
        self.stop_gif_button.setEnabled(False)
        
        if not self.env_instance:
            self.update_status("Environment not initialized.")
            return
        
        agent_id = self.agent_selector.value()
        agent_team_id = self.env_instance.agents[agent_id].get('team', -1) if agent_id < len(self.env_instance.agents) else -1
        world_bounds = {'width': self.env_instance.width, 'height': self.env_instance.height}
        max_steps = self.env_instance.max_steps if hasattr(self.env_instance, 'max_steps') else 1000
        recency_period = getattr(self.env_instance, 'recency_normalization_period', 200.0)
        
        output_dir = self.output_dir_edit.text()
        gif_frame_dir = os.path.join(output_dir, "gif_frames")
        os.makedirs(gif_frame_dir, exist_ok=True)
        
        fps = self.gif_fps.value()
        
        try:
            # Generate all frames
            self.update_status(f"Generating {len(self.recording_gif_frames)} frames for GIF...")
            QApplication.processEvents()
            
            # --- REFINED GIF GENERATION ---
            # Group frames by observation type
            frames_by_type = {'raw': [], 'memory': [], 'graph': []}

            for frame_idx, frame_data in enumerate(self.recording_gif_frames):
                agent_obs = frame_data['obs']
                step = frame_data['step']
                
                # We need to render the frame to a file and store the path
                if self.vis_raw_map_checkbox.isChecked() and 'map' in agent_obs:
                    frame_filename = f"agent_{agent_id}_raw_map_step_{step}.png"
                    frame_path = os.path.join(gif_frame_dir, frame_filename)
                    visualize_observation_map(
                        agent_obs['map'], RAW_CH, agent_id, step, 'raw',
                        output_dir=gif_frame_dir,
                        obs_radius=self.env_instance.agents[agent_id].get('obs_radius', constants_module.OBS_RADIUS),
                        recency_normalization_period=recency_period,
                        visualization_gamma=self.vis_gamma.value()
                    )
                    if os.path.exists(frame_path):
                        frames_by_type['raw'].append(frame_path)

                if self.vis_memory_map_checkbox.isChecked() and 'memory_map' in agent_obs:
                    frame_filename = f"agent_{agent_id}_memory_map_step_{step}.png"
                    frame_path = os.path.join(gif_frame_dir, frame_filename)
                    visualize_observation_map(
                        agent_obs['memory_map'], OCC_CH, agent_id, step, 'memory',
                        output_dir=gif_frame_dir,
                        coverage_map=agent_obs.get('coverage_map'),
                        recency_normalization_period=recency_period,
                        coverage_intensity_exponent=self.vis_coverage_exponent.value()
                    )
                    if os.path.exists(frame_path):
                        frames_by_type['memory'].append(frame_path)
                
                if self.vis_graph_checkbox.isChecked() and 'graph' in agent_obs:
                    frame_filename = f"agent_{agent_id}_graph_step_{step}.png"
                    frame_path = os.path.join(gif_frame_dir, frame_filename)
                    visualize_graph_observation(
                        agent_obs['graph'], agent_id, agent_team_id, step,
                        world_bounds, max_steps, recency_period, output_dir=gif_frame_dir
                    )
                    if os.path.exists(frame_path):
                        frames_by_type['graph'].append(frame_path)

            # Create a GIF for each observation type that has frames
            for obs_type, image_files in frames_by_type.items():
                if not image_files:
                    continue

                self.update_status(f"Creating GIF for {obs_type} map...")
                QApplication.processEvents()
                
                images = []
                # Sort files numerically to ensure correct order
                sorted_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                for img_path in sorted_files:
                    images.append(imageio.imread(img_path))
                
                if not images: continue

                # Add pause frames at the end
                for _ in range(10):
                    images.append(images[-1])
                
                gif_filename = f"agent_{agent_id}_{obs_type}_map_animation.gif"
                gif_path = os.path.join(output_dir, gif_filename)
                imageio.mimsave(gif_path, images, fps=fps)
                self.update_status(f"GIF saved: {gif_path}")

            self.recording_gif_frames = []
            
        except Exception as e:
            self.update_status(f"Error creating GIF: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        """Ensures the simulation is stopped when the GUI window is closed."""
        self.stop_simulation()
        super().closeEvent(event)

def main():
    """Main function to create and run the PyQt application."""
    app = QApplication(sys.argv)
    window = SwarmSimGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
