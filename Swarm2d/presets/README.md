# Simulation Presets

This directory contains JSON files with preset configurations for the `Swarm2d` environment. These presets allow you to quickly load a predefined set of parameters for running a simulation. They are particularly useful for saving and sharing specific experimental setups.

## How It Works

When a simulation is run, especially from the `simulation_gui.py`, it can load a preset file. The key-value pairs in the JSON file will override the default parameters of the `Swarm2DEnv`. This makes it easy to switch between different scenarios without having to manually set every parameter each time.

The GUI also allows you to modify the parameters and save the new configuration as a new preset file in this directory.

## Preset Files

-   `default_scenario.json`: An empty JSON file. Loading this will result in the environment using its default, hard-coded parameters from `env/env.py`. It serves as a blank slate.

-   `combat_scenario.json`: A preset tailored for a combat-focused scenario. It keeps the agent count high but reduces the number of resources. More importantly, it adjusts the reward structure by setting positive rewards for dealing damage (`r_combat_total_damage_dealt`) and killing enemy agents (`r_combat_agent_kill`), while slightly penalizing resource-gathering behaviors. This encourages the agents to learn and exhibit aggressive, combat-oriented behaviors.

## Creating Your Own Presets

You can easily create your own presets:

1.  **Use the GUI:** The recommended method is to use the `simulation_gui.py`. Configure the simulation to your liking and then use the "Save Preset" functionality.
2.  **Manual Creation:** You can create a new `.json` file in this directory. Add any parameter you wish to override from the `Swarm2DEnv`'s `__init__` method as a key-value pair. Any parameter not included in your JSON file will use the environment's default value.
