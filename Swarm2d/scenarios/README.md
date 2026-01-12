# Simulation Scenarios

This directory is intended to hold custom simulation scenario files. A scenario file defines the specific configuration of the environment for a particular experiment or task. This can include:

*   The number of teams and agents.
*   The number and placement of resources and obstacles.
*   The specific capabilities and parameters of the agents.
*   The reward structure for the task.

## Scenario File Format

Scenario files are typically in a format like JSON or YAML, which allows for easy loading and parsing. They contain key-value pairs that override the default environment parameters defined in `env/env.py`.

By saving different configurations as separate scenario files, you can easily switch between various experimental setups and ensure that your experiments are reproducible.

## Usage

While the `presets` directory contains some default, general-purpose configurations, this `scenarios` directory is the ideal place to store your own custom setups for specific research questions or tasks.

You can load a scenario file from the `simulation_gui.py` or specify it as a command-line argument when running a simulation. (Note: The implementation for loading from this specific directory may need to be added to the simulation scripts if it doesn't exist already).

### Example Scenario (`example_scenario.json` - illustrative)

```json
{
  "num_teams": 2,
  "num_agents_per_team": 5,
  "num_resources": 50,
  "num_obstacles": 5,
  "max_steps": 2000,
  "width": 800,
  "height": 800,
  "team_reward_overrides": {
    "0": {
      "r_resource_delivered": 25.0,
      "r_agent_killed": -10.0
    },
    "1": {
      "r_resource_delivered": 25.0,
      "r_agent_killed": -10.0
    }
  }
}
```
