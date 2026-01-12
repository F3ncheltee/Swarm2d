# Heuristic Policy

This directory contains the implementation of a rule-based, heuristic policy for the agents in the `Swarm2d` environment.

## Description

The `HeuristicPolicy` is a scripted policy that does not involve any learning. Instead, it uses a set of predefined rules and state machines to determine the agent's actions based on its observations. This type of policy is very useful as a baseline for measuring the performance of more complex, learned policies.

### Core Logic

The primary logic in `heuristic_policy.py` revolves around a state machine that guides the agent's behavior. The agent can be in one of several states, such as:

-   **Searching for Resources:** The agent explores the environment to find resources.
-   **Collecting Resource:** Once a resource is found, the agent moves towards it and attempts to pick it up.
-   **Returning to Hive:** After collecting a resource, the agent navigates back to its hive to drop it off.
-   **Engaging Enemy:** If an enemy agent is nearby, the agent may switch to a combat state to attack.

The policy uses the agent's graph-based observation to identify nearby entities (resources, enemies, hives) and makes decisions based on proximity and its current state.

## Usage

This policy can be selected in the `simulation_gui.py` or from the command line with `run_simulation.py`. Because it is deterministic and requires no training, it provides a stable and predictable benchmark for evaluating other RL-based policies.
