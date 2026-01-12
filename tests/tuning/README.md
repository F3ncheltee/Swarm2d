# Test & Tuning Scripts

This directory contains various scripts used for testing specific functionalities of the `Swarm2d` environment and for tuning the parameters of the physics engine, as well as obvisouly testing the environment logic and game mechanics.

These scripts are not part of the main simulation or training pipelines but serve as valuable tools for development, debugging, and fine-tuning the environment's behavior.

## Feature Test Scripts

These scripts are designed to isolate and test specific mechanics within the environment.

-   `test_pickup.py`: A test script focused on the resource pickup mechanic. It likely creates a simple scenario to verify that agents can correctly interact with and carry resources.
-   `combat_test.py`: A script for testing the basic combat mechanics, such as damage and agent death.
-   `combat_testWgrapple.py`: An extension of the combat test that specifically includes the grappling mechanic, allowing for focused testing of this complex interaction.

## Curriculum Tuning Scripts

This set of scripts is designed to fine-tune the environment parameters for different stages of a curriculum learning plan. Each script corresponds to a specific "phase" of learning, starting from basic agent movement and progressing to complex combat.

The goal of these scripts is to find the optimal settings (e.g., forces, rewards, agent capabilities) for each phase to ensure that agents can learn the desired behaviors effectively before moving on to the next, more complex stage.

-   `tune_phase0_agent_movement.py`: Tunes basic agent movement parameters.
-   `tune_phase1_agent_agility.py`: Focuses on tuning parameters related to agent agility and responsiveness.
-   `tune_phase2_single_carry.py`: Tunes the mechanics for a single agent carrying a resource.
-   `tune_phase3_agent_grapple.py`: Focuses on tuning the grappling mechanic.
-   `tune_phase4_coop_carry.py`: Tunes the physics and rewards for cooperative resource carrying.
-   `tune_phase5_combat.py`: A comprehensive script for tuning all parameters related to multi-agent combat.
