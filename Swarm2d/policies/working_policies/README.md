# Working Policies

This directory contains a collection of policies that are considered stable and functional. It includes baseline policies, templates, and wrappers for trained models.

## Core Policies

-   **/randomPolicy**: Implements the most basic policy where agents select their actions randomly. This is an essential baseline for ensuring that learned policies are performing better than pure chance.

-   **/trainedRLPolicy**: This is a crucial wrapper class for deploying a trained reinforcement learning model. The `TrainedRLPolicy` is designed to load the weights of a saved actor network (e.g., one of the architectures from `/policies/actors`) and use it for inference during simulation. It handles the processing of observations and the conversion of network outputs into environment actions.

## Other Policies & Templates

-   `simple_heuristic_policy.py`: A more streamlined or simplified version of the main `/policies/heuristicPolicy`.

-   `policy_template.py`: Provides a boilerplate template for creating new policies. It outlines the required class structure and methods (like `act`), making it easier to implement new agent behaviors while ensuring compatibility with the simulation environment.
