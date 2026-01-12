#!/usr/bin/env python3
"""
Curriculum Learning Strategy for Multi-Policy Swarm Training

This module implements a progressive curriculum that starts with simpler scenarios
and gradually increases complexity as policies improve.

Curriculum Stages:
1. Single Team (No Competition) - Learn basic behaviors
2. Two Teams (Limited Competition) - Learn cooperation within team
3. Three Teams (Moderate Competition) - Learn strategic positioning
4. Full Six Teams (Full Competition) - Learn advanced tactics
5. Dynamic Difficulty (Adaptive) - Adjust based on performance
"""

import torch
import numpy as np
import math
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import copy
from constants import REWARD_COMPONENT_KEYS

# Define the reward stages for the curriculum
# Each stage enables a new set of behaviors by setting their reward weights to 1.0
# All other rewards are explicitly set to 0.0 to disable them.
CURRICULUM_STAGES = [
    # Stage 0: Basic Movement & Survival
    {
        "name": "Stage 0: Survival",
        "rewards": {
            "r_exploration_intrinsic": 1.0,
            "r_death": -1.0,
            "r_teammate_lost_nearby": -1.0,
        }
    },
    # Stage 1: Resource Gathering
    {
        "name": "Stage 1: Gathering",
        "rewards": {
            "r_resource_found": 1.0,
            "r_attachment": 1.0,
            "r_progress": 1.0,
            "r_delivery": 1.0,
        }
    },
    # Stage 2: Basic Combat
    {
        "name": "Stage 2: Combat",
        "rewards": {
            "r_enemy_found": 1.0,
            "r_combat_continuous": 1.0,
            "r_combat_win": 1.0,
            "r_combat_lose": -1.0,
        }
    },
    # Stage 3: Advanced Tactics
    {
        "name": "Stage 3: Tactics",
        "rewards": {
            "r_grapple_control": 1.0,
            "r_grapple_break": 1.0,
            "r_torque_win": 1.0,
            "r_hive_found": 1.0,
            "r_hive_win": 1.0,
            "r_hive_lose": -1.0,
        }
    },
    # Stage 4: Full Task
    {
        "name": "Stage 4: Full Task",
        "rewards": {key: 1.0 for key in REWARD_COMPONENT_KEYS} # Enable all rewards with default weight
    }
]


class CurriculumManager:
    def __init__(self, num_teams: int):
        self.num_teams = num_teams
        self.current_stage_index = 0
        self.stages = self._build_cumulative_stages()

    def _build_cumulative_stages(self) -> List[Dict]:
        """
        Builds the curriculum stages so that each stage includes the rewards
        from all previous stages.
        """
        cumulative_stages = []
        current_rewards = {}
        for stage_def in CURRICULUM_STAGES:
            # Add the new rewards for this stage to the current set
            current_rewards.update(stage_def["rewards"])
            
            # Create the full reward dictionary for this stage
            # All known rewards not in the current set are set to 0.0
            stage_rewards = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
            stage_rewards.update(current_rewards)

            cumulative_stages.append({
                "name": stage_def["name"],
                "rewards": stage_rewards
            })
        return cumulative_stages

    def get_current_stage_name(self) -> str:
        """Returns the name of the current curriculum stage."""
        return self.stages[self.current_stage_index]["name"]

    def get_current_reward_overrides(self) -> Dict[str, Dict[str, float]]:
        """
        Generates the reward override dictionary for the current stage,
        formatted for all teams.
        """
        stage_rewards = self.stages[self.current_stage_index]["rewards"]
        
        # Apply the same reward structure to all teams
        overrides = {
            str(team_id): stage_rewards
            for team_id in range(self.num_teams)
        }
        return overrides

    def advance_stage(self) -> bool:
        """

        Advances the curriculum to the next stage.

        Returns:
            bool: True if the stage was advanced, False if it was already the last stage.
        """
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            print(f"--- CURRICULUM ADVANCED TO: {self.get_current_stage_name()} ---")
            return True
        else:
            print("--- CURRICULUM: Already at the final stage. ---")
            return False

