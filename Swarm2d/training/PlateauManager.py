from typing import List, Dict
from collections import deque
import numpy as np

class TeamPlateauManager:
    """
    Manages the training state of each team, stopping training for teams
    that have reached a learning plateau based on a performance metric.
    """
    def __init__(self,
                 team_ids: List[int],
                 window_size: int = 20,
                 patience: int = 5,
                 min_episodes: int = 50,
                 min_improvement_threshold: float = 0.01):
        """
        Args:
            team_ids (List[int]): A list of team IDs to manage.
            window_size (int): The number of recent episodes to average for performance evaluation.
            patience (int): How many check intervals to wait without improvement before stopping a team.
            min_episodes (int): Minimum number of episodes to run before checking for plateaus.
            min_improvement_threshold (float): The minimum relative improvement required to reset patience.
        """
        print("--- Initializing TeamPlateauManager ---")
        self.team_ids = team_ids
        self.window_size = window_size
        self.patience = patience
        self.min_episodes = min_episodes
        self.min_improvement_threshold = min_improvement_threshold
        self.check_interval = window_size // 2  # Check every 10 episodes by default

        self.performance_history = {tid: deque(maxlen=window_size) for tid in team_ids}
        self.best_performance = {tid: -float('inf') for tid in team_ids}
        self.patience_counters = {tid: 0 for tid in team_ids}
        self.is_frozen = {tid: False for tid in team_ids}
        self.total_teams_frozen = 0
        print(f"  - Window: {window_size}, Patience: {patience}, Min Episodes: {min_episodes}, Threshold: {min_improvement_threshold}")

    def update(self, episode: int, team_rewards: Dict[int, float]):
        """
        Updates the manager with the latest reward data from the episode.
        Should be called at the end of every episode.
        """
        if episode < self.min_episodes:
            return # Don't start checking until we have enough data

        for team_id, reward in team_rewards.items():
            if team_id not in self.team_ids or self.is_frozen[team_id]:
                continue # Skip teams not managed or already frozen

            self.performance_history[team_id].append(reward)

            # Check for plateau only at specified intervals and if the window is full
            if episode % self.check_interval == 0 and len(self.performance_history[team_id]) == self.window_size:
                self._check_plateau(episode, team_id)

    def _check_plateau(self, episode: int, team_id: int):
        """Checks if a single team has plateaued."""
        current_performance = np.mean(self.performance_history[team_id])
        best_performance = self.best_performance[team_id]

        # Check for significant improvement
        relative_improvement = (current_performance - best_performance) / (abs(best_performance) + 1e-6)

        if relative_improvement > self.min_improvement_threshold:
            print(f"[PlateauManager] Ep {episode}, Team {team_id}: Performance improved to {current_performance:.2f} (from {best_performance:.2f}). Resetting patience.")
            self.best_performance[team_id] = current_performance
            self.patience_counters[team_id] = 0 # Reset patience
        else:
            self.patience_counters[team_id] += 1
            print(f"[PlateauManager] Ep {episode}, Team {team_id}: No significant improvement. Current: {current_performance:.2f}, Best: {best_performance:.2f}. Patience: {self.patience_counters[team_id]}/{self.patience}")

        if self.patience_counters[team_id] >= self.patience:
            self.is_frozen[team_id] = True
            self.total_teams_frozen += 1
            print(f"\n!!!!!!!!! TEAM {team_id} PLATEAUED AND IS NOW FROZEN !!!!!!!!!\n")

    def is_team_frozen(self, team_id: int) -> bool:
        """Returns True if the team's training should be stopped."""
        return self.is_frozen.get(team_id, False)

    def all_teams_frozen(self) -> bool:
        """Returns True if all managed teams are frozen."""
        return self.total_teams_frozen == len(self.team_ids)



class EpisodeEarlyStopper:
    """
    Terminates an episode early if it becomes unproductive, defined by a lack
    of progress in key reward metrics over a sustained period.
    """
    def __init__(self,
                 num_teams: int,
                 patience: int = 250,        # Num steps to wait with no progress before stopping.
                 window_size: int = 100,     # Moving average window for progress score.
                 min_steps: int = 300,       # Min steps before stopper becomes active.
                 progress_threshold: float = 0.005 # Min avg progress score required to be considered "productive".
                 ):
        self.num_teams = num_teams
        self.patience = patience
        self.window_size = window_size
        self.min_steps = min_steps
        self.progress_threshold = progress_threshold
        self.reset()
        print(f"--- Initializing EpisodeEarlyStopper ---")
        print(f"  - Patience: {patience} steps, Window: {window_size} steps, Threshold: {progress_threshold}")

    def reset(self):
        """Resets the stopper for a new episode."""
        self.progress_history = deque(maxlen=self.window_size)
        self.stalled_steps_counter = 0

    def update(self, rewards_per_agent: List[Dict[str, float]], step: int):
        """
        Updates the stopper with the rewards from the current step.
        Args:
            rewards_per_agent: A list of reward dictionaries for all agents.
            step: The current step number in the episode.
        """
        if step < self.min_steps:
            return

        # Calculate the total "progress" across all agents this step
        # Progress is defined by delivering resources, winning combat/hives.
        total_progress_this_step = 0.0
        for r_dict in rewards_per_agent:
            total_progress_this_step += r_dict.get('r_delivery', 0.0)
            total_progress_this_step += r_dict.get('r_progress', 0.0) # Progress towards hive is important
            total_progress_this_step += r_dict.get('r_hive_win', 0.0)
            # A bit of combat can also be progress
            total_progress_this_step += r_dict.get('r_combat_win', 0.0) * 0.1

        self.progress_history.append(total_progress_this_step)

        # Only check for stalling if the history window is full
        if len(self.progress_history) == self.window_size:
            avg_progress = sum(self.progress_history) / self.window_size
            if avg_progress < self.progress_threshold:
                self.stalled_steps_counter += 1
            else:
                self.stalled_steps_counter = 0 # Reset counter if progress is made

    def should_stop(self) -> bool:
        """Returns True if the episode should be terminated early."""
        return self.stalled_steps_counter >= self.patience
