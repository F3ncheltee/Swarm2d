#!/usr/bin/env python3
"""
Hyperparameter Search Strategy for Multi-Policy Swarm Training

This module implements a systematic approach to finding optimal hyperparameters
for the three policy types (MAAC, NCA, SharedActor) in the swarm environment.

Strategy:
1. Start with a small-scale search to identify promising regions
2. Use Bayesian optimization for efficient exploration
3. Implement early stopping to avoid wasting compute
4. Use curriculum learning to progressively increase difficulty
"""

import torch
import numpy as np
import optuna
from typing import Dict, List, Tuple, Optional
import copy
import time
import os
from dataclasses import dataclass
from typing import Any
import uuid
import sys
import traceback

# To allow running as a script, add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from Swarm2d.env.env import Swarm2DEnv
from Swarm2d.trainingCustom.run_training import run_training_trial

@dataclass
class SearchConfig:
    """Configuration for hyperparameter search"""
    n_trials: int = 50
    n_startup_trials: int = 10
    n_warmup_steps: int = 100
    n_eval_episodes: int = 3
    early_stopping_patience: int = 5
    min_episodes: int = 50
    max_episodes: int = 200
    timeout_seconds: int = 3600  # 1 hour per trial
    study_name: str = "swarm_hyperopt"
    storage_url: str = "sqlite:///hyperopt.db"

class HyperparameterSearch:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.study = None
        self.best_params = {}
        
    def create_study(self, policy_type: str) -> optuna.Study:
        """Create an Optuna study for a specific policy type"""
        study_name = f"{self.config.study_name}_{policy_type}"
        
        # Create or load study
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.config.storage_url
            )
            print(f"Loaded existing study: {study_name}")
        except:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.config.storage_url,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=self.config.n_startup_trials,
                    n_ei_candidates=24
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=self.config.n_startup_trials,
                    n_warmup_steps=self.config.n_warmup_steps
                )
            )
            print(f"Created new study: {study_name}")
        
        return study
    
    def suggest_maac_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for MAAC policy"""
        return {
            # Learning rates
            "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True),
            
            # Architecture
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
            "temporal_hidden_dim": trial.suggest_categorical("temporal_hidden_dim", [64, 128, 256]),
            "role_embed_dim": trial.suggest_categorical("role_embed_dim", [8, 16, 32]),
            
            # GNN parameters
            "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [32, 64, 128]),
            "gnn_layers": trial.suggest_int("gnn_layers", 1, 4),
            "gnn_heads": trial.suggest_categorical("gnn_heads", [2, 4, 8]),
            
            # Temporal module
            "titan_nhead": trial.suggest_categorical("titan_nhead", [2, 4, 8]),
            "memory_length": trial.suggest_int("memory_length", 5, 20),
            
            # Regularization
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
        }
    
    def suggest_nca_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for NCA policy"""
        return {
            # Learning rates
            "policy_lr": trial.suggest_float("policy_lr", 1e-5, 1e-2, log=True),
            
            # Core architecture
            "h_dim": trial.suggest_categorical("h_dim", [32, 64, 128, 256]),
            "nca_iterations": trial.suggest_int("nca_iterations", 1, 5),
            "belief_dim": trial.suggest_categorical("belief_dim", [8, 16, 32]),
            "readout_hidden_dim": trial.suggest_categorical("readout_hidden_dim", [64, 128, 256]),
            
            # GNN parameters
            "gnn_hidden_dim": trial.suggest_categorical("gnn_hidden_dim", [32, 64, 128]),
            "gnn_layers": trial.suggest_int("gnn_layers", 1, 4),
            "gnn_heads": trial.suggest_categorical("gnn_heads", [2, 4, 8]),
            "msg_dim": trial.suggest_categorical("msg_dim", [16, 32, 64]),
            
            # Specialized modules
            "map_embed_dim": trial.suggest_categorical("map_embed_dim", [16, 32, 64]),
            "cnn_layers": trial.suggest_int("cnn_layers", 1, 3),
            "role_emb_dim": trial.suggest_categorical("role_emb_dim", [8, 16, 32]),
            "sym_emb_dim": trial.suggest_categorical("sym_emb_dim", [4, 8, 16]),
            "physics_feat_dim": trial.suggest_categorical("physics_feat_dim", [8, 16, 32]),
            "logic_net_hidden_dim": trial.suggest_categorical("logic_net_hidden_dim", [16, 24, 32]),
            "modulation_dim": trial.suggest_categorical("modulation_dim", [32, 64, 128]),
            "modulation_type": trial.suggest_categorical("modulation_type", ['gate', 'bias']),
            
            # Memory
            "decentralized_memory_slots": trial.suggest_categorical("decentralized_memory_slots", [8, 16, 32]),
            "decentralized_memory_dim": trial.suggest_categorical("decentralized_memory_dim", [16, 32, 64]),
            
            # Loss coefficients
            "role_entropy_coef": trial.suggest_float("role_entropy_coef", 0.001, 0.1, log=True),
            "aux_loss_coef": trial.suggest_float("aux_loss_coef", 0.01, 0.5, log=True),
            "belief_loss_coef": trial.suggest_float("belief_loss_coef", 0.001, 0.1, log=True),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            
            # Ablation flags (keep most important ones as True)
            "use_decentralized_memory": True,
            "use_neighbor_attention": True,
            "use_neural_logic": trial.suggest_categorical("use_neural_logic", [True, False]),
            "use_symbolic_layer": trial.suggest_categorical("use_symbolic_layer", [True, False]),
            "use_physics_features": True,
            "use_dynamics_pred": trial.suggest_categorical("use_dynamics_pred", [True, False]),
        }
    
    def suggest_shared_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for SharedActor policy"""
        return {
            # Learning rates
            "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True),
            
            # Core architecture
            "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "gru_hidden_dim": trial.suggest_categorical("gru_hidden_dim", [16, 32, 64]),
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
            "role_emb_dim": trial.suggest_categorical("role_emb_dim", [8, 16, 32]),
            "final_hidden_dim": trial.suggest_categorical("final_hidden_dim", [32, 64, 128]),
            
            # GNNs
            "comm_gnn_hidden_dim": trial.suggest_categorical("comm_gnn_hidden_dim", [16, 32, 64]),
            "mem_gnn_hidden_dim": trial.suggest_categorical("mem_gnn_hidden_dim", [16, 32, 64]),
            "mem_gnn_layers": trial.suggest_int("mem_gnn_layers", 1, 3),
            "mem_gnn_heads": trial.suggest_categorical("mem_gnn_heads", [2, 4, 8]),
            
            # Temporal & Memory
            "temporal_layers": trial.suggest_int("temporal_layers", 1, 3),
            "temporal_mem_size": trial.suggest_categorical("temporal_mem_size", [16, 32, 64]),
            
            # Planning & Control
            "latent_plan_dim": trial.suggest_categorical("latent_plan_dim", [12, 24, 48]),
            "llc_hidden_dim": trial.suggest_categorical("llc_hidden_dim", [24, 48, 96]),
            "semantic_dim": trial.suggest_categorical("semantic_dim", [16, 32, 64]),
            "global_context_dim": trial.suggest_categorical("global_context_dim", [16, 32, 64]),
            "role_gating_mlp_dim": trial.suggest_categorical("role_gating_mlp_dim", [16, 32, 64]),
            
            # Loss coefficients
            "role_entropy_coef": trial.suggest_float("role_entropy_coef", 0.001, 0.1, log=True),
            "aux_loss_coef": trial.suggest_float("aux_loss_coef", 0.01, 0.5, log=True),
            "contrastive_loss_coef": trial.suggest_float("contrastive_loss_coef", 0.01, 0.5, log=True),
            "contrastive_tau": trial.suggest_float("contrastive_tau", 0.01, 0.2),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 2.0),
            
            # Ablation flags
            "use_contrastive": trial.suggest_categorical("use_contrastive", [True, False]),
            "use_adaptive_graph_comm": True,
            "use_temporal_gru": True,
            "use_latent_plan": trial.suggest_categorical("use_latent_plan", [True, False]),
            "use_global_context": True,
            "use_dynamics_prediction": trial.suggest_categorical("use_dynamics_prediction", [True, False]),
        }
    
    def objective_function(self, trial: optuna.Trial, policy_type: str, device: torch.device) -> float:
        """Objective function for hyperparameter optimization"""
        try:
            # 1. Get suggested parameters
            if policy_type == "maac":
                params = self.suggest_maac_params(trial)
                trial_hps = {"maac": params}
            elif policy_type == "nca":
                params = self.suggest_nca_params(trial)
                trial_hps = {"nca": params}
            elif policy_type == "shared":
                params = self.suggest_shared_params(trial)
                trial_hps = {"shared": params}
            else:
                raise ValueError(f"Unknown policy type: {policy_type}")
            
            # 2. Set up environment and trial parameters
            # Each trial needs its own environment instance to avoid state pollution
            env = Swarm2DEnv(render_mode=False, debug=False, num_agents_per_team=10, num_teams=6)

            # Unique run name for each trial to avoid overwriting logs
            run_name = f"hpo_{policy_type}_{trial.number}_{uuid.uuid4().hex[:8]}"

            training_params = {
                'run_name': run_name,
                'seed': int(time.time()) % 10000, # Use a different seed for each trial
                'buffer_capacity': 10000, # Smaller buffer for faster HPO runs
                'total_agents': env.num_agents,
            }
            
            print(f"\n--- Starting Trial {trial.number} for {policy_type.upper()} ---")
            print(f"  Run Name: {run_name}")
            print(f"  Params: {params}")

            # 3. Run training with these parameters
            # The run_training_trial function is now the real one, with Optuna integration
            final_reward = run_training_trial(
                env=env, 
                params=training_params,
                trial_hyperparams=trial_hps,
                optuna_trial=trial
            )
            
            # 4. Clean up environment
            # env.close() is now called inside run_training_trial
            
            return final_reward
            
        except optuna.TrialPruned:
            # Re-raise the special exception to signal Optuna
            raise
        except Exception as e:
            print(f"!!!!!! Trial {trial.number} failed with a critical error: {e} !!!!!!")
            traceback.print_exc()
            # Return a very bad score to penalize this parameter set
            return -float('inf')
    
    def run_training_trial(self, params: Dict[str, Any], policy_type: str, env, device: torch.device, trial: optuna.Trial) -> float:
        """
        [DEPRECATED] This function is now replaced by the objective_function calling the main 
        training script directly. Keeping it for reference.
        """
        # This would integrate with your existing training loop
        return 0.0
    
    def run_search(self, policy_types: List[str], device: torch.device) -> Dict[str, Dict[str, Any]]:
        """Run hyperparameter search for multiple policy types"""
        results = {}
        
        for policy_type in policy_types:
            print(f"\n{'='*50}")
            print(f"Starting hyperparameter search for {policy_type.upper()}")
            print(f"{'='*50}")
            
            study = self.create_study(policy_type)
            
            # Create objective function with policy type and device
            objective = lambda trial: self.objective_function(trial, policy_type, device)
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds
            )
            
            # Store results
            results[policy_type] = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }
            
            print(f"\nBest parameters for {policy_type}:")
            print(f"  Best value: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")
        
        return results

def run_hyperparameter_search():
    """Main function to run hyperparameter search"""
    config = SearchConfig(
        n_trials=20,  # Start with fewer trials for testing
        n_startup_trials=5,
        timeout_seconds=1800,  # 30 minutes per trial
    )
    
    searcher = HyperparameterSearch(config)
    
    # Mock environment and device for demonstration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run search for all policy types
    policy_types = ["maac", "nca", "shared"]
    results = searcher.run_search(policy_types, device)
    
    # Save results
    import json
    with open("hyperparameter_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nHyperparameter search completed!")
    print("Results saved to hyperparameter_search_results.json")
    
    return results

if __name__ == "__main__":
    run_hyperparameter_search()

