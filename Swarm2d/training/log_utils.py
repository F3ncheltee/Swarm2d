from typing import Dict, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Batch
import os
import time
from torch.utils.tensorboard import SummaryWriter

from constants import MEM_NODE_FEATURE_DIM

class Logger:
    """Simple logger class for training logs."""
    
    def __init__(self, log_dir: str = "logs", run_name: str = "training", use_tensorboard: bool = False):
        self.log_dir = log_dir
        self.run_name = run_name
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        # Create base log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # --- TensorBoard Setup ---
        if self.use_tensorboard:
            try:
                # Create a specific directory for this run's TensorBoard logs
                tb_log_dir = os.path.join(log_dir, run_name)
                self.writer = SummaryWriter(log_dir=tb_log_dir)
                print(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_dir}")
            except ImportError:
                print("Warning: tensorboard package not found. TensorBoard logging will be disabled.")
                self.use_tensorboard = False
        
        # --- File Logging Setup ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{run_name}_{timestamp}.log")
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message to file and console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def log_metrics_dict(self, metrics: Dict, step: int):
        """Logs a dictionary of metrics to TensorBoard."""
        if self.use_tensorboard and self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def close(self):
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")

def log_input_details(name: str, obs_dict: Dict, is_critic_obs: bool = False):
    """Prints a detailed, formatted summary of an observation dictionary."""
    if not obs_dict:
        print(f"--- Details for {name}: (Observation is None or Empty) ---")
        return

    print("\n" + "="*80)
    print(f"====== INPUT DETAILS FOR: {name} ======")

    # For critics, the obs is a sequence. We'll inspect the last item.
    obs = obs_dict[-1] if is_critic_obs else obs_dict

    # --- Self Vector (Actors) ---
    if 'self' in obs and obs['self'] is not None:
        s = obs['self']
        print(f"  > self (Tensor): Shape={list(s.shape)}, Device={s.device}, Mean={s.mean():.3f}")

    # --- Map Tensors ---
    for map_key in ['map', 'raw_map', 'occ_map', 'memory_map']:
        if map_key in obs and obs[map_key] is not None:
            m = obs[map_key]
            print(f"  > {map_key} (Tensor): Shape={list(m.shape)}, Sum={m.sum():.2f}, Mean={m.mean():.4f}")

    # --- Cue Vector (Critics) ---
    if 'cues' in obs and obs['cues'] is not None:
        c = obs['cues']
        print(f"  > cues (Tensor): Shape={list(c.shape)}, Mean={c.mean():.3f}, Values[0:3]={c.squeeze()[:3].cpu().numpy().round(2)}")

    # --- Graph Objects ---
    for graph_key in ['graph', 'env_graph', 'memory_graph']:
        if graph_key in obs and obs[graph_key] is not None:
            g = obs[graph_key]
            is_batch = isinstance(g, Batch)
            num_graphs = g.num_graphs if is_batch else 1
            # Check for the 'count' feature added during clustering
            has_count_feature = hasattr(g, 'x') and g.x is not None and g.x.shape[1] == MEM_NODE_FEATURE_DIM + 1
            
            print(f"  > {graph_key} ({type(g).__name__}): NumGraphs={num_graphs}, Nodes={g.num_nodes}, Edges={g.num_edges}")
            if hasattr(g, 'x') and g.x is not None:
                print(f"    - Node Features (x): Shape={list(g.x.shape)}, Clustered (has 'count' feat): {has_count_feature}")
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                 print(f"    - Edge Features (edge_attr): Shape={list(g.edge_attr.shape)}")
    
    print("="*80 + "\n")