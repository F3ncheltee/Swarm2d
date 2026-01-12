#!/usr/bin/env python3
"""
Comprehensive Observation Debug System

This module provides detailed debugging for all observation components:
- Observation structure and tensor shapes
- Foveation and occlusion analysis
- Memory map inspection
- Unified graph analysis
- Observation radius and spatial relationships
"""

import torch
import numpy as np
import json
from typing import Dict, Any, Optional, Union, List
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import os

class ObservationDebugger:
    def __init__(self, save_debug_images: bool = False, debug_dir: str = "debug_observations", print_tensor_values: bool = False):
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        self.print_tensor_values = print_tensor_values
        if self.save_debug_images:
            os.makedirs(debug_dir, exist_ok=True)
    
    def debug_observation_structure(self, obs: Dict[str, Any], agent_idx: int, step: int) -> Dict[str, Any]:
        """Debug the overall observation structure"""
        debug_info = {
            'agent_idx': agent_idx,
            'step': step,
            'observation_keys': list(obs.keys()),
            'tensor_info': {},
            'graph_info': {},
            'other_info': {}
        }
        
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                debug_info['tensor_info'][key] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'range': [float(value.min().item()), float(value.max().item())],
                    'mean': float(value.mean().item()),
                    'std': float(value.std().item()),
                    'sparsity': float((value == 0).float().mean().item())
                }
                if self.print_tensor_values:
                    # Store a truncated version of the tensor values
                    truncated_values = value.detach().cpu().numpy().flatten()
                    if truncated_values.size > 10:
                        truncated_values = np.concatenate([truncated_values[:5], truncated_values[-5:]])
                    debug_info['tensor_info'][key]['values'] = truncated_values.tolist()

            # This should handle PyG Data, Batch, and HeteroData
            elif 'torch_geometric.data' in str(type(value)):
                graph_info = {
                    'num_nodes': value.num_nodes if hasattr(value, 'num_nodes') else 0,
                    'num_edges': value.num_edges if hasattr(value, 'num_edges') else 0,
                    'x_shape': list(value.x.shape) if hasattr(value, 'x') and value.x is not None else [0],
                    'edge_attr_shape': list(value.edge_attr.shape) if hasattr(value, 'edge_attr') and value.edge_attr is not None else None,
                    'has_self_loops': bool(value.has_self_loops()) if hasattr(value, 'has_self_loops') else None,
                    'is_directed': bool(value.is_directed()) if hasattr(value, 'is_directed') else None
                }
                debug_info['graph_info'][key] = graph_info
            elif isinstance(value, dict):
                debug_info['other_info'][key] = debug_observation_structure(value, agent_idx, step) # Recurse for nested dicts
            else:
                debug_info['other_info'][key] = {
                    'type': str(type(value)),
                    'value': str(value) if not callable(value) else 'callable'
                }
        
        return debug_info
    
    def debug_foveation_occlusion(self, agent_pos: np.ndarray, obs_radius: float, 
                                 fovea_radius: float, occlusion_map: Optional[torch.Tensor] = None,
                                 agent_idx: int = 0, step: int = 0) -> Dict[str, Any]:
        """Debug foveation and occlusion analysis"""
        debug_info = {
            'agent_idx': agent_idx,
            'step': step,
            'spatial_info': {
                'agent_position': agent_pos.tolist(),
                'observation_radius': float(obs_radius),
                'fovea_radius': float(fovea_radius),
                'fovea_ratio': float(fovea_radius / obs_radius) if obs_radius > 0 else 0.0
            }
        }
        
        if occlusion_map is not None:
            debug_info['occlusion_info'] = {
                'map_shape': list(occlusion_map.shape),
                'total_pixels': int(occlusion_map.numel()),
                'occluded_pixels': int((occlusion_map > 0).sum().item()),
                'occlusion_ratio': float((occlusion_map > 0).float().mean().item()),
                'channel_stats': {}
            }
            
            # Analyze each channel
            for ch in range(occlusion_map.shape[0]):
                channel_data = occlusion_map[ch]
                debug_info['occlusion_info']['channel_stats'][f'channel_{ch}'] = {
                    'range': [float(channel_data.min().item()), float(channel_data.max().item())],
                    'mean': float(channel_data.mean().item()),
                    'sparsity': float((channel_data == 0).float().mean().item())
                }
        
        return debug_info
    
    def debug_memory_map(self, memory_map: torch.Tensor, agent_idx: int, step: int) -> Dict[str, Any]:
        """Debug memory map structure and content"""
        debug_info = {
            'agent_idx': agent_idx,
            'step': step,
            'memory_map_info': {
                'shape': list(memory_map.shape),
                'channels': int(memory_map.shape[0]),
                'spatial_resolution': list(memory_map.shape[1:]),
                'total_elements': int(memory_map.numel()),
                'channel_analysis': {}
            }
        }
        
        # Analyze each channel
        for ch in range(memory_map.shape[0]):
            channel_data = memory_map[ch]
            debug_info['memory_map_info']['channel_analysis'][f'channel_{ch}'] = {
                'range': [float(channel_data.min().item()), float(channel_data.max().item())],
                'mean': float(channel_data.mean().item()),
                'std': float(channel_data.std().item()),
                'sparsity': float((channel_data == 0).float().mean().item()),
                'non_zero_elements': int((channel_data != 0).sum().item())
            }
        
        return debug_info
    
    def debug_unified_graph(self, graph: Union[Data, Batch], agent_idx: int, step: int) -> Dict[str, Any]:
        """Debug unified graph structure and connectivity"""
        debug_info = {
            'agent_idx': agent_idx,
            'step': step,
            'graph_info': {
                'num_nodes': int(graph.num_nodes),
                'num_edges': int(graph.num_edges),
                'node_features_shape': list(graph.x.shape) if graph.x is not None else None,
                'edge_features_shape': list(graph.edge_attr.shape) if graph.edge_attr is not None else None,
                'is_directed': bool(graph.is_directed()) if hasattr(graph, 'is_directed') else None,
                'has_self_loops': bool(graph.has_self_loops()) if hasattr(graph, 'has_self_loops') else None
            }
        }
        
        # Analyze connectivity
        if graph.num_edges > 0:
            degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
            for i in range(graph.num_nodes):
                degrees[i] = (graph.edge_index[0] == i).sum() + (graph.edge_index[1] == i).sum()
            
            debug_info['graph_info']['connectivity'] = {
                'max_degree': int(degrees.max().item()),
                'min_degree': int(degrees.min().item()),
                'avg_degree': float(degrees.float().mean().item()),
                'isolated_nodes': int((degrees == 0).sum().item()),
                'highly_connected_nodes': int((degrees > degrees.float().mean()).sum().item())
            }
        
        # Analyze node features
        if graph.x is not None:
            debug_info['graph_info']['node_features'] = {
                'feature_dim': int(graph.x.shape[1]),
                'feature_range': [float(graph.x.min().item()), float(graph.x.max().item())],
                'feature_mean': float(graph.x.mean().item()),
                'feature_std': float(graph.x.std().item())
            }
        
        # Analyze edge features
        if graph.edge_attr is not None:
            debug_info['graph_info']['edge_features'] = {
                'feature_dim': int(graph.edge_attr.shape[1]),
                'feature_range': [float(graph.edge_attr.min().item()), float(graph.edge_attr.max().item())],
                'feature_mean': float(graph.edge_attr.mean().item()),
                'feature_std': float(graph.edge_attr.std().item())
            }
        
        return debug_info
    
    def debug_observation_radius(self, agent_pos: np.ndarray, obs_radius: float, 
                               other_agents: List[Dict], agent_idx: int, step: int) -> Dict[str, Any]:
        """Debug observation radius and spatial relationships"""
        debug_info = {
            'agent_idx': agent_idx,
            'step': step,
            'radius_info': {
                'agent_position': agent_pos.tolist(),
                'observation_radius': float(obs_radius),
                'nearby_agents': [],
                'total_agents_in_radius': 0
            }
        }
        
        # Find agents within observation radius
        for i, other_agent in enumerate(other_agents):
            if other_agent and i != agent_idx:
                other_pos = np.array(other_agent.get('pos', [0, 0]))
                distance = np.linalg.norm(agent_pos - other_pos)
                
                if distance <= obs_radius:
                    debug_info['radius_info']['nearby_agents'].append({
                        'agent_id': i,
                        'position': other_pos.tolist(),
                        'distance': float(distance),
                        'team': other_agent.get('team', -1),
                        'alive': other_agent.get('alive', False)
                    })
                    debug_info['radius_info']['total_agents_in_radius'] += 1
        
        return debug_info
    
    def print_debug_summary(self, debug_info: Dict[str, Any], agent_idx: int, step: int):
        """Print a comprehensive debug summary"""
        print(f"\n{'='*60}\nOBSERVATION DEBUG - Agent {agent_idx}, Step {step}\n{'='*60}\n")
        
        print("--- Observation Structure ---")
        for key, value in debug_info.items():
            try:
                if isinstance(value, dict) and 'type' in value:
                    if value['type'] == 'Tensor':
                        print(f"  {key}: [{', '.join(map(str, value['shape']))}] {value['dtype']} on {value['device']}")
                        print(f"    Range: [{value['range'][0]:.4f}, {value['range'][1]:.4f}], Mean: {value['mean']:.4f}")
                        if 'values' in value:
                            formatted_values = [f"{v:.4f}" for v in value['values']]
                            ellipsis = "..." if len(value['values']) > 10 else ""
                            print(f"    Values: [{', '.join(formatted_values[:5])} {ellipsis} {', '.join(formatted_values[-5:])}]")
                    elif value['type'] == 'Graph':
                        print(f"  {key}: Graph with {value.get('num_nodes', 'N/A')} nodes, {value.get('num_edges', 'N/A')} edges")
                elif isinstance(value, dict) and 'num_nodes' in value: # A simpler check for graph-like dicts
                     print(f"  {key}: Graph with {value.get('num_nodes', 'N/A')} nodes, {value.get('num_edges', 'N/A')} edges")
                else:
                    # Fallback for any other dictionary or data type
                    print(f"  {key}: {str(value)}")
            except Exception as e:
                print(f"  Could not print summary for key '{key}': {e}")
        
        # Foveation/Occlusion
        if 'spatial_info' in debug_info:
            spatial = debug_info['spatial_info']
            print(f"\n--- Foveation/Occlusion ---")
            print(f"  Agent position: {spatial['agent_position']}")
            print(f"  Observation radius: {spatial['observation_radius']:.2f}")
            print(f"  Fovea radius: {spatial['fovea_radius']:.2f}")
            print(f"  Fovea ratio: {spatial['fovea_ratio']:.3f}")
            
            if 'occlusion_info' in debug_info:
                occ = debug_info['occlusion_info']
                print(f"  Occlusion map: {occ['map_shape']} {type(occ['map_shape'])}")
                print(f"  Occlusion ratio: {occ['occlusion_ratio']:.3f}")
                print(f"  Occluded pixels: {occ['occluded_pixels']}/{occ['total_pixels']}")
        
        # Memory Map
        if 'memory_map_info' in debug_info:
            mem = debug_info['memory_map_info']
            print(f"\n--- Memory Map ---")
            print(f"  Shape: {mem['shape']}, Channels: {mem['channels']}")
            for ch_name, ch_info in mem['channel_analysis'].items():
                print(f"  {ch_name}: Range=[{ch_info['range'][0]:.4f}, {ch_info['range'][1]:.4f}], Sparsity={ch_info['sparsity']:.3f}")
        
        # Unified Graph
        if 'graph_info' in debug_info:
            graph = debug_info['graph_info']
            print(f"\n--- Unified Graph ---")
            print(f"  Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")
            if 'connectivity' in graph:
                conn = graph['connectivity']
                print(f"  Max degree: {conn['max_degree']}, Avg degree: {conn['avg_degree']:.2f}")
                print(f"  Isolated nodes: {conn['isolated_nodes']}, Highly connected: {conn['highly_connected_nodes']}")
        
        # Observation Radius
        if 'radius_info' in debug_info:
            radius = debug_info['radius_info']
            print(f"\n--- Observation Radius ---")
            print(f"  Agents in radius: {radius['total_agents_in_radius']}")
            for agent in radius['nearby_agents'][:5]:  # Show first 5
                print(f"    Agent {agent['agent_id']}: dist={agent['distance']:.2f}, team={agent['team']}")
        
        print(f"{'='*60}")
    
    def save_debug_data(self, debug_info: Dict[str, Any], filename: str):
        """Save debug data to JSON file (with proper serialization)"""
        if not self.save_debug_images:
            return
        
        # Create a serializable version of debug_info
        serializable_info = self._make_serializable(debug_info)
        
        filepath = os.path.join(self.debug_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_info, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)

# Global instance
observation_debugger = ObservationDebugger(save_debug_images=False, print_tensor_values=True)
