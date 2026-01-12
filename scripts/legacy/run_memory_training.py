"""
FAST RL Training - MEMORY OBSERVATIONS ONLY
Uses ONLY the persistent memory map (occlusion map) + self vector.
Proves that the Memory Map contains sufficient information for exploration/tasks.
"""
import argparse
import sys
import os

# Add root to sys.path to allow importing Swarm2d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.training.memory_trainer import MemoryRLTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='exploration')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from (or 'latest' for default)")
    parser.add_argument('--opponent', type=str, default=None, help='Path to checkpoint for Team 1 (Opponent)')
    args = parser.parse_args()
    
    trainer = MemoryRLTrainer(num_episodes=args.episodes, scenario=args.scenario, opponent_path=args.opponent)
    
    if args.resume:
        ckpt_path = None if args.resume == 'latest' else args.resume
        trainer.load_checkpoint(ckpt_path)
        
    trainer.train()

if __name__ == "__main__":
    main()
