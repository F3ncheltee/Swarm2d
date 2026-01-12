"""
FAST RL Training - MAP OBSERVATIONS ONLY (No Graphs!)
Much faster than graph-based version
Uses raw map observations + memory map + self vector with CNN Policy
"""
import argparse
import sys
import os

# Add root to sys.path to allow importing Swarm2d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.training.map_trainer import MapRLTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='resource')
    parser.add_argument('--episodes', type=int, default=500) # Reduced from 500
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--opponent', type=str, default=None, help='Path to checkpoint for Team 1 (Combat only)')
    parser.add_argument('--team2', type=str, default=None, help='Path to checkpoint for Team 2 (Combat only)')
    args = parser.parse_args()
    
    trainer = MapRLTrainer(num_episodes=args.episodes, scenario=args.scenario, opponent_checkpoint=args.opponent, team2_checkpoint=args.team2)
    if args.resume:
        trainer.load_checkpoint()
    trainer.train()

if __name__ == "__main__":
    main()
