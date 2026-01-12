"""
Simple RL Training to Demonstrate Learning is Possible
Uses basic REINFORCE algorithm with a simple MLP policy
Designed to show learning curves in 3-6 hours
"""
import argparse
import sys
import os

# Add root to sys.path to allow importing Swarm2d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Swarm2d.training.graph_trainer import GraphRLTrainer

def main():
    parser = argparse.ArgumentParser(description='Simple RL training to demonstrate learning')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes (default: 200)')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Max steps per episode (default: 300)')
    parser.add_argument('--output-dir', type=str, default='rl_training_results',
                       help='Output directory (default: rl_training_results)')
    
    args = parser.parse_args()
    
    # Run training
    trainer = GraphRLTrainer(num_episodes=args.episodes, max_steps=args.max_steps)
    trainer.train()
    trainer.save_results(args.output_dir)
    
    print("\n" + "=" * 70)
    print("LEARNING DEMONSTRATION COMPLETE!")
    print(f"Check {args.output_dir}/ for results and learning curves")
    print("=" * 70)

if __name__ == "__main__":
    main()
