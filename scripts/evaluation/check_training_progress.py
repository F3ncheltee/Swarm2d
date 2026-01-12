"""Quick check if single-agent training is complete"""
import os
import time

print("\n" + "="*70)
print("TRAINING STATUS CHECK")
print("="*70)

# Check if plot exists (indicates completion)
if os.path.exists('single_agent_convergence.png'):
    mod_time = os.path.getmtime('single_agent_convergence.png')
    age = time.time() - mod_time
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Plot created: single_agent_convergence.png")
    print(f"   (created {age/60:.1f} minutes ago)")
    print(f"\n📊 Open the plot to see your convergence curves!")
else:
    print("⏳ Training still running...")
    print("   Expected time: ~20-30 minutes")
    
    # Check if output file exists
    if os.path.exists('training_output.txt'):
        # Read last few lines
        with open('training_output.txt', 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                print(f"\n   Last output line:")
                print(f"   {lines[-1].strip()}")
    else:
        print("   No output file yet...")

print("="*70 + "\n")



