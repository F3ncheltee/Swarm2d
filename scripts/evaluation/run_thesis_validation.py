#!/usr/bin/env python
"""Simple wrapper to run thesis validation"""

import sys
print("Script started!", flush=True)

try:
    from thesis_quick_validation import run_validation
    print("Import successful!", flush=True)
    
    print("\nRunning validation with 1 episode, 50 steps...", flush=True)
    run_validation(num_episodes=1, max_steps=50)
    print("\nValidation complete!", flush=True)
    
except Exception as e:
    print(f"\nERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)



