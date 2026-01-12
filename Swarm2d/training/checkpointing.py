
import os

### Checkpoint helper
def find_latest_checkpoint(chkpt_dir, specific_episode=None):
    """ Finds the latest valid checkpoint episode in the directory. """
    latest_episode = -1
    latest_path_base = None # We don't really need this if we reconstruct path
    if not os.path.isdir(chkpt_dir):
        return latest_episode # Return -1 if dir doesn't exist

    episodes_found = set()
    try:
        # Look for training state files as a primary indicator
        training_state_files = glob.glob(os.path.join(chkpt_dir, "training_state_ep*.pt"))
        for fname in training_state_files:
            try:
                ep_str = fname.split('_ep')[-1].split('.pt')[0]
                if ep_str.isdigit():
                    episodes_found.add(int(ep_str))
            except Exception:
                continue # Ignore files that don't match pattern

        if not episodes_found:
            # Fallback: look for policy files if no training_state files found
            policy_files = glob.glob(os.path.join(chkpt_dir, "T*_ep*_policy_*.pt"))
            for fname in policy_files:
                 try:
                     ep_str = fname.split('_ep')[-1].split('_')[0]
                     if ep_str.isdigit():
                         episodes_found.add(int(ep_str))
                 except Exception:
                    continue

        if specific_episode is not None:
            if specific_episode in episodes_found:
                latest_episode = specific_episode
            else:
                print(f"Warning: Specified checkpoint episode {specific_episode} not found in {chkpt_dir}.")
                return -1
        elif episodes_found:
            latest_episode = max(episodes_found)
        else:
             print(f"No valid checkpoint episode files found in {chkpt_dir}.")
             return -1 # No episodes found

    except Exception as e_scan:
        print(f"Error scanning checkpoint directory {chkpt_dir}: {e_scan}")
        return -1

    # Return only the episode number. Path base will be reconstructed during loading.
    return latest_episode
