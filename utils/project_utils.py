import os
from datetime import datetime
import shutil
from pathlib import Path
import yaml

def update_experiment_path(args, short=False):
    path_components = []
    mode = "ft" if args.pretrained_lora_path else "scratch"
    path_components.append(f"{args.train_architecture}")
    if args.train_architecture == "lora":
        path_components.append(f"{args.lora_rank}")
    
    if not short:
        path_components.append(f"pose_cfg-{args.pose_cfg}")
        path_components.append(f"mouth_cfg-{args.mouth_cfg}")
        path_components.append(f"pose_relax-{args.pose_relax}-f{args.pose_relax_num}-{mode}")

    experiment_name = "_".join(path_components)
    experiment_name = "{}-".format(args.exp_prefix) + experiment_name if args.exp_prefix else experiment_name

    full_path = os.path.join(args.output_path, experiment_name)
    print(f"Experiment path: {full_path}")
    os.makedirs(full_path, exist_ok=True)

    args.output_path = full_path
    return args


def print_args(args):
    print("=" * 80)
    print("CONFIGURATION PARAMETERS:")
    print("=" * 80)
    
    args_dict = vars(args)
    max_key_length = max(len(key) for key in args_dict.keys())
    
    for key in sorted(args_dict.keys()):
        value = args_dict[key]
        print(f"  {key.ljust(max_key_length)} : {value}")
    
    print("=" * 80)
    print(f"Total number of cfg parameters: {len(args_dict)}")
    print("=" * 80)


def save_args_to_yaml(args, output_path):
    """
    Save all hyperparameters to hparams.yaml file in the output directory.
    
    This follows PyTorch Lightning convention for hyperparameter tracking.
    Only saves from the main process in distributed training to avoid conflicts.
    
    Args:
        args: Parsed arguments from argparse (will be converted to dict)
        output_path: Directory where to save the hparams.yaml file
        
    Returns:
        str: Path to the saved hparams.yaml file, or None if not saved
    """
    # In distributed training, only save from rank 0 (main process)
    try:
        # Try to get the current process rank
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return None  # Skip saving from non-main processes
    except:
        # If distributed training is not set up, continue with saving
        pass
    
    # Check environment variable for local rank (common in distributed setups)
    local_rank = os.environ.get('LOCAL_RANK', '0')
    global_rank = os.environ.get('RANK', '0')
    if local_rank != '0' or global_rank != '0':
        return None  # Skip saving from non-main processes
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create output directory {output_path}: {e}")
        return None
    
    # Convert args to dictionary, filtering out non-serializable objects
    args_dict = vars(args) if hasattr(args, '__dict__') else dict(args)
    
    # Filter out problematic types for YAML serialization
    filtered_dict = {}
    for key, value in args_dict.items():
        if value is None:
            filtered_dict[key] = None
        elif isinstance(value, (str, int, float, bool, list, dict)):
            filtered_dict[key] = value
        elif isinstance(value, tuple):
            filtered_dict[key] = list(value)  # Convert tuple to list for YAML
        else:
            # Store string representation for non-serializable objects
            filtered_dict[key] = str(value)
    
    # Save to hparams.yaml with file locking to prevent conflicts
    hparams_path = os.path.join(output_path, 'hparams.yaml')
    lock_path = hparams_path + '.lock'
    
    try:
        # Simple file-based locking mechanism
        if os.path.exists(lock_path):
            # Another process is writing, skip
            return hparams_path
            
        # Create lock file
        with open(lock_path, 'w') as lock_file:
            lock_file.write(str(os.getpid()))
        
        # Save args to hparams.yaml
        with open(hparams_path, 'w', encoding='utf-8') as f:
            yaml.dump(filtered_dict, f, default_flow_style=False, sort_keys=True, indent=2, allow_unicode=True)
        print(f"[Hyperparameters] Saved hparams.yaml to: {hparams_path}")
        print(f"[Hyperparameters] Total parameters: {len(filtered_dict)}")
        
        # Remove lock file
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except:
                pass
        
        return hparams_path
            
    except Exception as e:
        print(f"Error: Failed to save hyperparameters to {hparams_path}: {e}")
        import traceback
        traceback.print_exc()
        # Clean up lock file if it exists
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except:
                pass
        return None
