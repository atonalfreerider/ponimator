"""
JSON Export Utilities for Ponimator

Exports interactive human-human motion data to structured JSON format.
"""

import json
import numpy as np
import torch
from typing import List, Dict, Any
import os


def numpy_to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def export_motion_to_json(
    output_path: str,
    betas: torch.Tensor,
    motion_pred: torch.Tensor,
    trans_pred: torch.Tensor,
    gender: torch.Tensor,
    inter_time_idx: int,
    metadata: Dict[str, Any] = None
):
    """
    Export interactive human-human motion to JSON format.
    
    Args:
        output_path: Path to save JSON file
        betas: SMPL-X shape parameters (P, 10) where P=2 persons
        motion_pred: Predicted motion parameters (P, T, D) where T=frames, D=dimensions
        trans_pred: Predicted translation (P, T, 3)
        gender: Gender for each person (P,) - 0=male, 1=female, 2=neutral
        inter_time_idx: Frame index of interactive pose
        metadata: Additional metadata (name, etc.)
    """
    
    if metadata is None:
        metadata = {}
    
    num_persons = motion_pred.shape[0]
    num_frames = motion_pred.shape[1]
    
    # Build the output structure
    output_data = {
        "metadata": {
            "sequence_name": metadata.get("name", "unknown"),
            "total_frames": num_frames,
            "num_persons": num_persons,
            "inter_time_idx": inter_time_idx,
            "coordinate_system": "camera",
            "units": "meters",
            "format_version": "1.0",
            "smplx_model": "full",
            "note": "Interactive human-human motion data. All coordinates in camera space."
        },
        "frames": {}
    }
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame_data = {
            "frame_number": frame_idx,
            "is_interactive_frame": frame_idx == inter_time_idx,
            "persons": []
        }
        
        # Process each person
        for person_idx in range(num_persons):
            # Extract motion parameters for this person and frame
            motion_params = motion_pred[person_idx, frame_idx]  # (D,)
            
            # Split motion into root_orient and body_pose
            # Assuming motion_params is in 6D rotation format
            # Convert back to axis-angle if needed
            root_orient = motion_params[:6]  # First 6D rotation
            body_pose = motion_params[6:]  # Remaining rotations
            
            person_data = {
                "person_id": person_idx,
                "smplx_parameters": {
                    "betas": numpy_to_serializable(betas[person_idx]),  # Shape (10,)
                    "root_orient": numpy_to_serializable(root_orient),  # (6,) or (3,)
                    "body_pose": numpy_to_serializable(body_pose),  # Remaining dimensions
                    "translation": numpy_to_serializable(trans_pred[person_idx, frame_idx]),  # (3,)
                    "gender": int(gender[person_idx]),  # 0=male, 1=female, 2=neutral
                }
            }
            
            frame_data["persons"].append(person_data)
        
        output_data["frames"][str(frame_idx)] = frame_data
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported motion to: {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Total frames: {num_frames}")
    print(f"   Number of persons: {num_persons}")
    print(f"   Interactive frame: {inter_time_idx}")
    print(f"   Coordinate system: CAMERA SPACE")
    
    return output_path
