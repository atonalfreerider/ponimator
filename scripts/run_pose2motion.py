import os
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import numpy as np
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C

from ponimator.utils.inference_utils import load_model
from ponimator.utils.utils import rotation_aa_to_6d, process_gender
from ponimator.utils.config_utils import get_config
from ponimator.datasets.wild_interpose import Buddi_Dataset
from ponimator.models import ContactMotionGen
from ponimator.utils.json_export import export_motion_to_json


def map_gender(gender_list):
    """Map gender strings to integers: male=0, female=1, neutral=2"""
    gender_map = {'male': 0, 'female': 1, 'neutral': 2}
    if gender_list is None:
        return None
    if len(gender_list) != 2:
        raise ValueError("Gender must be a list of exactly 2 strings")
    return torch.tensor([gender_map.get(g.lower(), 2) for g in gender_list])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--model_path", type=str, default="checkpoints/contactmotion.ckpt")
    parser.add_argument("--model_config_path", type=str, default="configs/motion.yaml")
    parser.add_argument("--body_model_root", type=str, default="body_models")
    parser.add_argument("--seq_len", type=int, default=30, help="sequence length (trained on 30)")
    # inference settings
    parser.add_argument("--data_source", choices=["buddi"], default="buddi", help="interactive pose data source")
    parser.add_argument("--data_dir", type=str, default="data/buddi/Couple_6806", help="data dir")
    parser.add_argument("--inter_time_idx", type=int, default=None, help="Index of the interactive pose range (0-29)")
    parser.add_argument("--gender", nargs="+", default=None, help="gender for two persons as list of strings (e.g., male female), where male=0, female=1, neutral=2")
    
    # utilization
    parser.add_argument("--save", action="store_true", help="save the results")
    parser.add_argument("--export_json", action="store_true", help="export results to JSON")
    parser.add_argument("--disable_vis", action="store_true", help="disable visualization")
    args = parser.parse_args()
    
    gender_arg = map_gender(args.gender)
    
    # Configure SMPL-X model path
    C.update_conf({'smplx_models': args.body_model_root})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx_layers = [
        SMPLLayer(model_type='smplx', gender='male', num_betas=10, device=device),
        SMPLLayer(model_type='smplx', gender='female', num_betas=10, device=device),
        SMPLLayer(model_type='smplx', gender='neutral', num_betas=10, device=device)
    ]

    model_cfg = get_config(args.model_config_path) 
    model = ContactMotionGen(model_cfg)
    load_model(args.model_path, model)
    model.eval()
    model.to(device)
    
    if args.data_source == "buddi":
        dataset = Buddi_Dataset(args.data_dir, smplx_layer=smplx_layers[2], device=device)
    else:
        raise NotImplementedError

    for sample in dataset:
        
        motions = torch.cat([sample['root_orient'].reshape(2, -1, 3), sample['pose'].reshape(2, -1, 3)], dim=1)
        
        inter_pose = rotation_aa_to_6d(motions.reshape(-1, 3)).reshape(2, -1, 6)
        inter_pose = inter_pose.reshape(2, -1)
        inter_pose = inter_pose[None, :, None, :].to(device)
        inter_trans = sample['trans'][None, :, None, :].to(device)
            
        if "gender" in sample:
            gender = sample['gender']
        elif gender_arg is not None:
            gender = gender_arg
        else:
            gender = torch.tensor([2, 2])

        if args.inter_time_idx is None:
            args.inter_time_idx = args.seq_len // 2
            
        _, inter_joints = process_gender(smplx_layers, inter_pose, inter_trans, sample['betas'].unsqueeze(0).to(device), gender.unsqueeze(0).to(device))
        inter_joints = inter_joints[:, :, :, :22]

        with torch.no_grad():
            sample_output = model.decoder.forward_inference(inter_pose, inter_trans, inter_joints, args.seq_len, mid_index=args.inter_time_idx)
        output = sample_output["output"].squeeze(0).cpu()
        
        trans_pred = output[:, :, -3:]
        motion_pred = output[:, :, :-3]
        name = sample['name']
        
        save_dir = os.path.join(args.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        if args.save:
            output_results = {
                "betas": sample['betas'].cpu(),
                "motion_pred": motion_pred.cpu(),
                "trans_pred": trans_pred.cpu(),
                "gender": gender.cpu(), 
                "inter_time_idx": args.inter_time_idx,
            }

            save_path = os.path.join(save_dir, f"motion_pred.pkl")
            with open(save_path, "wb") as f:
                torch.save(output_results, f)
            print(f"Saved results to {save_path}")
        
        if args.export_json:
            json_path = os.path.join(save_dir, "poses.json")
            export_motion_to_json(
                output_path=json_path,
                betas=sample['betas'].cpu(),
                motion_pred=motion_pred.cpu(),
                trans_pred=trans_pred.cpu(),
                gender=gender.cpu(),
                inter_time_idx=args.inter_time_idx,
                metadata={"name": name}
            )

    print("Done!")