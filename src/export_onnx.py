import torch
import torch.nn as nn
from models import MLP, get_model_for_endgame
import os
import json
import argparse

def export_model(model_path, output_onnx_path, metadata_path):
    # Load metadata
    if not os.path.exists(metadata_path):
        print(f"Error: metadata {metadata_path} not found")
        return
        
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    # Try to find input_size in different places in the JSON
    if 'dataset' in meta and 'input_size' in meta['dataset']:
        input_size = meta['dataset']['input_size']
        num_pieces = meta['dataset']['num_pieces']
    elif 'dimensions' in meta:
        input_size = meta['dimensions']
        config = meta.get('config', 'KQKvK')
        sides = config.lower().split('v')
        num_pieces = len(sides[0]) + len(sides[1])
    else:
        # Defaults for V4/V5 if not found
        num_pieces = meta.get('dataset', {}).get('num_pieces', 3)
        if num_pieces == 3:
            input_size = 45 # V4/V5
        else:
            input_size = 68 # V4/V5 for 4-piece
            
    num_wdl_classes = meta['args']['wdl_classes'] if 'args' in meta else 3
    model_type = meta['args']['model'] if 'args' in meta else 'mlp'
    
    print(f"Loading model with input_size={input_size}, num_pieces={num_pieces}")
    
    model = get_model_for_endgame(
        model_type=model_type,
        num_pieces=num_pieces,
        num_wdl_classes=num_wdl_classes,
        input_size=input_size
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['wdl_logits', 'dtz'],
        dynamic_axes={'input': {0: 'batch_size'}, 'wdl_logits': {0: 'batch_size'}, 'dtz': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--output", type=str, required=True, help="Path to .onnx output")
    parser.add_argument("--meta", type=str, required=True, help="Path to metadata .json")
    args = parser.parse_args()
    
    export_model(args.model, args.output, args.meta)
