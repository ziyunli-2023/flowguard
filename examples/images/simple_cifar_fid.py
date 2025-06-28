#!/usr/bin/env python3
"""
Simple CIFAR-10 FID test using real CIFAR-10 images as reference.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image
import argparse
import numpy as np
from cleanfid import fid
from tqdm import tqdm

# Import the model class
from flowguard_cifar import CIFARFlowGuardModel

def save_cifar_real_images(savedir="./cifar_real_images", n_images=1000):
    """Save real CIFAR-10 images for FID comparison."""
    os.makedirs(savedir, exist_ok=True)
    
    # Load CIFAR-10 without normalization (we want raw images)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    print(f"Saving {n_images} real CIFAR-10 images to {savedir}")
    
    for i in range(min(n_images, len(dataset))):
        img, _ = dataset[i]
        save_path = os.path.join(savedir, f"real_{i:06d}.png")
        save_image(img, save_path)
        
        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1}/{n_images} real images")
    
    print(f"✅ Saved {n_images} real CIFAR-10 images")

def generate_samples(model, n_samples=1000, savedir="./gen_images"):
    """Generate samples using the model and save them."""
    os.makedirs(savedir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    print(f"Generating {n_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating samples"):
            # Generate random noise
            z0 = torch.randn(1, 3, 32, 32, device=device)
            
            # Simple Euler integration
            x = z0
            for t in range(100):  # 100 steps
                t_tensor = torch.full((1,), t / 100.0, device=device)
                v = model(t_tensor, x)
                x = x + v * 0.01  # dt = 0.01
            
            # Convert from [-1,1] to [0,1] range
            x = (x * 0.5 + 0.5).clamp(0, 1)

    save_path = os.path.join(savedir, f"gen_{i:06d}.png")
    save_image(x, save_path)
    
    print(f"✅ Saved {n_samples} samples to {savedir}")

def compute_fid_with_cleanfid(gen_dir, real_dir):
    """Compute FID using cleanfid between two directories."""
    
    score = fid.compute_fid(
        fdir1=gen_dir,    # Generated images
        fdir2=real_dir    # Real CIFAR-10 images
    )
    print(f"✅ FID Score: {score:.3f}")
    return score
   

def main():
    parser = argparse.ArgumentParser(description="Simple CIFAR-10 FID Test")
    parser.add_argument("--model_path", type=str, 
                       default="examples/images/models/cifar/fm_cifar10_weights_step_400000.pt",
                       help="Path to pretrained model")
    parser.add_argument("--model_type", type=str, default="fm",
                       choices=["otcfm", "icfm", "fm", "si"],
                       help="Type of flow matching model")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--gen_dir", type=str, default="./gen_images",
                       help="Directory to save generated images")
    parser.add_argument("--real_dir", type=str, default="./cifar_real_images",
                       help="Directory to save real CIFAR-10 images")
    parser.add_argument("--n_real", type=int, default=1000,
                       help="Number of real images to use for comparison")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Step 1: Save real CIFAR-10 images
    print("\n" + "="*50)
    print("Step 1: Saving real CIFAR-10 images")
    print("="*50)
    save_cifar_real_images(args.real_dir, args.n_real)
    
    # Step 2: Load model and generate samples
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        return
    
    print("\n" + "="*50)
    print("Step 2: Loading model and generating samples")
    print("="*50)
    
    flowguard_model = CIFARFlowGuardModel(args.model_type, device)
    model = flowguard_model.load_model(args.model_path)
    
    generate_samples(model, args.n_samples, args.gen_dir)
    
    # Step 3: Compute FID
    print("\n" + "="*50)
    print("Step 3: Computing FID")
    print("="*50)
    
    if not os.path.exists(args.gen_dir):
        print(f"❌ Generated images directory not found: {args.gen_dir}")
        return
    
    if not os.path.exists(args.real_dir):
        print(f"❌ Real images directory not found: {args.real_dir}")
        return
    
    fid_score = compute_fid_with_cleanfid(args.gen_dir, args.real_dir)
    
    print("\n" + "="*50)
    print("Results")
    print("="*50)
    print(f"Model: {args.model_type}")
    print(f"Generated images: {args.gen_dir}")
    print(f"Real images: {args.real_dir}")
    print(f"FID Score: {fid_score:.3f}")
    print("="*50)

if __name__ == "__main__":
    main() 