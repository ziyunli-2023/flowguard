import os
import argparse
import copy
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from torchdiffeq import odeint
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel

# Configuration
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.n_epochs = 3
        self.lr = 2e-4
        self.sigma = 0.0
        self.model_type = "otcfm"  # "otcfm", "icfm", "fm", "si"
        self.savedir = "models/mnist"
        self.num_channels = 32
        self.num_res_blocks = 1
        self.eval_batch_size = 100
        self.n_steps_euler = 30
        self.n_steps_higher = 100
        
        # FID evaluation settings
        self.compute_fid = False
        self.fid_batch_size = 1024
        self.fid_num_samples = 50000

# Solvers
class Solvers:
    """Collection of different ODE solvers for flow matching"""
    
    @staticmethod
    def euler_solver(
        model: nn.Module,
        z0: torch.Tensor,
        n_steps: int = 30
    ) -> torch.Tensor:
        """
        Euler integration solver for ODE:
        x_{k+1} = x_k + f(x_k, t_k) * dt
        
        Args:
            model: Neural network model
            z0: Initial condition tensor
            n_steps: Number of integration steps
            
        Returns:
            Trajectory tensor of shape (n_steps+1, batch_size, channels, height, width)
        """
        device = z0.device
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z0
        traj = [x]
        
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0
            t_tensor = torch.full((x.shape[0],), t0, device=device)
            v = model(t_tensor, x)
            x = x + v * dt
            traj.append(x)
            
        return torch.stack(traj, dim=0)
    
    @staticmethod
    def rk4_solver(
        model: nn.Module,
        z0: torch.Tensor,
        n_steps: int = 30
    ) -> torch.Tensor:
        """
        Runge-Kutta 4th order solver for ODE
        
        Args:
            model: Neural network model
            z0: Initial condition tensor
            n_steps: Number of integration steps
            
        Returns:
            Trajectory tensor
        """
        device = z0.device
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        x = z0
        traj = [x]
        
        for t0, t1 in zip(ts[:-1], ts[1:]):
            dt = t1 - t0
            t_tensor = torch.full((x.shape[0],), t0, device=device)
            
            # RK4 steps
            k1 = model(t_tensor, x)
            k2 = model(t_tensor + dt/2, x + dt * k1 / 2)
            k3 = model(t_tensor + dt/2, x + dt * k2 / 2)
            k4 = model(t_tensor + dt, x + dt * k3)
            
            x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(x)
            
        return torch.stack(traj, dim=0)
    
    @staticmethod
    def dopri5_solver(
        model: nn.Module,
        z0: torch.Tensor,
        rtol: float = 1e-4,
        atol: float = 1e-4
    ) -> torch.Tensor:
        """
        Dormand-Prince 5th order adaptive solver using torchdiffeq
        
        Args:
            model: Neural network model
            z0: Initial condition tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Trajectory tensor
        """
        t_span = torch.linspace(0.0, 1.0, 2, device=z0.device)
        traj = odeint(model, z0, t_span, rtol=rtol, atol=atol, method='dopri5')
        return traj
    
    @staticmethod
    def torchdyn_solver(
        model: nn.Module,
        z0: torch.Tensor,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-4
    ) -> torch.Tensor:
        """
        TorchDyn solver wrapper
        
        Args:
            model: Neural network model
            z0: Initial condition tensor
            solver: Solver type ("dopri5", "euler", "tsit5", etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Trajectory tensor
        """
        node = NeuralODE(model, solver=solver, sensitivity="adjoint", atol=atol, rtol=rtol)
        t_span = torch.linspace(0.0, 1.0, 2, device=z0.device)
        traj = node.trajectory(z0, t_span=t_span)
        return traj

# Evaluation functions
class Evaluator:
    """Evaluation utilities for MNIST and CIFAR models"""
    
    @staticmethod
    def evaluate_mnist_quality(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate MNIST generation quality using various metrics
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run evaluation on
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Generate samples
        with torch.no_grad():
            z0 = torch.randn(n_samples, 1, 28, 28, device=device)
            traj = Solvers.euler_solver(model, z0, n_steps=50)
            generated_samples = traj[-1]
            
            # Get real samples
            real_samples = []
            for batch in test_loader:
                if len(real_samples) * batch[0].shape[0] >= n_samples:
                    break
                real_samples.append(batch[0].to(device))
            real_samples = torch.cat(real_samples, dim=0)[:n_samples]
        
        # Compute metrics
        metrics = {}
        
        # MSE between generated and real samples
        mse = torch.nn.functional.mse_loss(generated_samples, real_samples).item()
        metrics['mse'] = mse
        
        # L1 distance
        l1 = torch.mean(torch.abs(generated_samples - real_samples)).item()
        metrics['l1'] = l1
        
        # Perceptual similarity (simplified)
        # In practice, you might want to use a pre-trained network for this
        metrics['perceptual_similarity'] = mse  # Placeholder
        
        return metrics
    
    @staticmethod
    def compute_fid_cifar(
        model: nn.Module,
        device: torch.device,
        batch_size: int = 1024,
        num_samples: int = 50000,
        integration_steps: int = 100,
        integration_method: str = "dopri5"
    ) -> float:
        """
        Compute FID score for CIFAR-10 generation
        
        Args:
            model: Trained model
            device: Device to run evaluation on
            batch_size: Batch size for generation
            num_samples: Number of samples to generate
            integration_steps: Number of integration steps
            integration_method: Integration method ("euler", "dopri5", etc.)
            
        Returns:
            FID score
        """
        try:
            from cleanfid import fid
        except ImportError:
            print("cleanfid not available. Install with: pip install cleanfid")
            return -1.0
        
        model.eval()
        
        def generate_batch(unused_latent):
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 32, 32, device=device)
                
                if integration_method == "euler":
                    traj = Solvers.euler_solver(model, x, n_steps=integration_steps)
                elif integration_method == "dopri5":
                    traj = Solvers.dopri5_solver(model, x)
                else:
                    traj = Solvers.torchdyn_solver(model, x, solver=integration_method)
                
                # Convert to image format
                img = traj[-1]  # Take final step
                img = (img * 127.5 + 128).clip(0, 255).to(torch.uint8)
                return img
        
        print("Computing FID score...")
        score = fid.compute_fid(
            gen=generate_batch,
            dataset_name="cifar10",
            batch_size=batch_size,
            dataset_res=32,
            num_gen=num_samples,
            dataset_split="train",
            mode="legacy_tensorflow",
        )
        
        return score
    
    @staticmethod
    def visualize_samples(
        model: nn.Module,
        device: torch.device,
        n_samples: int = 64,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize generated samples
        
        Args:
            model: Trained model
            device: Device to run on
            n_samples: Number of samples to generate
            save_path: Path to save visualization
        """
        model.eval()
        
        with torch.no_grad():
            z0 = torch.randn(n_samples, 1, 28, 28, device=device)
            traj = Solvers.euler_solver(model, z0, n_steps=50)
            samples = traj[-1].clip(-1, 1)
            
            # Create grid
            grid = make_grid(
                samples.view([-1, 1, 28, 28]),
                value_range=(-1, 1),
                padding=0,
                nrow=8
            )
            
            # Convert to PIL and display
            img = ToPILImage()(grid)
            plt.figure(figsize=(12, 12))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title('Generated MNIST Samples')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"Saved visualization to {save_path}")
            
            plt.show()

# Training functions
class Trainer:
    """Training utilities for flow matching models"""
    
    @staticmethod
    def get_flow_matcher(model_type: str, sigma: float = 0.0):
        """Get flow matcher based on model type"""
        if model_type == "otcfm":
            return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif model_type == "icfm":
            return ConditionalFlowMatcher(sigma=sigma)
        elif model_type == "fm":
            return TargetConditionalFlowMatcher(sigma=sigma)
        elif model_type == "si":
            return VariancePreservingConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: Config,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """
        Train a flow matching model
        
        Args:
            model: Model to train
            train_loader: Training data loader
            config: Configuration object
            save_path: Path to save model
            
        Returns:
            Trained model
        """
        device = config.device
        model = model.to(device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        FM = Trainer.get_flow_matcher(config.model_type, config.sigma)
        
        # Training loop
        model.train()
        for epoch in range(config.n_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{config.n_epochs}"):
                optimizer.zero_grad()
                x1 = data[0].to(device)
                x0 = torch.randn_like(x1)
                
                # Sample from flow matcher
                result = FM.sample_location_and_conditional_flow(x0, x1)
                if len(result) == 3:
                    t, xt, ut = result
                else:
                    t, xt, ut, *_ = result
                
                vt = model(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        return model

# Data loading
def get_mnist_loaders(config: Config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get MNIST data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transform
    )
    
    testset = datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, test_loader

# Main function
def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="MNIST Flow Matching Training and Evaluation")
    parser.add_argument("--model_type", type=str, default="otcfm", 
                       choices=["otcfm", "icfm", "fm", "si"],
                       help="Type of flow matching model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--compute_fid", action="store_true", help="Compute FID score")
    parser.add_argument("--visualize", action="store_true", help="Visualize generated samples")
    parser.add_argument("--savedir", type=str, default="models/mnist", help="Directory to save model")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.model_type = args.model_type
    config.n_epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.savedir = args.savedir
    config.compute_fid = args.compute_fid
    
    print(f"Configuration: {vars(config)}")
    print(f"Device: {config.device}")
    
    # Create model
    model = UNetModel(
        dim=(1, 28, 28),
        num_channels=config.num_channels,
        num_res_blocks=config.num_res_blocks
    ).to(config.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train model
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    model_path = os.path.join(config.savedir, "model.pth")
    trained_model = Trainer.train_model(model, train_loader, config, model_path)
    
    # Evaluation
    if args.eval:
        print("\n" + "="*50)
        print("Starting Evaluation")
        print("="*50)
        
        # Load model if not already trained
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Skipping evaluation.")
            return
        
        trained_model.load_state_dict(torch.load(model_path, map_location=config.device))
        
        # Evaluate MNIST quality
        print("Evaluating MNIST generation quality...")
        metrics = Evaluator.evaluate_mnist_quality(trained_model, test_loader, config.device)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Compute FID if requested
        if args.compute_fid:
            print("\nComputing FID score...")
            fid_score = Evaluator.compute_fid_cifar(trained_model, config.device)
            print(f"FID Score: {fid_score:.4f}")
        
        # Visualize samples
        if args.visualize:
            print("\nGenerating sample visualization...")
            viz_path = os.path.join(config.savedir, "generated_samples.png")
            Evaluator.visualize_samples(trained_model, config.device, save_path=viz_path)
    
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main() 