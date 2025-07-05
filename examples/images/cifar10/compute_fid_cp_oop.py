# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "examples/images/models/cifar", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 2, help="number of inference steps") # 100
flags.DEFINE_string("integration_method", "euler", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50, help="number of samples to generate") # 50000
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 10, help="Batch size to compute FID") # 1024
flags.DEFINE_string("output_dir", "examples/images/cifar10/logs", help="output_directory")

FLAGS(sys.argv)

# Setup logging to both terminal and file
def setup_logging():
    """Setup logging to both terminal and log.txt file"""
    log_file = f"{FLAGS.output_dir}/log.txt"
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Create a custom print function that writes to both terminal and file
    def log_print(*args, **kwargs):
        # Print to terminal
        print(*args, **kwargs)
        
        # Write to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            # Convert all arguments to strings and join them
            message = ' '.join(str(arg) for arg in args)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            f.flush()  # Ensure immediate write
    
    return log_print

# Use the logging function
log_print = setup_logging()

# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)

# Load the model
PATH = f"{FLAGS.input_dir}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)
new_net.eval()

# Global variables to track filtering statistics
total_samples_processed = 0
total_samples_filtered = 0

tau = 0.02

class Point:
    """Represents a single sample point with its state"""
    def __init__(self, x, v, mask=True):
        self.x = x  # position
        self.v = v  # velocity
        self.mask = mask  # active mask
    
    def update_state(self, x_new, v_new):
        """Update position and velocity"""
        self.x = x_new
        self.v = v_new
    
    def set_inactive(self):
        """Mark this point as inactive"""
        self.mask = False
    
    def is_active(self):
        """Check if this point is active"""
        return self.mask

class PointBatch:
    """Manages a batch of points"""
    def __init__(self, x_batch, v_batch, device):
        self.device = device
        self.points = []
        for i in range(x_batch.shape[0]):
            self.points.append(Point(x_batch[i], v_batch[i], True))
    
    def get_active_x(self):
        """Get x values of active points"""
        active_x = []
        for point in self.points:
            if point.is_active():
                active_x.append(point.x)
        return torch.stack(active_x) if active_x else torch.empty(0, *self.points[0].x.shape, device=self.device)
    
    def get_active_v(self):
        """Get v values of active points"""
        active_v = []
        for point in self.points:
            if point.is_active():
                active_v.append(point.v)
        return torch.stack(active_v) if active_v else torch.empty(0, *self.points[0].v.shape, device=self.device)
    
    def get_all_x(self):
        """Get all x values as a batch tensor"""
        return torch.stack([point.x for point in self.points])
    
    def get_all_v(self):
        """Get all v values as a batch tensor"""
        return torch.stack([point.v for point in self.points])
    
    def update_active_points(self, x_new, v_new):
        """Update active points with new x and v values"""
        active_idx = 0
        for point in self.points:
            if point.is_active():
                point.update_state(x_new[active_idx], v_new[active_idx])
                active_idx += 1
    
    def filter_points(self, residuals, tau):
        """Filter points based on residuals"""
        active_idx = 0
        for point in self.points:
            if point.is_active():
                if residuals[active_idx] > tau:
                    point.set_inactive()
                active_idx += 1
    
    def count_active(self):
        """Count number of active points"""
        return sum(1 for point in self.points if point.is_active())
    
    def get_active_mask(self):
        """Get boolean mask of active points"""
        return torch.tensor([point.is_active() for point in self.points], device=self.device)

def step_residual(
    vk: torch.Tensor,
    vk1: torch.Tensor,
    dt: float = 1.0
) -> torch.Tensor:
    """
    Compute score for each sample in the batch:
    s_k = || v_k - v_{k+1} ||_2^2 / dt^2
    Returns: tensor of shape [batch_size] with scores for each sample
    """
    # Calculate residual squared for each sample
    residual = (vk - vk1).pow(2)  # [batch_size, channels, height, width]
    # Average over spatial dimensions to get score for each sample
    score = residual.mean(dim=(1,2,3)) / (dt ** 2)  # [batch_size]
    return score

def my_gen_1_img_filter_oop(unused_latent):
    global total_samples_processed, total_samples_filtered
    
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        t_0 = 0.0
        
        # Initialize point batch
        v_0 = new_net(torch.full((x.shape[0],), t_0, device=device), x)
        point_batch = PointBatch(x, v_0, device)
        
        # Track initial batch size
        initial_batch_size = x.shape[0]
        total_samples_processed += initial_batch_size

        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            dt = 1.0 / FLAGS.integration_steps
            
            for step in range(FLAGS.integration_steps):
                # Only compute for active samples
                if point_batch.count_active() == 0:
                    break  # Exit early if all samples are filtered out
                
                # Get current state
                x_0 = point_batch.get_all_x()
                v_0 = point_batch.get_all_v()
                active_mask = point_batch.get_active_mask()
                
                # Euler step: x_1 = x_0 + v_0 * dt
                x_1 = x_0 + v_0 * dt
                
                # Only compute velocity for active samples to save computation
                if active_mask.any():
                    x_1_active = x_1[active_mask]
                    v_1_active = new_net(torch.full((active_mask.sum(),), dt * (step + 1), device=device), x_1_active)
                    
                    # Create full v_1 tensor, only update active positions
                    v_1 = v_0.clone()
                    v_1[active_mask] = v_1_active
                    
                    # Calculate residual and filter
                    s_k = step_residual(v_0, v_1, dt)
                    point_batch.filter_points(s_k[active_mask], tau)
                    
                    # Update active points
                    point_batch.update_active_points(x_1_active, v_1_active)
                
                print(f"Step {step}: {point_batch.count_active()}/{initial_batch_size} samples active")
                print(f"mask: {point_batch.get_active_mask()}")
            
            # Count filtered samples for this batch
            filtered_in_batch = initial_batch_size - point_batch.count_active()
            total_samples_filtered += filtered_in_batch
            
            # Final result: use the last state
            traj = point_batch.get_all_x()
            
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
            traj = traj[-1, :]
    
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img

def my_gen_1_img_filter_original(unused_latent):
    """Original procedural version for comparison"""
    global total_samples_processed, total_samples_filtered
    
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        t_0 = 0.0
        x_0 = x
        v_0 = new_net(torch.full((x_0.shape[0],), t_0, device=device), x_0) 
        # Create mask to track which samples need to continue computation
        active_mask = torch.ones(x_0.shape[0], dtype=torch.bool, device=device)
        
        # Track initial batch size
        initial_batch_size = x_0.shape[0]
        total_samples_processed += initial_batch_size

        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            dt = 1.0 / FLAGS.integration_steps
            
            for step in range(FLAGS.integration_steps):
                # Only compute for active samples
                if not active_mask.any():
                    break  # Exit early if all samples are filtered out

                # Only compute for active samples to save computation
                x_1 = x_0 + v_0 * dt
                v_1_active = new_net(torch.full((active_mask.sum(),), dt * (step + 1), device=device), x_1[active_mask])   

                # Create full v_1 tensor, only update active positions
                v_1 = v_0.clone()  # Start with previous velocity
                v_1[active_mask] = v_1_active  # use the previous velocity to update the active positions

                s_k = step_residual(v_0, v_1, dt) # if v1 is stable, will go the next step

                active_mask[s_k > tau] = False # current step's active mask
                print(f"Step {step}: {active_mask.sum().item()}/{x_0.shape[0]} samples active")
                print(f"mask: {active_mask}")

                x_0 = x_1
                v_0 = v_1
            
            # Count filtered samples for this batch
            filtered_in_batch = initial_batch_size - active_mask.sum().item()
            total_samples_filtered += filtered_in_batch
            
            # Final result: use the last state
            traj = x_0
            
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
            traj = traj[-1, :]
    
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img

def compute_filter_percentage():
    """Compute and return the overall filter percentage"""
    if total_samples_processed == 0:
        return 0.0
    return (total_samples_filtered / total_samples_processed) * 100.0

# Test both versions
print("=== Testing Original Version ===")
my_gen_1_img_filter_original(torch.randn(10, 3, 32, 32, device=device))

print("\n=== Testing OOP Version ===")
my_gen_1_img_filter_oop(torch.randn(10, 3, 32, 32, device=device))

log_print("Start computing FID with filtering (OOP version)")
score = fid.compute_fid(
    gen=my_gen_1_img_filter_oop,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
)
log_print()
log_print("FID has been computed")
log_print(f"Model path: {PATH}")
log_print(f"{FLAGS.integration_steps} steps Euler FID with filtering (tau={tau}): {score:.4f}")

# Compute and display filter statistics
filter_percentage = compute_filter_percentage()
log_print(f"Filter statistics:")
log_print(f"  Total samples processed: {total_samples_processed}")
log_print(f"  Total samples filtered: {total_samples_filtered}")
log_print(f"  Filter percentage: {filter_percentage:.2f}%") 