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


# Define the integration method if euler is used
# if FLAGS.integration_method == "euler":
#     node = NeuralODE(new_net, solver=FLAGS.integration_method)


# def gen_1_img(unused_latent):
#     with torch.no_grad():
#         x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
#         if FLAGS.integration_method == "euler":
#             print("Use method: ", FLAGS.integration_method)
#             t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
#             traj = node.trajectory(x, t_span=t_span)
#         else:
#             print("Use method: ", FLAGS.integration_method)
#             t_span = torch.linspace(0, 1, 2, device=device)
#             traj = odeint(
#                 new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
#             )
#     traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
#     img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    
#     return img

# Global variables to track filtering statistics
total_samples_processed = 0
total_samples_filtered = 0

tau = 100

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
    score = residual.mean() / dt.pow(2)  # [batch_size]
    return score
       

def my_gen_1_img_filter(unused_latent):
    global total_samples_processed, total_samples_filtered
    
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        device = x.device
        dt = 1.0 / FLAGS.integration_steps
        t_k = 0.0
        x_k = x
        v_k = new_net(torch.full((x_k.shape[0],), t_k, device=device), x_k)
        
        # Track initial batch size
        initial_batch_size = x_k.shape[0]
        total_samples_processed += initial_batch_size
        
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            dt = 1.0 / FLAGS.integration_steps
            
            # Create mask to track which samples need to continue computation
            active_mask = torch.ones(x_k.shape[0], dtype=torch.bool, device=device)
            
            for step in range(FLAGS.integration_steps):
                # Only compute for active samples
                if not active_mask.any():
                    break  # Exit early if all samples are filtered out
                
                # Get indices of active samples
                active_indices = torch.where(active_mask)[0]
                
                # Only process active samples
                x_k_active = x_k[active_mask]
                v_k_active = v_k[active_mask]
                t_k_active = torch.full((x_k_active.shape[0],), t_k, device=device)
                
                # Euler one-step integration
                x_k1_active = x_k_active + v_k_active * dt
                t_k1 = t_k + dt
                
                v_k1_active = new_net(t_k_active, x_k1_active)
                
                s_k_active = step_residual(v_k_active, v_k1_active, dt)
                
                new_active_mask = s_k_active < tau
            
                active_mask[active_indices] = new_active_mask
                
                x_k[active_mask] = x_k1_active[new_active_mask]
                v_k[active_mask] = v_k1_active[new_active_mask]
                
                t_k = t_k1
                
                # Optional: print current number of active samples
                if step % 10 == 0:
                    active_count = active_mask.sum().item()
                    print(f"Step {step}: {active_count}/{x_k.shape[0]} samples active")
            
            # Count filtered samples for this batch
            filtered_in_batch = initial_batch_size - active_mask.sum().item()
            total_samples_filtered += filtered_in_batch
            
            # Final result: use the last state
            traj = x_k
            
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
            traj = traj[-1, :]
    
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return img


def my_gen_1_img(unused_latent):
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
    traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    
    return img


def compute_filter_percentage():
    """Compute and return the overall filter percentage"""
    if total_samples_processed == 0:
        return 0.0
    return (total_samples_filtered / total_samples_processed) * 100.0


log_print("Start computing FID with filtering")
score = fid.compute_fid(
    gen=my_gen_1_img_filter,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
)
log_print()
log_print("FID has been computed")
# print()
# print("Total NFE: ", new_net.nfe)
log_print(f"Model path: {PATH}")
# print(FLAGS.integration_steps + 1)
# print("euler FID: ", score)
log_print(f"{FLAGS.integration_steps} steps Euler FID with filtering (tau={tau}): {score:.4f}")

# Compute and display filter statistics
filter_percentage = compute_filter_percentage()
log_print(f"Filter statistics:")
log_print(f"  Total samples processed: {total_samples_processed}")
log_print(f"  Total samples filtered: {total_samples_filtered}")
log_print(f"  Filter percentage: {filter_percentage:.2f}%")
