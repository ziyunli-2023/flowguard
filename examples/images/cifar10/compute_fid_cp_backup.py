# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys
from datetime import datetime
from tkinter import N

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from tqdm import tqdm
import math
import numpy as np
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "examples/images/models/cifar", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps") # 100
flags.DEFINE_string("integration_method", "euler", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate") # 50000
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

def score_path(
    z0: torch.Tensor,
    solver: str = "euler",
    n_steps: int = 100,
    scoring_method: str = "local"
):

    if solver in ["euler", "rk4", "adaptive_euler"]:
        device = z0.device
        dt = 1.0 / n_steps
        t_k = 0.0
        x_k = z0
        v_k = new_net(torch.full((x_k.shape[0],), t_k, device=device), x_k)
        s_max = 0.0
        
        for i in range(n_steps):

            x_k1 = x_k + v_k * dt
            t_k1 = t_k + dt

            v_k1 = new_net(torch.full((x_k1.shape[0],), t_k1, device=device), x_k1)

            s_k = step_residual(v_k, v_k1, dt, t_k1).item()
            # print(f"i: {i}, s_k: {s_k}")
            s_max = max(s_max, s_k)

            x_k, v_k, t_k = x_k1, v_k1, t_k1
            
        # show image
        # img = (x_k * 127.5 + 128).clip(0, 255).to(torch.uint8)
        # plt.imshow(img[0].cpu().permute(1, 2, 0))
        # plt.show()
        print()
        print(f"s_max: {s_max}")
        return x_k, s_max


def calibrate(
    solver: str = "euler",
    n_cal: int = 2000,
    alpha: float = 0.1,
    n_steps: int = 30,
    scoring_method: str = "local",
    device: str = "cuda"
) -> float:

    scores = []
    for _ in tqdm(range(n_cal), desc=f"Calibrating ({scoring_method}, {solver})"):
        z0 = torch.randn(1, 3, 32, 32, device=device)  # CIFAR-10: 3 channels, 32x32
        _, s = score_path(z0, solver, n_steps, scoring_method)
        scores.append(s)
    
    k = math.ceil((1 - alpha) * (n_cal + 1))
    tau = np.partition(scores, k - 1)[k - 1]
    np.save(f"scores_cifar_{scoring_method}_{solver}.npy", scores)
    np.save(f"tau_cifar_{scoring_method}_{solver}.npy", tau)
    print(f"[Calibrate] α={alpha:.2f}  τ={tau:.5g} (method: {scoring_method}, solver: {solver})")
    return tau



def step_residual(
    vk: torch.Tensor,
    vk1: torch.Tensor,
    dt: float = 1.0,
    t_k: float = 0.0,
) -> torch.Tensor:
    """
    Compute score for each sample in the batch:
    s_k = || v_k - v_{k+1} ||_2^2 / dt^2
    Returns: tensor of shape [batch_size] with scores for each sample
    """
    # Calculate residual squared for each sample
    residual = (vk - vk1).pow(2)  # [batch_size, channels, height, width]
    # Average over spatial dimensions to get score for each sample
    score = residual.mean(dim=(1,2,3)) * ((1 - t_k)**2 / dt) # [batch_size]
    return score
       

def my_gen_1_img_filter(unused_latent):
    global total_samples_processed, total_samples_filtered
    
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        # dt = 1.0 / FLAGS.integration_steps
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
                x_1 = x_0 + v_0 * dt # go one step forward
                v_1_active = new_net(torch.full((active_mask.sum(),), dt * (step + 1), device=device), x_1[active_mask])   # compute the velocity at the new position

                # Create full v_1 tensor, only update active positions
                v_1 = v_0.clone()  # Start with previous velocity
                v_1[active_mask] = v_1_active  # use the previous velocity to update the active positions

                s_k = step_residual(v_0, v_1, dt, dt * (step + 1)) # if v1 is stable, will go the next step

                active_mask[s_k > tau] = False # current step's active mask
                print(f"Step {step}: {active_mask.sum().item()}/{x_0.shape[0]} samples active")
                print(f"mask: {active_mask}")

                x_0 = x_1
                v_0 = v_1
                
                # # Optional: print current number of active samples
                # if step % 10 == 0:
                #     active_count = active_mask.sum().item()
                #     print(f"Step {step}: {active_count}/{x_1.shape[0]} samples active")
            
            # Count filtered samples for this batch
            filtered_in_batch = initial_batch_size - active_mask.sum().item()
            total_samples_filtered += filtered_in_batch
            
            # Final result: use the last state
            traj = x_0[active_mask]
            
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

# tau = calibrate(solver="euler", n_cal=2000, alpha=0.1, n_steps=FLAGS.integration_steps, scoring_method="local", device=device)
tau = 0.087718
my_gen_1_img_filter(torch.randn(10, 3, 32, 32, device=device))

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
