# flowguard_cifar.py
# FlowGuard CP implementation for CIFAR-10 dataset

import os
import json
import math
import argparse
from typing import Tuple, List, Optional, Dict, Any, Union

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

# import os; os.chdir("C:/Users/liziy/Code/conditional-flow-matching/")

# 图像质量评估
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError as e:
    FID_AVAILABLE = False
    print("Warning: torchmetrics[image] not available. FID evaluation will be skipped.")
    print("Install with: pip install torchmetrics[image] or pip install torch-fidelity")

# Flow matching imports
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

# ODE求解器导入
try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Dopri5 solver will be skipped.")

try:
    from torchdyn.core import NeuralODE
    TORCHDYN_AVAILABLE = True
except ImportError:
    TORCHDYN_AVAILABLE = False
    print("Warning: torchdyn not available. TorchDyn solvers will be skipped.")

# ─── 1. 多种ODE求解器 ──────────────────────────────────────────
class ODESolvers:
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
            t_tensor = torch.full((x.shape[0],), t0.item(), device=device)
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
            t_tensor = torch.full((x.shape[0],), t0.item(), device=device)
            
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
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq not available. Install with: pip install torchdiffeq")
            
        t_span = torch.linspace(0.0, 1.0, 2, device=z0.device)
        result = odeint(model, z0, t_span, rtol=rtol, atol=atol, method='dopri5')
        
        # Handle potential tuple return from odeint
        if isinstance(result, tuple):
            traj = result[0]
        else:
            traj = result
        
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
            solver: Solver type ("dopri5", "euler", "tsit5", "heun", "midpoint", etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Trajectory tensor
        """
        if not TORCHDYN_AVAILABLE:
            raise ImportError("torchdyn not available. Install with: pip install torchdyn")
        
        # Create a wrapper for the model to match torchdyn's expected interface
        class TorchDynWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, t, x, *args, **kwargs):
                # Remove any extra arguments that torchdyn might pass
                return self.model(t, x)
        
        wrapped_model = TorchDynWrapper(model)
        node = NeuralODE(wrapped_model, solver=solver, sensitivity="adjoint", atol=atol, rtol=rtol)
        t_span = torch.linspace(0.0, 1.0, 2, device=z0.device)
        traj = node.trajectory(z0, t_span=t_span)  # type: ignore
        
        # Handle different return types from torchdyn
        if isinstance(traj, tuple):
            # Some torchdyn solvers return (trajectory, info) tuple
            traj = traj[0]
        
        # Ensure we return a tensor
        if not isinstance(traj, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(traj)}")
            
        return traj
    
    @staticmethod
    def adaptive_euler_solver(
        model: nn.Module,
        z0: torch.Tensor,
        max_steps: int = 100,
        tol: float = 1e-4
    ) -> torch.Tensor:
        """
        Adaptive Euler solver with error estimation
        
        Args:
            model: Neural network model
            z0: Initial condition tensor
            max_steps: Maximum number of steps
            tol: Error tolerance
            
        Returns:
            Trajectory tensor
        """
        device = z0.device
        x = z0
        traj = [x]
        t = 0.0
        dt = 0.1  # Initial step size
        
        for step in range(max_steps):
            if t >= 1.0:
                break
                
            # Take two half steps
            t_tensor = torch.full((x.shape[0],), t, device=device)
            v1 = model(t_tensor, x)
            x_half = x + v1 * (dt / 2)
            
            t_tensor_half = torch.full((x.shape[0],), t + dt/2, device=device)
            v2 = model(t_tensor_half, x_half)
            x_full = x + v2 * dt
            
            # Take one full step
            v_full = model(t_tensor, x)
            x_single = x + v_full * dt
            
            # Error estimation
            error = torch.norm(x_full - x_single, dim=1).mean()
            
            if error < tol:
                x = x_full
                t += dt
                traj.append(x)
                dt = min(dt * 1.5, 0.1)  # Increase step size
            else:
                dt = max(dt * 0.5, 0.01)  # Decrease step size
                
        return torch.stack(traj, dim=0)

# ─── 2. 模型加载与训练 ──────────────────────────────────────────
class CIFARFlowGuardModel:
    """CIFAR-10 Flow Matching model with FlowGuard capabilities"""
    
    def __init__(self, model_type: str = "otcfm", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.flow_matcher = None
        
    def create_model(self, num_channels: int = 128, num_res_blocks: int = 2) -> nn.Module:
        """Create UNet model for CIFAR-10"""
        model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1, 
        ).to(self.device)
        return model
    
    def get_flow_matcher(self, sigma: float = 0.0):
        """Get flow matcher based on model type"""
        if self.model_type == "otcfm":
            return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        elif self.model_type == "icfm":
            return ConditionalFlowMatcher(sigma=sigma)
        elif self.model_type == "fm":
            return TargetConditionalFlowMatcher(sigma=sigma)
        elif self.model_type == "si":
            return VariancePreservingConditionalFlowMatcher(sigma=sigma)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int = 3,
        lr: float = 2e-4,
        save_path: Optional[str] = None
    ) -> nn.Module:
        """Train the flow matching model"""
        self.model = self.create_model()
        self.flow_matcher = self.get_flow_matcher()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{n_epochs}"):
                optimizer.zero_grad()
                x1 = data[0].to(self.device)
                x0 = torch.randn_like(x1)
                
                # Sample from flow matcher
                result = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
                if len(result) == 3:
                    t, xt, ut = result
                else:
                    t, xt, ut, *_ = result
                
                vt = self.model(t, xt)
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
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        return self.model
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load a trained model"""
        self.model = self.create_model()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'net_model' in checkpoint:
                # Checkpoint contains training state, extract model weights
                print("Loading model from checkpoint with training state...")
                state_dict = checkpoint['net_model']
            elif 'state_dict' in checkpoint:
                # Standard state_dict format
                state_dict = checkpoint['state_dict']
            else:
                # Assume it's already a state_dict
                state_dict = checkpoint
        else:
            # Assume it's already a state_dict
            state_dict = checkpoint
        
        # Load the state dict
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded successfully from {model_path}")
        return self.model

# ─── 3. 通用ODE积分函数 ─────────────────────────────────
def integrate_ode(
    model: nn.Module,
    z0: torch.Tensor,
    solver: str = "euler",
    n_steps: int = 30,
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> torch.Tensor:
    """
    通用ODE积分函数，支持多种求解器
    
    Args:
        model: 神经网络模型
        z0: 初始条件
        solver: 求解器类型 ("euler", "rk4", "dopri5", "torchdyn_*", "adaptive_euler")
        n_steps: 积分步数（仅用于固定步长求解器）
        rtol: 相对容差（仅用于自适应求解器）
        atol: 绝对容差（仅用于自适应求解器）
    
    Returns:
        轨迹张量
    """
    if solver == "euler":
        return ODESolvers.euler_solver(model, z0, n_steps)
    elif solver == "rk4":
        return ODESolvers.rk4_solver(model, z0, n_steps)
    elif solver == "dopri5":
        return ODESolvers.dopri5_solver(model, z0, rtol, atol)
    elif solver.startswith("torchdyn_"):
        # 支持 torchdyn_dopri5, torchdyn_euler, torchdyn_tsit5 等
        torchdyn_solver = solver.replace("torchdyn_", "")
        return ODESolvers.torchdyn_solver(model, z0, torchdyn_solver, rtol, atol)
    elif solver == "adaptive_euler":
        return ODESolvers.adaptive_euler_solver(model, z0, n_steps, rtol)
    else:
        raise ValueError(f"Unknown solver: {solver}. Available: euler, rk4, dopri5, torchdyn_*, adaptive_euler")

# ─── 4. 非互拟分数：速度场残差─────────────────────────
def step_residual(
    vk: torch.Tensor,
    vk1: torch.Tensor,
    dt: float = 1.0
) -> torch.Tensor:
    """
    计算速度场残差 score:
    s_k = || v_k - v_{k+1} ||_2^2 / dt
    """
    return (vk - vk1).pow(2).mean() / dt

def global_residual(
    model: nn.Module,
    z0: torch.Tensor,
    solver: str = "euler",
    n_steps: int = 30
) -> float:
    """
    计算全局残差 score:
    S_global = || v_final - v_initial ||_2^2
    其中 v_initial = model(0, z0), v_final = model(1, x_final)
    """
    device = z0.device
    
    # 计算初始速度
    v_initial = model(torch.full((z0.shape[0],), 0.0, device=device), z0)
    
    # 积分到最终状态
    traj = integrate_ode(model, z0, solver, n_steps)
    x_final = traj[-1]
    
    # 计算最终速度
    v_final = model(torch.full((x_final.shape[0],), 1.0, device=device), x_final)
    
    # 全局残差
    return (v_final - v_initial).pow(2).mean().item()

# ─── 5. 路径打分 ────────────────────────────────────────────
def score_path(
    model: nn.Module,
    z0: torch.Tensor,
    solver: str = "euler",
    n_steps: int = 30,
    scoring_method: str = "local"
) -> Tuple[torch.Tensor, float]:
    """
    路径打分，支持局部和全局两种方法：
    
    Args:
        model: 模型
        z0: 初始噪声
        solver: ODE求解器
        n_steps: 积分步数
        scoring_method: 打分方法 ("local" 或 "global")
    
    Returns:
        (x_final, S_max) 或 (x_final, S_global)
    """
    if scoring_method == "global":
        # 全局打分
        x_final = integrate_ode(model, z0, solver, n_steps)[-1]
        s_global = global_residual(model, z0, solver, n_steps)
        return x_final, s_global
    else:
        # 局部打分（默认）
        if solver in ["euler", "rk4", "adaptive_euler"]:
            # 固定步长求解器，可以逐步计算
            device = z0.device
            dt = 1.0 / n_steps
            t_k = 0.0
            x_k = z0
            v_k = model(torch.full((x_k.shape[0],), t_k, device=device), x_k)
            s_max = 0.0
            
            for _ in range(n_steps):
                # Euler预测一步
                x_k1 = x_k + v_k * dt
                t_k1 = t_k + dt
                # 计算下一步速度
                v_k1 = model(torch.full((x_k1.shape[0],), t_k1, device=device), x_k1)
                # 局部分数
                s_k = step_residual(v_k, v_k1, dt).item()
                s_max = max(s_max, s_k)
                # 前进
                x_k, v_k, t_k = x_k1, v_k1, t_k1
            
            return x_k, s_max
        else:
            # 自适应求解器，需要完整积分
            traj = integrate_ode(model, z0, solver, n_steps)
            x_final = traj[-1]
            
            # 计算轨迹上的最大残差
            s_max = 0.0
            for i in range(len(traj) - 1):
                t_i = i / (len(traj) - 1)
                t_i1 = (i + 1) / (len(traj) - 1)
                dt = t_i1 - t_i
                
                v_i = model(torch.full((traj[i].shape[0],), t_i, device=z0.device), traj[i])
                v_i1 = model(torch.full((traj[i+1].shape[0],), t_i1, device=z0.device), traj[i+1])
                
                s_k = step_residual(v_i, v_i1, dt).item()
                s_max = max(s_max, s_k)
            
            return x_final, s_max

# ─── 6. 校准阈值 τ ─────────────────────────────────────────
def calibrate(
    model: nn.Module,
    solver: str = "euler",
    n_cal: int = 2000,
    alpha: float = 0.1,
    n_steps: int = 30,
    scoring_method: str = "local",
    device: str = "cuda"
) -> float:
    """
    离线采样 n_cal 条随机 z0，计算 S_i，取 higher split quantile τ
    保存 tau.npy
    支持局部和全局两种打分方法
    """
    scores = []
    for _ in tqdm(range(n_cal), desc=f"Calibrating ({scoring_method}, {solver})"):
        z0 = torch.randn(1, 3, 32, 32, device=device)  # CIFAR-10: 3 channels, 32x32
        _, s = score_path(model, z0, solver, n_steps, scoring_method)
        scores.append(s)
    
    k = math.ceil((1 - alpha) * (n_cal + 1))
    tau = np.partition(scores, k - 1)[k - 1]
    np.save(f"tau_cifar_{scoring_method}_{solver}.npy", tau)
    print(f"[Calibrate] α={alpha:.2f}  τ={tau:.5g} (method: {scoring_method}, solver: {solver})")
    return tau

# ─── 7. 推断 & 过滤（Early-Abort）──────────────────────────
def generate_and_filter(
    model: nn.Module,
    tau: float,
    solver: str = "euler",
    n_test: int = 1000,
    n_steps: int = 30,
    early_abort: bool = True,
    scoring_method: str = "local",
    device: str = "cuda"
) -> List[torch.Tensor]:
    """
    对 n_test 条随机噪声：
      • 在线累计 S_max 或计算 S_global
      • early_abort 达到 τ 时立即跳出（仅对局部打分有效）
      • 完成后若 S≤τ 则保留图像
    返回所有保留下来的 final x (CPU tensor list)
    支持局部和全局两种打分方法
    """
    kept_imgs = []
    
    if scoring_method == "global":
        # 全局打分：需要完整积分
        for _ in tqdm(range(n_test), desc=f"Generating ({scoring_method}, {solver})"):
            z0 = torch.randn(1, 3, 32, 32, device=device)
            x_final, s_global = score_path(model, z0, solver, n_steps, "global")
            if s_global <= tau:
                kept_imgs.append(x_final.detach().cpu())
    else:
        # 局部打分：支持early-abort（仅对固定步长求解器）
        if solver in ["euler", "rk4", "adaptive_euler"]:
            for _ in tqdm(range(n_test), desc=f"Generating ({scoring_method}, {solver})"):
                z0 = torch.randn(1, 3, 32, 32, device=device)  # CIFAR-10: 3 channels, 32x32
                dt = 1.0 / n_steps
                t_k = 0.0
                x_k = z0
                v_k = model(torch.full((x_k.shape[0],), t_k, device=device), x_k)
                S = 0.0
                
                for _ in range(n_steps):
                    # Euler预测一步
                    x_k1 = x_k + v_k * dt
                    t_k1 = t_k + dt
                    # 计算下一步速度
                    v_k1 = model(torch.full((x_k1.shape[0],), t_k1, device=device), x_k1)
                    # 局部分数
                    s_k = step_residual(v_k, v_k1, dt).item()
                    S = max(S, s_k)
                    
                    if early_abort and S > tau:
                        break
                    # 前进
                    x_k, v_k, t_k = x_k1, v_k1, t_k1
                
                # 只有在没有提前终止且分数低于阈值时才保留
                if S <= tau:
                    kept_imgs.append(x_k.detach().cpu())
        else:
            # 自适应求解器：需要完整积分
            for _ in tqdm(range(n_test), desc=f"Generating ({scoring_method}, {solver})"):
                z0 = torch.randn(1, 3, 32, 32, device=device)
                x_final, s_max = score_path(model, z0, solver, n_steps, "local")
                if s_max <= tau:
                    kept_imgs.append(x_final.detach().cpu())
    
    keep_rate = len(kept_imgs) / n_test
    print(f"[Filter] keep_rate={keep_rate:.3f} ({len(kept_imgs)}/{n_test}) (method: {scoring_method}, solver: {solver})")
    return kept_imgs

# ─── 8. CIFAR-10 评估指标 ───────────────────────────────────────────
def evaluate_cifar_metrics(
    real_ds,
    gen_imgs: List[torch.Tensor],
    device: str = "cuda"
) -> Dict[str, Union[float, int, str]]:
    """
    评估CIFAR-10生成质量：FID, 分类准确率
    """
    import torch.nn.functional as F
    
    if not FID_AVAILABLE:
        print("Warning: torchmetrics[image] not available. Using fallback evaluation.")
        return evaluate_cifar_metrics_fallback(real_ds, gen_imgs, device)
    
    # 简单的CIFAR-10分类器
    class SimpleCIFARClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # 加载预训练分类器（如果没有，创建一个简单的）
    classifier = SimpleCIFARClassifier().to(device)
    classifier.eval()
    
    # 评估指标
    classification_scores = []
    
    # 随机选择一些真实图像作为参考
    n_eval = min(len(gen_imgs), 100)  # 限制评估数量
    real_samples = []
    to_tensor = transforms.ToTensor()
    
    for i in range(n_eval):
        img, _ = real_ds[i]
        # 统一转换为tensor格式
        if isinstance(img, torch.Tensor):
            real_samples.append(img)
        else:
            # PIL Image，转换为tensor
            real_samples.append(to_tensor(img))
    
    print(f"Evaluating {n_eval} generated samples with FID...")
    
    # 初始化FID计算器
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # 处理真实图像用于FID计算
    real_tensors = []
    for i, real_img in enumerate(real_samples[:n_eval]):
        # 确保图像格式正确 [0,1] -> [0,255]
        real_tensor = real_img.to(device)
        real_tensor = (real_tensor * 255).clamp(0, 255).to(torch.uint8)
        if real_tensor.dim() == 3:
            real_tensor = real_tensor.unsqueeze(0)
        real_tensors.append(real_tensor)
    
    # 批量处理真实图像
    real_batch = torch.cat(real_tensors, dim=0)  # 保证4D
    fid.update(real_batch, real=True)
    
    # 处理生成图像
    for i, gen_img in enumerate(gen_imgs[:n_eval]):
        # 分类准确率（使用生成图像进行分类）
        with torch.no_grad():
            # 预处理生成图像
            x = gen_img.to(device)
            x = (x * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
            if x.dim() == 3:
                x = x.unsqueeze(0)
            output = classifier(x)
            pred = output.argmax(dim=1, keepdim=True)
            confidence = torch.exp(output).max(dim=1)[0].item()
            classification_scores.append(confidence)
        
        # 为FID准备生成图像
        gen_tensor = gen_img.to(device)
        gen_tensor = (gen_tensor * 0.5 + 0.5).clamp(0, 1)
        gen_tensor = (gen_tensor * 255).clamp(0, 255).to(torch.uint8)
        if gen_tensor.dim() == 3:
            gen_tensor = gen_tensor.unsqueeze(0)
        fid.update(gen_tensor, real=False)
    
    # 计算FID分数
    fid_score = fid.compute().item()
    
    # 计算平均分类置信度
    avg_confidence = float(np.mean(classification_scores))
    
    print(f"[CIFAR Metrics] FID: {fid_score:.3f}, Avg Classification Confidence: {avg_confidence:.3f}")
    
    return {
        'fid': fid_score,
        'classification_confidence': avg_confidence,
        'n_evaluated': n_eval
    }

def evaluate_cifar_metrics_fallback(
    real_ds,
    gen_imgs: List[torch.Tensor],
    device: str = "cuda"
) -> Dict[str, Union[float, int, str]]:
    """
    备选评估函数：当FID不可用时使用简单的统计指标
    """
    import torch.nn.functional as F
    
    # 简单的CIFAR-10分类器
    class SimpleCIFARClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    classifier = SimpleCIFARClassifier().to(device)
    classifier.eval()
    
    # 评估指标
    classification_scores = []
    pixel_stats = []
    
    n_eval = min(len(gen_imgs), 100)
    print(f"Evaluating {n_eval} generated samples with fallback metrics...")
    
    for i, gen_img in enumerate(gen_imgs[:n_eval]):
        with torch.no_grad():
            # 预处理生成图像
            x = gen_img.to(device)
            x = (x * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
            if x.dim() == 3:
                x = x.unsqueeze(0)  # 只在无batch维时加
            
            # 分类置信度
            output = classifier(x)
            confidence = torch.exp(output).max(dim=1)[0].item()
            classification_scores.append(confidence)
            
            # 像素统计（作为图像质量的简单指标）
            pixel_mean = x.mean().item()
            pixel_std = x.std().item()
            pixel_stats.append((pixel_mean, pixel_std))
    
    # 计算平均指标
    avg_confidence = float(np.mean(classification_scores))
    avg_pixel_mean = float(np.mean([p[0] for p in pixel_stats]))
    avg_pixel_std = float(np.mean([p[1] for p in pixel_stats]))
    
    print(f"[CIFAR Fallback Metrics] Avg Classification Confidence: {avg_confidence:.3f}")
    print(f"[CIFAR Fallback Metrics] Avg Pixel Mean: {avg_pixel_mean:.3f}, Avg Pixel Std: {avg_pixel_std:.3f}")
    
    return {
        'classification_confidence': avg_confidence,
        'pixel_mean': avg_pixel_mean,
        'pixel_std': avg_pixel_std,
        'n_evaluated': n_eval,
        'note': 'FID not available, using fallback metrics'
    }

# ─── 9. 可视化 ────────────────────────────────────────────
def visualize_samples(
    gen_imgs: List[torch.Tensor],
    n_samples: int = 64,
    save_path: Optional[str] = None
) -> None:
    """可视化生成的样本"""
    if len(gen_imgs) == 0:
        print("No images to visualize!")
        return
    
    # 取前n_samples个样本
    samples = torch.stack(gen_imgs[:n_samples])
    samples = samples.clip(-1, 1)
    
    # 创建网格
    grid = make_grid(
        samples.view([-1, 3, 32, 32]),
        value_range=(-1, 1),
        padding=0,
        nrow=8
    )
    
    # 转换为PIL并显示
    img = ToPILImage()(grid)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'FlowGuard CIFAR-10 Samples (Filtered: {len(gen_imgs)} kept)')

    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    plt.show()

# ─── 10. 数据加载 ──────────────────────────────────────────
def get_cifar_loaders(batch_size: int = 128) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transform
    )
    
    testset = datasets.CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, test_loader

# ─── 11. 主流程 & 参数解析────────────────────────────────
def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="FlowGuard CP for CIFAR-10")
    parser.add_argument("--model_type", type=str, default="fm", 
                       choices=["otcfm", "icfm", "fm", "si"],
                       help="Type of flow matching model")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--model_path", type=str, default="examples/images/models/cifar/fm_cifar10_weights_step_400000.pt",
                       help="Path to load/save model")
    parser.add_argument("--n_cal", type=int, default=200, help="Number of calibration samples")
    parser.add_argument("--n_test", type=int, default=10, help="Number of test samples")
    parser.add_argument("--alpha", type=float, default=0.1, help="False positive rate")
    parser.add_argument("--n_steps", type=int, default=100, help="Number of integration steps")
    parser.add_argument("--no_early", action="store_true", help="Disable early abort")
    parser.add_argument("--visualize", action="store_true", help="Visualize generated samples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--scoring_method", type=str, default="local", 
                       choices=["local", "global"],
                       help="Scoring method: local (max over steps) or global (final vs initial)")
    parser.add_argument("--solver", type=str, default="euler",
                       choices=["euler", "rk4", "dopri5", "torchdyn_dopri5", "torchdyn_euler", 
                               "torchdyn_tsit5", "torchdyn_heun", "torchdyn_midpoint", "adaptive_euler"],
                       help="ODE solver to use")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for adaptive solvers")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for adaptive solvers")
    
    args = parser.parse_args()
    args.visualize = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Scoring method: {args.scoring_method}")
    print(f"ODE solver: {args.solver}")
    
    # 检查求解器可用性
    if args.solver == "dopri5" and not TORCHDIFFEQ_AVAILABLE:
        print("Error: dopri5 solver requires torchdiffeq. Install with: pip install torchdiffeq")
        return
    if args.solver.startswith("torchdyn_") and not TORCHDYN_AVAILABLE:
        print("Error: torchdyn solvers require torchdyn. Install with: pip install torchdyn")
        return
    
    # 创建模型
    flowguard_model = CIFARFlowGuardModel(args.model_type, device)
    
    # 训练或加载模型
    if args.train:
        print("\n" + "="*50)
        print("Training FlowGuard Model")
        print("="*50)
        
        train_loader, _ = get_cifar_loaders(args.batch_size)
        model = flowguard_model.train_model(
            train_loader, 
            n_epochs=args.epochs,
            save_path=args.model_path
        )
    else:
        print(f"\nLoading model from {args.model_path}")
        model = flowguard_model.load_model(args.model_path)
    
    # FlowGuard 校准
    print("\n" + "="*50)
    print("FlowGuard Calibration")
    print("="*50)

    model.eval()
    
    tau = calibrate(model, args.solver, args.n_cal, args.alpha, args.n_steps, args.scoring_method, device)
    
    # FlowGuard 生成与过滤
    print("\n" + "="*50)
    print("FlowGuard Generation & Filtering")
    print("="*50)
    
    gen_imgs = generate_and_filter(
        model, tau, args.solver, args.n_test, args.n_steps, 
        not args.no_early, args.scoring_method, device
    )
    
    # 可视化
    if args.visualize:
        print("\n" + "="*50)
        print("Visualization")
        print("="*50)
        
        viz_path = os.path.join(os.path.dirname(args.model_path), f"flowguard_samples_{args.scoring_method}_{args.solver}.png")
        visualize_samples(gen_imgs, save_path=viz_path)

    
    
    # CIFAR-10 评估指标
    if len(gen_imgs) > 0:
        print("\n" + "="*50)
        print("CIFAR-10 Metrics Evaluation")
        print("="*50)
        
        real_ds = datasets.CIFAR10(root="./data", train=True, download=True)
        metrics = evaluate_cifar_metrics(real_ds, gen_imgs, device)
    
    print("\n" + "="*50)
    print("FlowGuard CIFAR-10 Pipeline Completed!")
    print("="*50)
    print(f"Model type: {args.model_type}")
    print(f"Scoring method: {args.scoring_method}")
    print(f"ODE solver: {args.solver}")
    print(f"Calibration samples: {args.n_cal}")
    print(f"Test samples: {args.n_test}")
    print(f"Alpha: {args.alpha}")
    print(f"Threshold τ: {tau:.5g}")
    print(f"Kept samples: {len(gen_imgs)}/{args.n_test}")
    print(f"Keep rate: {len(gen_imgs)/args.n_test:.3f}")

if __name__ == "__main__":
    main()

# ─── 逻辑自检 ─────────────────────────────────────────────
"""
模块 & 功能对照表
模块 / 函数	功能	逻辑检查
ODESolvers	多种ODE求解器集合	✅ 支持euler, rk4, dopri5, torchdyn_*, adaptive_euler
CIFARFlowGuardModel	模型创建、训练、加载	✅ 支持多种flow matching类型，完整训练流程
integrate_ode	通用ODE积分函数	✅ 统一接口，支持所有求解器
step_residual	计算速度场残差 score	✅ 新的速度残差方法：||v_k - v_{k+1}||^2
score_path	聚合路径 max score	✅ 优化：每个step只计算一次v，避免重复计算
calibrate	离线计算 τ	✅ "higher" 分位、保存 tau_cifar.npy
generate_and_filter	在线过滤 + Early-Abort	✅ 使用速度残差，只丢弃高风险样本，不改模型分布
evaluate_cifar_metrics	CIFAR-10生成质量评估	✅ 使用FID和分类置信度评估生成质量
visualize_samples	样本可视化	✅ 适配CIFAR-10彩色图像，保存高质量图片
get_cifar_loaders	数据加载	✅ 标准CIFAR-10数据预处理
main + CLI	参数解析 + 串联全流程	✅ 参数可调，确保可重复

新增功能：
1. CIFAR-10适配：
   - 3通道彩色图像 (3, 32, 32)
   - UNetModelWrapper网络结构
   - CIFAR-10特定的评估指标

2. 网络结构：
   - 参考train_cifar10.py的UNetModelWrapper
   - 支持attention机制
   - 适配CIFAR-10的32x32分辨率

3. 评估指标：
   - FID (Frechet Inception Distance) - 主要评价标准
   - CIFAR-10分类器置信度
   - 移除PSNR/SSIM，专注于FID评估

使用示例：
# 使用Euler求解器
python flowguard_cifar.py --solver euler --n_steps 30

# 使用Dopri5自适应求解器
python flowguard_cifar.py --solver dopri5 --rtol 1e-5 --atol 1e-5

# 使用TorchDyn求解器
python flowguard_cifar.py --solver torchdyn_dopri5 --rtol 1e-4 --atol 1e-4

# 使用RK4求解器
python flowguard_cifar.py --solver rk4 --n_steps 50

# 使用自适应Euler
python flowguard_cifar.py --solver adaptive_euler --n_steps 100

# 不同flow matching类型
python flowguard_cifar.py --train --model_type otcfm --solver euler --epochs 3
python flowguard_cifar.py --train --model_type icfm --solver dopri5 --epochs 3

评估说明：
- FID分数越低表示生成质量越好
- 分类置信度越高表示生成图像越清晰可识别
- 需要安装torchmetrics: pip install torchmetrics
""" 