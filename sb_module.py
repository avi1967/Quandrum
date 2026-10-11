import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional

class SimulatedBifurcation(nn.Module):
    """
    Discrete Simulated Bifurcation (dSB) implemented as a static PyTorch Module.
    Fully exportable via torch.jit.trace / ONNX.
    
    Fixed tensor shapes:
      x, y: (128, 128) - 128 replicas x 128 spins
      J:    (128, 128) - Coupling matrix
      h:    (128,)     - Magnetic field vector
      c0:   (1,)       - Scaling constant
    """
    def __init__(self, steps: int = 100, dt: float = 0.5, a0: float = 1.0):
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.a0 = a0

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        J: torch.Tensor,
        h: torch.Tensor,
        c0: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs unrolled dSB solver loop using only matmul and elementwise operations.
        Returns final position state x of shape (128, 128).
        """
        for t in range(self.steps):
            # Pump schedule a(t) ramping from 0 to a0
            a_t = self.a0 * (t / float(self.steps))
            
            # Spin direction vector s in {-1.0, 1.0}
            s = torch.sign(x)
            
            # Force calculation: c0 * (s @ J + h)
            # s is (128, 128), J is (128, 128) -> s @ J is (128, 128)
            force = c0 * (torch.matmul(s, J) + h)
            
            # Update momentum y
            y = y + (-(self.a0 - a_t) * x + force) * self.dt
            
            # Update position x
            x = x + self.a0 * y * self.dt
            
            # Wall boundary enforcement: clamp x in [-1.0, 1.0] and reset momentum y to 0 where wall is hit
            hit_wall = (torch.abs(x) >= 1.0)
            x = torch.clamp(x, -1.0, 1.0)
            # Trace-safe zeroing of y where hit_wall is True
            y = torch.where(hit_wall, torch.zeros_like(y), y)

        return x

def solve_simulated_bifurcation(
    J_np: np.ndarray,
    h_np: np.ndarray,
    num_spins: int,
    num_replicas: int = 128,
    max_spins: int = 128,
    steps: int = 100,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python wrapper that pads J, h to (128, 128), computes c0, initializes x & y,
    runs the PyTorch SB module, and picks the best replica spin vector.
    
    Returns:
      best_spins: (num_spins,) array in {-1, 1}
      all_spins: (num_replicas, num_spins) array of raw spins before replica selection
    """
    torch.manual_seed(seed)
    
    # Calculate c0 scale factor: 0.5 * sqrt((N-1) / sum(J^2))
    sum_J2 = np.sum(J_np ** 2)
    if sum_J2 > 0 and num_spins > 1:
        c0_val = float(0.5 * np.sqrt((num_spins - 1) / sum_J2))
    else:
        c0_val = 1.0

    # Zero-pad J to (max_spins, max_spins)
    J_padded = np.zeros((max_spins, max_spins), dtype=np.float32)
    J_padded[:num_spins, :num_spins] = J_np
    
    # Zero-pad h to (max_spins,)
    h_padded = np.zeros(max_spins, dtype=np.float32)
    h_padded[:num_spins] = h_np

    # Convert to PyTorch Tensors
    J_t = torch.tensor(J_padded, dtype=dtype, device=device)
    h_t = torch.tensor(h_padded, dtype=dtype, device=device)
    c0_t = torch.tensor(c0_val, dtype=dtype, device=device)

    # Initialize random x around 0, and y at 0
    x_init = 0.1 * torch.randn((num_replicas, max_spins), dtype=dtype, device=device)
    y_init = torch.zeros((num_replicas, max_spins), dtype=dtype, device=device)

    # Instantiate PyTorch Module
    sb_net = SimulatedBifurcation(steps=steps).to(dtype=dtype, device=device)
    sb_net.eval()

    with torch.no_grad():
        x_out = sb_net(x_init, y_init, J_t, h_t, c0_t)

    # Convert back to NumPy
    x_out_np = x_out.cpu().to(torch.float32).numpy()
    spins_out_np = np.sign(x_out_np)
    # Ensure no exact 0s in spins
    spins_out_np[spins_out_np == 0] = 1.0

    # Evaluate Ising energy for all replicas on original non-padded num_spins
    spins_unpadded = spins_out_np[:, :num_spins] # shape (128, N)
    
    energies = []
    for r in range(num_replicas):
        s_r = spins_unpadded[r]
        e = -0.5 * float(s_r @ J_np @ s_r) - float(h_np @ s_r)
        energies.append(e)

    best_replica_idx = int(np.argmin(energies))
    best_spins = spins_unpadded[best_replica_idx]

    return best_spins, spins_unpadded
