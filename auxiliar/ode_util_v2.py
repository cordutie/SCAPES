
import torch
import torch.nn as nn
from torchdiffeq import odeint

# (Assuming AtomFlowGenerator, AtomEncoder, and MemoryPositionalEncoding are imported here)

# ==========================================
# 1. YOUR ODE SOLVER LOGIC
# ==========================================
class ODEFuncWithCounter(torch.nn.Module):
    def __init__(self, u, x0, context, max_nfe=None):
        super().__init__()
        self.u = u
        self.context = context # This will be a dict containing {'memory': ..., 'clap': ...}
        self.nfe = 0  
        self.max_nfe = max_nfe  
        self.x0_shape = x0.shape
        self.exceeded_limit = False

    def forward(self, t_scalar, x_flat):
        if self.max_nfe is not None and self.nfe >= self.max_nfe:
            if not self.exceeded_limit:
                # print(f"Warning: NFE limit ({self.max_nfe}) reached.")
                self.exceeded_limit = True
            return torch.zeros_like(x_flat)
        
        self.nfe += 1  
        x = x_flat.view(self.x0_shape)
        
        # ODE solvers sometimes pass t as a 0D tensor, we need to batch it
        t_batch = torch.full((x.size(0), 1), t_scalar.item(), device=x.device)
        
        # Call the vector field
        dxdt = self.u(x, t_batch, self.context)
        return dxdt.reshape(-1)

def sample_with_ode_capped(u, x0, context, max_nfe=32, method='midpoint', **kwargs):
    if method == 'midpoint' and 'step_size' not in kwargs:
        kwargs['step_size'] = 0.0625  # 1/16 -> ~16 steps (32 NFE)
        
    device = x0.device
    t = torch.tensor([0.0, 1.0], device=device)

    ode_func = ODEFuncWithCounter(u, x0, context, max_nfe=max_nfe)
    x0_flat = x0.reshape(-1)
    
    ode_kwargs = {'method': method}
    if 'step_size' in kwargs:
        ode_kwargs['options'] = {'step_size': kwargs['step_size']}
    
    try:
        sol = odeint(ode_func, x0_flat, t, **ode_kwargs)
        x1 = sol[-1].view_as(x0)
    except Exception as e:
        print(f"ODE integration failed: {e}")
        x1 = x0
        
    return x1