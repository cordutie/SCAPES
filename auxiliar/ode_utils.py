import torch
from torchdiffeq import odeint

class ODEFuncWithCounter(torch.nn.Module):
    def __init__(self, u, x0, context, max_nfe=None):
        super().__init__()
        self.u = u
        self.context = context
        self.nfe = 0  # Number of function evaluations
        self.max_nfe = max_nfe  # Maximum allowed evaluations
        self.x0_shape = x0.shape
        self.exceeded_limit = False

    def forward(self, t_scalar, x_flat):
        # Check if we've exceeded the limit
        if self.max_nfe is not None and self.nfe >= self.max_nfe:
            if not self.exceeded_limit:
                print(f"Warning: NFE limit ({self.max_nfe}) reached. Using last computed derivative.")
                self.exceeded_limit = True
            # Return zero derivatives to effectively stop integration
            return torch.zeros_like(x_flat)
        
        self.nfe += 1  # Increment counter
        x = x_flat.view(self.x0_shape)
        t_batch = torch.full((x.size(0),), t_scalar.item(), device=x.device)
        dxdt = self.u(x, t_batch, self.context)
        return dxdt.reshape(-1)

def sample_with_ode(u, x0, context, t_span=(0.0, 1.0), atol=1e-5, rtol=1e-5, 
                   max_nfe=None, method='dopri5', step_size=None, nfe_report=False):
    """
    Sample audio using ODE integration with conditioning.
    Also tracks number of steps (function evaluations) and optionally caps them.
    
    Args:
        u: Vector field function
        x0: Initial condition
        context: Conditioning context
        t_span: Time span tuple (start, end)
        atol: Absolute tolerance for adaptive methods
        rtol: Relative tolerance for adaptive methods
        max_nfe: Maximum number of function evaluations (None = unlimited)
        method: ODE solving method ('dopri5', 'midpoint', 'euler', etc.)
        step_size: Fixed step size for fixed-step methods (None = adaptive)
    
    Returns:
        Final state x1
    """
    device = x0.device
    t0, t1 = t_span
    t = torch.tensor([t0, t1], device=device)

    # Wrap vector field with counter and NFE limiting
    ode_func = ODEFuncWithCounter(u, x0, context, max_nfe=max_nfe)
    
    x0_flat = x0.reshape(-1)
    
    # Prepare ODE solver options
    ode_kwargs = {'atol': atol, 'rtol': rtol, 'method': method}
    
    # Add step size for fixed-step methods
    if step_size is not None:
        ode_kwargs['options'] = {'step_size': step_size}
    
    try:
        sol = odeint(ode_func, x0_flat, t, **ode_kwargs)
        x1 = sol[-1].view_as(x0)
    except Exception as e:
        print(f"ODE integration failed: {e}")
        if max_nfe is not None and ode_func.nfe >= max_nfe:
            print("This might be due to NFE limit being reached.")
        # Return the initial condition as fallback
        x1 = x0

    if nfe_report:
        status_msg = f"ODE steps (function evaluations): {ode_func.nfe}"
        if max_nfe is not None:
            status_msg += f" / {max_nfe} (limit)"
            if ode_func.exceeded_limit:
                status_msg += " - LIMIT REACHED"
        print(status_msg)
        
    return x1

# Convenience functions for common AudioBox-style configurations
def sample_with_ode_capped(u, x0, context, max_nfe=32, method='midpoint', **kwargs):
    """
    Sample with NFE capped similar to AudioBox default (32 evaluations).
    Uses midpoint method by default.
    """
    if method == 'midpoint' and 'step_size' not in kwargs:
        # Calculate step size for ~16 steps (32 NFE with midpoint)
        kwargs['step_size'] = 0.0625  # 1/16
    
    return sample_with_ode(u, x0, context, max_nfe=max_nfe, method=method, **kwargs)

def sample_with_ode_fast(u, x0, context, max_nfe=16, method='midpoint', **kwargs):
    """
    Fast sampling with fewer evaluations.
    """
    if method == 'midpoint' and 'step_size' not in kwargs:
        kwargs['step_size'] = 0.125  # 1/8 -> 8 steps, 16 NFE
    
    return sample_with_ode(u, x0, context, max_nfe=max_nfe, method=method, **kwargs)

def sample_with_ode_quality(u, x0, context, max_nfe=128, method='midpoint', **kwargs):
    """
    High quality sampling with more evaluations.
    """
    if method == 'midpoint' and 'step_size' not in kwargs:
        kwargs['step_size'] = 0.015625  # 1/64 -> 64 steps, 128 NFE
    
    return sample_with_ode(u, x0, context, max_nfe=max_nfe, method=method, **kwargs)

# Example usage:
"""
# Your original usage (unlimited NFE, ~3000 steps):
x1 = sample_with_ode(u, x0, context)

# Capped at 32 NFE (AudioBox style):
x1_capped = sample_with_ode_capped(u, x0, context)

# Custom NFE limit:
x1_custom = sample_with_ode(u, x0, context, max_nfe=100, method='dopri5')

# Fixed-step with specific NFE:
x1_fixed = sample_with_ode(u, x0, context, max_nfe=50, method='euler', step_size=0.02)
"""