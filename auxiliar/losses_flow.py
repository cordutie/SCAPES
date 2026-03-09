import torch
import torch.nn.functional as F

# def psi_conditioned(s, X0, X1):
#     return (1 - s) * X0 + s * X1

# def Dt_psi_conditioned(s, X0, X1):
#     return -1 * X0 + X1

# def flow_matching_loss(model, x1, context):
#     x0 = torch.randn_like(x1) 
#     # print("x0 shape:", x0.shape)
#     s  = torch.rand(x1.size(0), 1, 1, device=x1.device)
#     # print("s shape:", s.shape)

#     xs            = psi_conditioned(   s, x0, X1 = x1)
#     u_conditioned = Dt_psi_conditioned(s, x0, X1 = x1)
#     # print("u_conditioned shape:", u_conditioned.shape)
#     u_model       = model.vector_field(s = s, xs = xs, context = context)
#     # print("u_model shape:", u_model.shape)
#     loss          = F.mse_loss(u_model, u_conditioned)
#     return loss

# ==========================================
# FLOW MATCHING MATH FUNCTIONS
# ==========================================
def psi_conditioned(s, X0, X1):
    """The Optimal Transport path between noise and data."""
    return (1 - s) * X0 + s * X1

def Dt_psi_conditioned(s, X0, X1):
    """The derivative of the path (the target velocity vector)."""
    return X1 - X0

def flow_matching_loss(model, x1, context, encoded_past):
    """
    Computes the Flow Matching MSE loss.
    x1: (B, T_frames, frame_dim) - Ground truth target atom
    context: (B, context_dim) - CLAP embedding
    encoded_past: (B, N_past, T_frames, d_model) - LocalEncoder output

    Mathematically this is computing a estimator of the form:
        E_{s ~ U(0,1), x0 ~ N(0,I)} [ || u_model(s, psi_conditioned(s, x0, x1), context) - Dt_psi_conditioned(s, x0, x1) ||^2 ]
    """
    # 1. Generate Noise
    x0 = torch.randn_like(x1) 
    
    # 2. Sample time 's'. 
    # Shape it as (B, 1, 1) so it broadcasts perfectly over (B, T_frames, frame_dim) for the math
    s = torch.rand(x1.size(0), 1, 1, device=x1.device)

    # 3. Calculate Path and Target Velocity
    xs = psi_conditioned(s, x0, X1=x1)
    u_conditioned = Dt_psi_conditioned(s, x0, X1=x1)
    
    # 4. Predict Velocity
    # The model expects 's' as (B, 1), so we squeeze the last dimension
    s_model = s.squeeze(-1) 
    u_model = model(x_t=xs, s=s_model, context_vector=context, encoded_past=encoded_past)
    
    # 5. Compute MSE
    loss = F.mse_loss(u_model, u_conditioned)
    
    return loss, u_model, u_conditioned
