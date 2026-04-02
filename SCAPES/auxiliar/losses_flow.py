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

# # The conditional flow psi_t(x) = psi_t(x|x1) taking N(0, sigma_max*I) to N(x1, sigma_min*I)
#     def psi(self, t, x, x1, sigma_min=0.01, sigma_max=1.0):
#         return (t * (sigma_min / sigma_max - 1) + 1) * x + t * x1

#     # The speed of the conditional flow (D/Dt)psi_t(x) = u_t( psi_t(x) | x1)
#     def Dt_psi(self, t, x, x1, sigma_min=0.01, sigma_max=1.0):
#         return (sigma_min / sigma_max - 1) * x + x1

# ==========================================
# FLOW MATCHING MATH FUNCTIONS
# ==========================================
def psi_conditioned(s, X0, X1, sigma_min = 0.01, sigma_max = 1.0):
    """The Optimal Transport path between noise and data."""
    s = sigma_min + (sigma_max - sigma_min) * s

    return (1 - s) * X0 + s * X1

def Dt_psi_conditioned(s, X0, X1):
    """The derivative of the path (the target velocity vector)."""
    return X1 - X0

def flow_matching_loss(model, x0, x1, context, encoded_past, scale_weight=1.0):
    """
    x0, x1: (B, 21, 129)
    scale_weight: Hyperparameter to boost the importance of the 129th channel.
    """
    # 1. Sample time 's'
    s = torch.rand(x1.size(0), 1, 1, device=x1.device)

    # 2. Calculate Path and Target Velocity
    xs = psi_conditioned(s, x0, X1=x1)
    u_conditioned = Dt_psi_conditioned(s, x0, X1=x1)
    
    # 3. Predict Velocity
    s_model = s.squeeze(-1) 
    u_model = model(x_t=xs, s=s_model, context_vector=context, encoded_past=encoded_past)
    
    # --- NEW: Split Latents (0-127) and Scale (128) ---
    # Velocity for latents
    u_model_latents = u_model[:, :, :128]
    u_cond_latents  = u_conditioned[:, :, :128]
    
    # Velocity for scale
    u_model_scale = u_model[:, :, 128:]
    u_cond_scale  = u_conditioned[:, :, 128:]
    
    # 4. Compute Independent MSEs
    loss_latents = F.mse_loss(u_model_latents, u_cond_latents)
    loss_scale   = F.mse_loss(u_model_scale, u_cond_scale)
    
    # Combine with weighting
    total_loss = loss_latents + (scale_weight * loss_scale)
    
    return total_loss, loss_latents, loss_scale
