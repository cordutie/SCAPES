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

def Dt_psi_conditioned(s, X0, X1, sigma_min=0.01, sigma_max=1.0):
    """The derivative of the path (the target velocity vector).

    Chain rule: d/ds psi = d/ds' psi * ds'/ds = (X1 - X0) * (sigma_max - sigma_min).
    """
    return (sigma_max - sigma_min) * (X1 - X0)

def flow_matching_loss(model, x0, x1, context, encoded_past, structure_vector=None, scale_weight=3.0, sigma_min=0.01, sigma_max=1.0):
    """
    x0, x1: (B, 21, 129)
    scale_weight: Hyperparameter to boost the importance of the 129th channel.
    """
    # 1. Sample time 's'
    s = torch.rand(x1.size(0), 1, 1, device=x1.device)

    # 2. Calculate Path and Target Velocity
    xs = psi_conditioned(s, x0, X1=x1)
    u_conditioned = Dt_psi_conditioned(s, x0, X1=x1, sigma_min=sigma_min, sigma_max=sigma_max)
    
    # 3. Predict Velocity
    s_model = s.squeeze(-1) 
    u_model = model(x_t=xs, s=s_model, context_vector=context, encoded_past=encoded_past, structure_vector=structure_vector)
    
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

    # Model's estimate of the final data point (for adversarial regularization)
    X_hat = xs + (1.0 - s) * u_model
    
    return total_loss, loss_latents, loss_scale, X_hat, s


def time_phase_regularizer(x1_hat: torch.Tensor, x1_true: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Penalizes incorrect temporal slopes, weighted heavily towards the end of the flow (s -> 1).
    """
    hat_latents = x1_hat[:, :, :128]
    true_latents = x1_true[:, :, :128]

    diff_hat = hat_latents[:, 1:, :] - hat_latents[:, :-1, :]
    diff_true = true_latents[:, 1:, :] - true_latents[:, :-1, :]

    # 1. Calculate UNREDUCED MSE so we keep the Batch dimension separate
    mse_unreduced = F.mse_loss(diff_hat, diff_true, reduction='none')
    
    # 2. Average the error across the Time and Feature dimensions (leaving shape: [Batch])
    mse_per_batch = mse_unreduced.mean(dim=(1, 2))
    
    # 3. Apply the 's' weighting (reshape s to match [Batch])
    # You can also try s_weights = s.view(-1) ** 2 for an even later fade-in!
    s_weights = s.view(-1) 
    weighted_mse = mse_per_batch * s_weights
    
    # 4. Return the final scalar for backprop
    return weighted_mse.mean()

def fft_phase_regularizer(x1_hat: torch.Tensor, x1_true: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Penalizes incorrect modulation phase in the frequency domain, weighted by true magnitude and ODE time 's'.
    """
    hat_latents = x1_hat[:, :, :128]
    true_latents = x1_true[:, :, :128]

    fft_hat = torch.fft.rfft(hat_latents, dim=1)
    fft_true = torch.fft.rfft(true_latents, dim=1)

    phase_hat = fft_hat / (fft_hat.abs() + 1e-8)
    phase_true = fft_true / (fft_true.abs() + 1e-8)

    mse_complex = F.mse_loss(torch.view_as_real(phase_hat), torch.view_as_real(phase_true), reduction='none')
    mse_per_bin = mse_complex.sum(dim=-1) 

    # 1. Weight by magnitude and average across Frequency and Time dimensions -> shape: [Batch]
    mse_per_batch = (mse_per_bin * fft_true.abs()).mean(dim=(1, 2))

    # 2. Apply the 's' weighting per batch item
    s_weights = s.view(-1)
    weighted_mse = mse_per_batch * s_weights

    # 3. Return final scalar
    return weighted_mse.mean()
