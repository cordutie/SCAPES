import torch
import torch.nn.functional as F

# ==========================================
# SPECTRAL REPRESENTATION HELPERS
# ==========================================

def time_to_spectral(latent: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Transform time-domain latents to spectral representation (log-mag, cos-phase, sin-phase).

    Args:
        latent: [B, T, 128] time-domain latents
        epsilon: small constant for log stability

    Returns:
        [B, F, 384] spectral representation, F = T//2 + 1
    """
    fft = torch.fft.rfft(latent, dim=1)
    mag = fft.abs()
    angle = fft.angle()
    log_mag = torch.log(mag + epsilon)
    cos_phase = torch.cos(angle)
    sin_phase = torch.sin(angle)
    return torch.cat([log_mag, cos_phase, sin_phase], dim=-1)


def spectral_to_time(spectral: torch.Tensor, T: int, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Reconstruct time-domain latents from spectral representation.

    Args:
        spectral: [B, F, 384] where F = T//2 + 1
        T: original time dimension length
        epsilon: small constant matching time_to_spectral

    Returns:
        [B, T, 128] time-domain latents
    """
    log_mag = spectral[:, :, :128]
    cos_phase = spectral[:, :, 128:256]
    sin_phase = spectral[:, :, 256:384]
    mag = torch.exp(log_mag) - epsilon
    mag = torch.clamp(mag, min=0.0)
    real = mag * cos_phase
    imag = mag * sin_phase
    complex_repr = torch.complex(real, imag)
    return torch.fft.irfft(complex_repr, n=T, dim=1)


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


def spectral_flow_matching_loss(model, x0, x1_time, context, encoded_past, structure_vector=None,
                                 scale_weight=3.0, sigma_min=0.01, sigma_max=1.0, epsilon=1e-6):
    """
    Flow matching loss computed in the spectral domain.

    The target latents are transformed to a spectral representation
    (log-magnitude, cos-phase, sin-phase) before computing the OT path.
    The loss is a weighted sum of 4 independent MSEs: mag, cos, sin, scale.

    Args:
        x0: [B, F, 385] noise sampled in spectral space
        x1_time: [B, T, 129] target in time domain (as loaded from dataset)
        scale_weight: boost for the scale channel loss

    Returns:
        total_loss, mag_loss, cos_loss, sin_loss, scale_loss, X_hat, s
        X_hat is in time domain [B, T, 129] for discriminator compatibility
    """
    T = x1_time.size(1)

    # 1. Transform target latents to spectral domain
    latent_time = x1_time[:, :, :128]           # [B, T, 128]
    latent_spectral = time_to_spectral(latent_time, epsilon)  # [B, F, 384]

    # Scale: single value per atom expanded along frequency bins
    scale_val = x1_time[:, 0:1, 128:]           # [B, 1, 1] — constant across T
    n_freq = latent_spectral.size(1)
    scale_expanded = scale_val.expand(-1, n_freq, -1)  # [B, F, 1]

    x1_spectral = torch.cat([latent_spectral, scale_expanded], dim=-1)  # [B, F, 385]

    # 2. Sample time and compute OT path in spectral space
    s = torch.rand(x1_spectral.size(0), 1, 1, device=x1_spectral.device)
    # s = torch.pow(s, 0.33)  # Optional: bias towards later times for better convergence
    xs = psi_conditioned(s, x0, X1=x1_spectral, sigma_min=sigma_min, sigma_max=sigma_max)
    u_conditioned = Dt_psi_conditioned(s, x0, X1=x1_spectral, sigma_min=sigma_min, sigma_max=sigma_max)

    # 3. Predict velocity (model expects spectral-dim input)
    s_model = s.squeeze(-1)
    u_model = model(x_t=xs, s=s_model, context_vector=context,
                    encoded_past=encoded_past, structure_vector=structure_vector)

    # 4. Split velocity into 4 components
    u_model_mag   = u_model[:, :, :128]
    u_model_cos   = u_model[:, :, 128:256]
    u_model_sin   = u_model[:, :, 256:384]
    u_model_scale = u_model[:, :, 384:]

    u_cond_mag   = u_conditioned[:, :, :128]
    u_cond_cos   = u_conditioned[:, :, 128:256]
    u_cond_sin   = u_conditioned[:, :, 256:384]
    u_cond_scale = u_conditioned[:, :, 384:]

    # 5. Compute 4 independent MSEs
    loss_mag   = F.mse_loss(u_model_mag, u_cond_mag)
    loss_cos   = F.mse_loss(u_model_cos, u_cond_cos)
    loss_sin   = F.mse_loss(u_model_sin, u_cond_sin)
    loss_scale = F.mse_loss(u_model_scale, u_cond_scale)

    total_loss = loss_mag + loss_cos + loss_sin + scale_weight * loss_scale

    # 6. X_hat for adversarial training — convert back to time domain
    X_hat_spectral = xs + (1.0 - s) * u_model                          # [B, F, 385]
    X_hat_latent_time = spectral_to_time(X_hat_spectral[:, :, :384], T, epsilon)  # [B, T, 128]
    X_hat_scale = X_hat_spectral[:, :, 384:].mean(dim=1, keepdim=True)  # [B, 1, 1]
    X_hat_scale = X_hat_scale.expand(-1, T, -1)                         # [B, T, 1]
    X_hat = torch.cat([X_hat_latent_time, X_hat_scale], dim=-1)         # [B, T, 129]

    return total_loss, loss_mag, loss_cos, loss_sin, loss_scale, X_hat, s
