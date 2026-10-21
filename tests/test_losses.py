"""Slice B4: flow-matching loss generalised to n_latent / has_scale, with prefix and fixed s (design Model, D5, D17)."""
import torch
import torch.nn.functional as F


# ---- V1 functions, copied verbatim from SCAPES/auxiliar/losses_flow.py at main 5304260 ----
def psi_conditioned(s, X0, X1, sigma_min = 0.01, sigma_max = 1.0):
    """The Optimal Transport path between noise and data."""
    s = sigma_min + (sigma_max - sigma_min) * s

    return (1 - s) * X0 + s * X1

def Dt_psi_conditioned(s, X0, X1, sigma_min=0.01, sigma_max=1.0):
    """The derivative of the path (the target velocity vector).

    Chain rule: d/ds psi = d/ds' psi * ds'/ds = (X1 - X0) * (sigma_max - sigma_min).
    """
    return (sigma_max - sigma_min) * (X1 - X0)

def v1_flow_matching_loss(model, x0, x1, context, encoded_past, structure_vector=None, scale_weight=3.0, sigma_min=0.01, sigma_max=1.0):
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
# ---- end of V1 copy ----


def zero_model(x_t, s, context_vector, encoded_past, structure_vector=None):
    return torch.zeros_like(x_t)


def smooth_model(x_t, s, context_vector, encoded_past, structure_vector=None):
    return torch.tanh(0.5 * x_t + s.reshape(-1, 1, 1))


def _inputs(b, t, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, p, generator=g), torch.randn(b, t, p, generator=g), torch.randn(b, 1024, generator=g)


def test_no_scale_known_value():
    from SCAPES.auxiliar.losses_flow import flow_matching_loss
    x0, x1, ctx = _inputs(4, 32, 16)
    total, l_lat, l_scale, _, _ = flow_matching_loss(zero_model, x0, x1, ctx, None, n_latent=16, has_scale=False)
    want = float(((0.99 * (x1 - x0)) ** 2).mean())
    assert float(l_scale) == 0.0
    assert abs(float(total) - want) <= 1e-6
    assert abs(float(l_lat) - want) <= 1e-6


def test_matches_v1_with_scale():
    from SCAPES.auxiliar.losses_flow import flow_matching_loss
    x0, x1, ctx = _inputs(4, 48, 129)
    torch.manual_seed(123)
    ref = v1_flow_matching_loss(smooth_model, x0, x1, ctx, None)
    torch.manual_seed(123)
    got = flow_matching_loss(smooth_model, x0, x1, ctx, None, n_latent=128, has_scale=True)
    for a, b in zip(got, ref):
        assert float((a - b).abs().max()) <= 1e-6


def test_zero_model_standardised():
    from SCAPES.auxiliar.losses_flow import flow_matching_loss
    x0, x1, ctx = _inputs(64, 24, 128, seed=1)
    total, *_ = flow_matching_loss(zero_model, x0, x1, ctx, None, n_latent=128, has_scale=False)
    want = 0.9801 * 2
    assert abs(float(total) - want) <= 0.05 * want


def test_prefix_and_fixed_s():
    """Prefix keyword: the model sees cat([prefix, x_s]); loss and X_hat use only the last U positions.
    Fixed s keyword: the same s gives the same loss without touching the global RNG."""
    from SCAPES.auxiliar.losses_flow import flow_matching_loss
    x0, x1, ctx = _inputs(4, 32, 16)
    prefix = torch.randn(4, 2, 16, generator=torch.Generator().manual_seed(5))
    seen = {}

    def recording(x_t, s, context_vector, encoded_past, structure_vector=None):
        seen["x_t"] = x_t.clone()
        return torch.tanh(0.5 * x_t + s.reshape(-1, 1, 1))

    def noisy_prefix(x_t, s, context_vector, encoded_past, structure_vector=None):
        out = torch.tanh(0.5 * x_t + s.reshape(-1, 1, 1))
        out[:, :2] = 100.0 * torch.randn_like(out[:, :2])
        return out

    s = torch.full((4, 1, 1), 0.375)
    state = torch.get_rng_state()
    a = flow_matching_loss(recording, x0, x1, ctx, None, n_latent=16, has_scale=False, prefix=prefix, s=s)
    assert torch.equal(torch.get_rng_state(), state)
    assert seen["x_t"].shape == (4, 34, 16)
    assert torch.equal(seen["x_t"][:, :2], prefix)
    b = flow_matching_loss(noisy_prefix, x0, x1, ctx, None, n_latent=16, has_scale=False, prefix=prefix, s=s)
    assert abs(float(a[0]) - float(b[0])) <= 1e-6
    assert a[3].shape == (4, 32, 16)
    # without the prefix the target positions see the same x_s, so the loss is unchanged for a positionwise model
    c = flow_matching_loss(recording, x0, x1, ctx, None, n_latent=16, has_scale=False, s=s)
    assert abs(float(a[0]) - float(c[0])) <= 1e-6
