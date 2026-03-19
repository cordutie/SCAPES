import torch

class MultiscaleSpectrogramLoss:
    def __init__(self, 
                 scales=[8192, 4096, 2048], 
                 overlap=0.75, 
                 verbose=False):
        """
        Multi-scale STFT-based spectrogram loss with optional frequency detail filtering.
        """
        self.scales = scales
        self.overlap = overlap
        self.verbose = verbose

    @staticmethod
    def safe_log(x):
        return torch.log(x + 1e-7)

    def compute_stfts(self, signal):
        """
        Computes multi-scale STFTs with optional detail filtering.
        """
        stfts = []
        for scale in self.scales:
            hop_length = int(scale * (1 - self.overlap))
            window = torch.hann_window(scale).to(signal)

            S = torch.stft(
                signal,
                n_fft=scale,
                hop_length=hop_length,
                win_length=scale,
                window=window,
                center=True,
                normalized=True,
                return_complex=True
            ).abs()
            # Transpose to get shape (B, T, F) if batched, or (T, F)
            S = S.transpose(-2, -1)
            # # Apply equal loudness weighting if needed
            # equal_loudness_weights = torch.linspace(1.5, 0.5, S.shape[-1], device=S.device)
            # # Multiply by equal_loudness_weights on last dimension
            # S = S * equal_loudness_weights.view(1, 1, -1)

            if self.verbose:
                print(f"Shape of STFT for scale {scale}: {S.shape}")

            stfts.append(S)
        return stfts

    def __call__(self, x, x_hat):
        """
        Computes the spectrogram loss between x and x_hat.
        """
        ori_stfts = self.compute_stfts(x)
        rec_stfts = self.compute_stfts(x_hat)

        loss = 0
        for s_x, s_y in zip(ori_stfts, rec_stfts):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (self.safe_log(s_x) - self.safe_log(s_y)).abs().mean()
            loss += lin_loss + log_loss
        return loss / len(self.scales)