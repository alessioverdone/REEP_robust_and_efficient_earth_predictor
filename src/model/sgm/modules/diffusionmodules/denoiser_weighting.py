import torch


class UnitWeighting:
    def __call__(self, sigma):
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2.0
    
class MinSNRWeight:
    """
    MODIFICA 3 (denoiser_config.params.weighting_config.params).

    mode="min" e' il comportamento storico: snr[snr < gamma] = gamma, cioe' il
    clip sul MINIMO. Il peso resta >= gamma ma sui sigma piccoli non ha alcun
    tetto: con LegacyDDPMDiscretization sigma_min ~ 1e-2, quindi sigma**-2
    arriva a ~1e4 e il gradiente e' dominato dai campioni quasi puliti.

    mode="max" e' il min-SNR-gamma vero: min(SNR, gamma), cioe' il clip sul
    MASSIMO, che e' il tetto che si vuole davvero.
    """

    def __init__(self, gamma: float = 5.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(
                f"MinSNRWeight.mode deve essere 'min' o 'max', non {mode!r}"
            )
        self.gamma = float(gamma)
        self.mode = mode

    def __call__(self, sigma):
        snr = sigma**-2.0
        if self.mode == "max":
            return torch.clamp(snr, max=self.gamma)
        # equivalente a snr[snr < gamma] = gamma, ma senza in-place
        return torch.clamp(snr, min=self.gamma)
