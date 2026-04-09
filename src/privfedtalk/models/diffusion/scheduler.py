import torch


class DiffusionScheduler:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float):
        self.timesteps = int(timesteps)

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alpha_bar[:-1]])
        self.posterior_var = betas * (1.0 - prev) / (1.0 - alpha_bar)

    def to(self, device):
        for k in [
            "betas",
            "alphas",
            "alpha_bar",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
            "posterior_var",
        ]:
            setattr(self, k, getattr(self, k).to(device))
        return self

    def sample_timesteps(self, batch_size: int, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        a = a.to(t.device)
        t = t.to(a.device)
        return a.gather(0, t).view(-1, 1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        return (x_t - sqrt_omab * eps_pred) / torch.clamp(sqrt_ab, min=1e-6)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        betas_t = self._extract(self.betas, t, x_t.shape)
        alphas_t = self._extract(self.alphas, t, x_t.shape)
        alpha_bar_t = self._extract(self.alpha_bar, t, x_t.shape)

        mean = (1.0 / torch.sqrt(alphas_t)) * (
            x_t - (betas_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        noise = torch.randn_like(x_t)
        var = self._extract(self.posterior_var, t, x_t.shape)
        mask = (t > 0).float().view(-1, 1, 1, 1, 1)

        return mean + mask * torch.sqrt(torch.clamp(var, min=1e-20)) * noise
