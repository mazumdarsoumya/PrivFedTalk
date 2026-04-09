from __future__ import annotations

from typing import Dict, Any, Optional, ClassVar
import torch
import torch.nn.functional as F

from privfedtalk.losses.perceptual_loss import TinyPerceptual, perceptual_l1


class LocalObjective:
    _tiny_cache: ClassVar[dict] = {}
    _lpips_cache: ClassVar[dict] = {}
    _face_cache: ClassVar[dict] = {}

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.devkey = str(device)

        loss_cfg = cfg.get("loss", {})
        self.lambda_tdc = float(loss_cfg.get("lambda_tdc", 0.25))
        self.lambda_id = float(loss_cfg.get("lambda_id", 0.10))
        self.lambda_perc = float(loss_cfg.get("lambda_perc", 0.10))
        self.lambda_sync = float(loss_cfg.get("lambda_sync", 0.05))

        self.perc_fallback = self._get_tiny_perc(device)
        self.lpips_net = self._get_lpips(device)
        self.face_id = self._get_face_id(device)

    @classmethod
    def _get_tiny_perc(cls, device: torch.device):
        key = str(device)
        if key not in cls._tiny_cache:
            net = TinyPerceptual().to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            cls._tiny_cache[key] = net
        return cls._tiny_cache[key]

    @classmethod
    def _get_lpips(cls, device: torch.device):
        key = str(device)
        if key in cls._lpips_cache:
            return cls._lpips_cache[key]
        try:
            import lpips
            net = lpips.LPIPS(net="alex").to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            cls._lpips_cache[key] = net
            return net
        except Exception:
            cls._lpips_cache[key] = None
            return None

    @classmethod
    def _get_face_id(cls, device: torch.device):
        key = str(device)
        if key in cls._face_cache:
            return cls._face_cache[key]
        try:
            from facenet_pytorch import InceptionResnetV1
            net = InceptionResnetV1(pretrained="vggface2").to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            cls._face_cache[key] = net
            return net
        except Exception:
            cls._face_cache[key] = None
            return None

    def _center_frame(self, video: torch.Tensor) -> torch.Tensor:
        return video[:, video.size(1) // 2]

    def _resize_face(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)

    def _noise_target(
        self,
        model,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        eps_from_model: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if eps_from_model is not None:
            return eps_from_model.detach()

        sched = model.scheduler
        sqrt_ab = sched._extract(sched.sqrt_alpha_bar.to(zt.device), t, z0.shape)
        sqrt_omab = sched._extract(sched.sqrt_one_minus_alpha_bar.to(zt.device), t, z0.shape)
        return (zt - sqrt_ab * z0) / torch.clamp(sqrt_omab, min=1e-6)

    def _tdc_loss(self, eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
        if eps_pred.size(1) < 2:
            return eps_pred.new_tensor(0.0)
        d_pred = eps_pred[:, 1:] - eps_pred[:, :-1]
        d_true = eps_true[:, 1:] - eps_true[:, :-1]
        return F.l1_loss(d_pred, d_true)

    def _perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        if self.lpips_net is not None:
            p = pred * 2.0 - 1.0
            t = target * 2.0 - 1.0
            return self.lpips_net(p, t).mean()
        return perceptual_l1(self.perc_fallback, pred, target)

    def _identity_similarity(self, pred_center: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        pred_center = pred_center.clamp(0.0, 1.0)
        ref = ref.clamp(0.0, 1.0)
        if self.face_id is not None:
            pred_face = self._resize_face(pred_center)
            ref_face = self._resize_face(ref)
            pred_emb = self.face_id(pred_face)
            ref_emb = self.face_id(ref_face)
            sim = F.cosine_similarity(pred_emb, ref_emb, dim=1)
            sim01 = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)
            return sim01.mean()
        # fallback proxy if face net unavailable
        return (1.0 - F.l1_loss(pred_center, ref)).clamp(0.0, 1.0)

    def _sync_proxy_loss(self, pred_video: torch.Tensor, audio: Optional[torch.Tensor]) -> torch.Tensor:
        if audio is None or pred_video.size(1) < 2:
            return pred_video.new_tensor(0.0)

        b, t, c, h, w = pred_video.shape
        mouth = pred_video[:, :, :, h // 2 :, w // 4 : (3 * w) // 4]
        motion = (mouth[:, 1:] - mouth[:, :-1]).abs().mean(dim=(2, 3, 4))  # [B, T-1]

        audio_env = audio.abs().unsqueeze(1)                  # [B, 1, L]
        audio_env = F.adaptive_avg_pool1d(audio_env, t - 1).squeeze(1)

        motion = motion / (motion.mean(dim=1, keepdim=True) + 1e-6)
        audio_env = audio_env / (audio_env.mean(dim=1, keepdim=True) + 1e-6)

        return F.mse_loss(motion, audio_env)

    def _temporal_jitter(self, pred_video: torch.Tensor, gt_video: torch.Tensor) -> torch.Tensor:
        if pred_video.size(1) < 2:
            return pred_video.new_tensor(0.0)
        pred_delta = pred_video[:, 1:] - pred_video[:, :-1]
        gt_delta = gt_video[:, 1:] - gt_video[:, :-1]
        return F.l1_loss(pred_delta, gt_delta)

    def __call__(self, model, batch: Dict[str, torch.Tensor], epoch: int, stage: str):
        video = batch["video"].to(self.device, non_blocking=True).float()
        ref = batch["ref"].to(self.device, non_blocking=True).float()
        audio = batch.get("audio", None)
        if audio is not None:
            audio = audio.to(self.device, non_blocking=True).float()

        out = model(video=video, audio=audio, ref=ref)

        z0 = out["z0"]
        zt = out["zt"]
        t = out["t"]
        eps_pred = out["eps_pred"]
        eps_true = self._noise_target(model, z0, zt, t, out.get("eps", None))

        # Paper-aligned diffusion objective
        diff_loss = F.mse_loss(eps_pred, eps_true)
        tdc_loss = self._tdc_loss(eps_pred, eps_true)

        # Decode x0 prediction for auxiliary losses
        x0_pred = model.scheduler.predict_x0(zt, t, eps_pred)
        video_hat = model.vae.decode_video(x0_pred).clamp(0.0, 1.0)

        center_gt = self._center_frame(video)
        center_hat = self._center_frame(video_hat)

        id_sim = self._identity_similarity(center_hat, ref)
        id_loss = 1.0 - id_sim

        perc_loss = self._perceptual_loss(center_hat, center_gt)
        sync_loss = self._sync_proxy_loss(video_hat, audio)
        tjitter = self._temporal_jitter(video_hat, video)

        total = (
            diff_loss
            + self.lambda_tdc * tdc_loss
            + self.lambda_id * id_loss
            + self.lambda_perc * perc_loss
            + self.lambda_sync * sync_loss
        )

        # Keep a proxy scalar for logging only; do not use as paper headline metric
        temporal_stability = (1.0 / (1.0 + tjitter)).clamp(0.0, 1.0)
        acc = (0.5 * id_sim + 0.5 * temporal_stability).clamp(0.0, 1.0)

        stats = {
            "diff_loss": float(diff_loss.detach().item()),
            "tdc_loss": float(tdc_loss.detach().item()),
            "identity_loss": float(id_loss.detach().item()),
            "perceptual_loss": float(perc_loss.detach().item()),
            "sync_loss": float(sync_loss.detach().item()),
            "temporal_jitter": float(tjitter.detach().item()),
            "identity_sim": float(id_sim.detach().item()),
            "temporal_stability": float(temporal_stability.detach().item()),
        }
        return total, acc, stats
