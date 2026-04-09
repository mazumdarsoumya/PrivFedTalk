import argparse, os, torch
from privfedtalk.utils.config import load_config
from privfedtalk.models.build_model import build_model
from privfedtalk.utils.io import ensure_dir
from privfedtalk.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser(description="Inference demo (saves tensor).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="outputs/samples/demo.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device).eval()
    model.scheduler.to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", {}), strict=False)

    B = 1
    T = cfg["data"]["video_frames"]
    H = cfg["data"]["video_size"]
    W = cfg["data"]["video_size"]
    audio = torch.randn(B, cfg["data"]["audio_len"], device=device)
    ref = torch.randn(B, 3, H, W, device=device)

    with torch.no_grad():
        sample = model.sample(audio=audio, ref=ref, num_frames=T, height=H, width=W)
    ensure_dir(os.path.dirname(args.out))
    torch.save({"sample": sample.cpu()}, args.out)
    print("Saved:", args.out, "shape:", tuple(sample.shape))

if __name__ == "__main__":
    main()
