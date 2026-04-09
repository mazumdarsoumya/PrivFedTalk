import os, json, torch
from typing import Dict, Any
from privfedtalk.models.build_model import build_model
from privfedtalk.data.datamodule import DataModule
from privfedtalk.utils.dist import resolve_device
from privfedtalk.utils.io import ensure_dir
from privfedtalk.metrics.identity_arcface import identity_similarity
from privfedtalk.metrics.sync_syncnet import sync_score
from privfedtalk.metrics.temporal_jitter import temporal_jitter
from privfedtalk.metrics.lpips_metric import lpips_like

def evaluate_and_report(cfg: Dict[str, Any]):
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = cfg.get("output_dir", "outputs")
    reports_dir = os.path.join(out_dir, "reports")
    ensure_dir(reports_dir); ensure_dir(os.path.join(reports_dir,"tables")); ensure_dir(os.path.join(reports_dir,"figures"))

    dm = DataModule(cfg); dm.setup()
    loader = dm.get_client_loader(0, shuffle=False)

    model = build_model(cfg).to(device).eval()
    model.scheduler.to(device)

    ckpt = os.path.join(out_dir, "checkpoints", "server", "global_last.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", {}), strict=False)

    id_scores=[]; sync_scores=[]; jit_scores=[]; lp_scores=[]
    n_batches = int(cfg["eval"]["num_batches"])

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches: break
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            ref = batch["ref"].to(device)

            out = model(video=video, audio=audio, ref=ref)
            x0_pred = model.scheduler.predict_x0(out["zt"], out["t"], out["eps_pred"])
            video_hat = model.vae.decode_video(x0_pred)

            id_ref = model.id_enc(ref)
            id_hat = model.id_enc(video_hat[:,0])
            aud_emb = model.audio_enc(audio)
            vid_emb = id_hat

            id_scores.append(identity_similarity(id_ref, id_hat))
            sync_scores.append(sync_score(aud_emb, vid_emb))
            jit_scores.append(temporal_jitter(video_hat))
            lp_scores.append(lpips_like(video_hat[:,0], video[:,0], device=device))

    summary = {
        "methods": {
            cfg["fl"]["aggregator"]: {
                "identity": float(sum(id_scores)/max(1,len(id_scores))),
                "sync": float(1.0 - (sum(sync_scores)/max(1,len(sync_scores)))),
                "lpips": float(sum(lp_scores)/max(1,len(lp_scores))),
                "temporal_jitter": float(sum(jit_scores)/max(1,len(jit_scores))),
            }
        },
        "notes": "Synthetic evaluation. Integrate ArcFace/SyncNet/FID for journal-grade metrics."
    }

    out_path = os.path.join(reports_dir, "metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote:", out_path)
    return summary
