from privfedtalk.utils.config import load_config
from privfedtalk.fl.server.orchestrator import run_single_round

def test_one_round():
    cfg = load_config("configs/default.yaml")
    info = run_single_round(cfg, 0)
    assert info["round"] == 0
