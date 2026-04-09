import torch
from privfedtalk.fl.privacy.secure_aggregation import secure_mask_updates
from privfedtalk.fl.protocol.serialization import add_state

def sum_states(states):
    agg = {}
    for s in states:
        agg = add_state(agg, s) if agg else {k: v.clone() for k, v in s.items()}
    return agg

def test_secureagg_cancels():
    d1 = {"a": torch.ones(3,3), "b": torch.zeros(2)}
    d2 = {"a": torch.ones(3,3)*2, "b": torch.ones(2)}
    original = sum_states([d1, d2])

    masked = secure_mask_updates([d1, d2], client_ids=[0, 1], round_seed=123, scale=1e-3)
    masked_sum = sum_states(masked)

    assert torch.allclose(masked_sum["a"], original["a"])
    assert torch.allclose(masked_sum["b"], original["b"])
