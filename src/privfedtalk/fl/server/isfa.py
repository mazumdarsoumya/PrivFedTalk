#/home/vineet/PycharmProjects/PrivFedTalk/src/privfedtalk/fl/server/isfa.py

from __future__ import annotations

import math
from typing import List, Dict

from privfedtalk.fl.utils.adapter_state import AdapterState, weighted_sum_states


def compute_client_score(identity_sim: float, temporal_stability: float, alpha: float = 0.5) -> float:
    """
    ISFA score:
        s_k = alpha * IDSim_k + (1-alpha) * TempStab_k
    """
    alpha = float(alpha)
    return alpha * float(identity_sim) + (1.0 - alpha) * float(temporal_stability)


def compute_fedavg_weights(client_infos: List[Dict]) -> List[float]:
    n = [max(1, int(info.get("n_samples", 1))) for info in client_infos]
    total = float(sum(n))
    return [x / total for x in n]


def compute_isfa_weights(client_infos: List[Dict], gamma: float = 1.0) -> List[float]:
    """
    ISFA weighting:
        w_k ∝ n_k * exp(gamma * s_k)
    """
    numerators = []
    for info in client_infos:
        n_k = max(1, int(info.get("n_samples", 1)))
        score = float(info.get("score", 0.0))
        numerators.append(n_k * math.exp(float(gamma) * score))

    denom = float(sum(numerators)) + 1e-12
    return [x / denom for x in numerators]


def aggregate_client_deltas(
    deltas: List[AdapterState],
    client_infos: List[Dict],
    mode: str = "isfa",
    gamma: float = 1.0,
):
    if len(deltas) == 0:
        return {}

    mode = mode.lower()
    if mode == "fedavg":
        weights = compute_fedavg_weights(client_infos)
    else:
        weights = compute_isfa_weights(client_infos, gamma=gamma)

    agg = weighted_sum_states(deltas, weights)
    return agg, weights