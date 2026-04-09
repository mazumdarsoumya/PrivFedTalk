from typing import List, Dict
import torch
from privfedtalk.fl.protocol.messages import ClientUpdate
from privfedtalk.fl.server.aggregator_fedavg import aggregate_fedavg

def aggregate_fedprox(updates: List[ClientUpdate], mu: float = 0.01) -> Dict[str, torch.Tensor]:
    _ = mu
    return aggregate_fedavg(updates)
