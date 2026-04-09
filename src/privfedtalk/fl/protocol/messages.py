from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ClientUpdate:
    client_id: int
    num_samples: int
    delta: Dict[str, Any]
    score: float
