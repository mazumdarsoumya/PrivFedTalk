import random
from typing import Dict, List

def build_synthetic_client_partition(num_clients: int, samples_per_client: int, non_iid: bool, iid_fraction: float) -> Dict:
    N = num_clients * samples_per_client
    all_indices = list(range(N))
    client_indices: List[List[int]] = []
    for k in range(num_clients):
        start = k * samples_per_client
        end = (k + 1) * samples_per_client
        client_indices.append(all_indices[start:end])

    if non_iid and iid_fraction > 0:
        mix_n = int(samples_per_client * iid_fraction)
        pool = all_indices[:]
        random.shuffle(pool)
        for k in range(num_clients):
            base = client_indices[k]
            for i in range(mix_n):
                base[i] = pool[(k * mix_n + i) % len(pool)]
            client_indices[k] = list(sorted(set(base)))[:samples_per_client]

    return {"num_clients": num_clients, "samples_per_client": samples_per_client, "client_indices": client_indices}
