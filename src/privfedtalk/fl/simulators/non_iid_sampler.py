import random
from typing import List

def make_client_samplers(num_clients: int, dataset_len: int, non_iid: bool, iid_fraction: float, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    indices = list(range(dataset_len))
    per = dataset_len // num_clients
    client_idxs: List[List[int]] = []
    for k in range(num_clients):
        start = k * per
        end = (k + 1) * per if k < num_clients - 1 else dataset_len
        client_idxs.append(indices[start:end])

    if non_iid:
        mix = int(per * iid_fraction)
        pool = indices[:]
        rng.shuffle(pool)
        for k in range(num_clients):
            block = client_idxs[k][:]
            for i in range(min(mix, len(block))):
                block[i] = pool[(k * mix + i) % len(pool)]
            client_idxs[k] = sorted(set(block))

    return client_idxs
