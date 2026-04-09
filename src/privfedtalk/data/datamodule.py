from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader
from privfedtalk.data.datasets.synthetic import SyntheticTalkingHeadDataset
from privfedtalk.fl.simulators.non_iid_sampler import make_client_samplers

@dataclass
class DataModule:
    cfg: Dict[str, Any]
    dataset: Optional[torch.utils.data.Dataset] = None
    client_samplers: Optional[List[List[int]]] = None

    def setup(self):
        self.dataset = SyntheticTalkingHeadDataset(self.cfg) if self.cfg["data"]["name"] == "synthetic" else SyntheticTalkingHeadDataset(self.cfg)
        self.client_samplers = make_client_samplers(
            num_clients=self.cfg["data"]["num_clients"],
            dataset_len=len(self.dataset),
            non_iid=self.cfg["data"]["non_iid"],
            iid_fraction=self.cfg["data"]["iid_fraction"],
            seed=self.cfg["seed"],
        )

    def get_client_loader(self, client_id: int, shuffle: bool = True):
        assert self.dataset is not None and self.client_samplers is not None
        idxs = self.client_samplers[client_id]
        subset = torch.utils.data.Subset(self.dataset, idxs)
        return DataLoader(
            subset,
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["data"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
