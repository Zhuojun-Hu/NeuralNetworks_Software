
import os.path as osp

from torch_geometric.data import InMemoryDataset

# watchmal imports
from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)


class EasyInMemoryDataset(InMemoryDataset):
    """
    10/04/2025 : Decision not to use the transform from pyg bc not convenient for storing transformed data in the cache
    
    """

    def __init__(self, folder_path, graph_file_names=['data.pt'], transforms=None, verbose=0):
        super().__init__(root=folder_path, transform=None)

        self.graph_file_names = graph_file_names
        self.transforms = transforms

        # log.info(f"processed_paths : {self.processed_paths}")
        # log.info(f"Processed dir : {self.processed_dir}")

        log.info(f"Len [before load]: {self.len()}")
        log.info(f"Loading from : {self.processed_dir}/{self.processed_file_names[0]}")
        self.load(f"{self.processed_dir}/{self.processed_file_names[0]}")
        log.info(f"Len [after load]: {self.len()}")

    @property
    def processed_file_names(self):
        return self.graph_file_names


    def get(self, idx):
    # Caution : 
    # The pipeline for get is : (starting at the loader)
    # getitem (Dataset) -> get (this class) 
    # If you can super().get() in this class, it's the one from
    # InMemoryDataset, where the big Data is sliced in Data cached in _data_list.
    # then Dataset gets this data, and apply transform.

    # Conclusion : If transform is given in __init__(),
    # this data object will NOT have already be transformed.
    # (See torch_geometric.Data.Dataset.__getitem__() l.292 - 293)
        data = super().get(idx)
            
        # debug purpose
        # log.info(f"Graph [before idx & transforms] : {data}")
        # log.info(f"Data.y, idx: {idx} : {data.y}")
        # for node in range(3):
        #     log.info(f"Data.x, idx: {idx}, node: {node} : {data.x[node]}")
            
        data.idx = idx # for watchmal compatibility
        data = data if self.transforms is None else self.transforms(data.clone())

        # debug purpose
        # log.info(f"Graph [after idx & transforms] : {data}")

        # if isinstance(data, dict):
        #     log.info(f"Data.y, idx: {idx} : {data['data'].y}")
        #     for node in range(3):
        #         log.info(f"Data.x, idx: {idx}, node: {node} : {data['data'].x[node]}")
        # else:
        #     log.info(f"Data.y, idx: {idx} : {data.y}")
        #     for node in range(3):
        #         log.info(f"Data.x, idx: {idx}, node: {node} : {data.x[node]}")

        return data


    def map_labels(in_label, label_set):
        # This method is for watchmal compatibility
        pass