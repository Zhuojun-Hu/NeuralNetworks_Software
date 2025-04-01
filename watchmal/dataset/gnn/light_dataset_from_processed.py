
import os.path as osp

from torch_geometric.data import InMemoryDataset

# watchmal imports
from watchmal.utils.logging_utils import setup_logging

log = setup_logging(__name__)


class EasyInMemoryDataset(InMemoryDataset):


    def __init__(self, folder_path, graph_file_names=['data.pt'], transform=None, transforms=None, verbose=0):
        super().__init__(root=folder_path, transform=transform)

        self.graph_file_names = graph_file_names

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
        data.idx = idx # for watchmal compatibility
    
        return data


    def map_labels(in_label, label_set):
        # This method is for watchmal compatibility
        pass