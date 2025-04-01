


from typing import List, Tuple
from torch.utils.data import Dataset


class PyGConcatDataset(Dataset):

    def __init__(self, datasets: List[Dataset]):
        self._datasets = datasets
        self._len = sum(len(dataset) for dataset in datasets)
        self._indexes = []

        # Calculate distribution of indexes in all datasets
        cumulative_index = 0
        for idx, dataset in enumerate(datasets):
            next_cumulative_index = cumulative_index + len(dataset)
            self._indexes.append((cumulative_index, next_cumulative_index, idx))
            cumulative_index = next_cumulative_index

        print(f"Datasets summary length: {self._len}")
        print(f"Datasets indexes: {self._indexes}")

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:

        for start, stop, dataset_index in self._indexes:
            if start <= index < stop:
                dataset = self._datasets[dataset_index]
                return dataset[index - start]

    def __len__(self) -> int:
        return self._len

    @property
    def processed_dir(self):
        return [dataset.processed_dir for dataset in self._datasets]

    @property
    def processed_file_names(self, i=0):
        return [dataset.processed_file_names[i] for dataset in self._datasets]

    @property
    def transform(self):
        # We consider all transforms are the same accross all datatets
        # hence the [0]
        return self._datasets[0].transform


