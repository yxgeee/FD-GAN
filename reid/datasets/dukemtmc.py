from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class DukeMTMC(Dataset):

    def __init__(self, root, split_id=0, num_val=100):
        super(DukeMTMC, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "Please follow README.md to prepare DukeMTMC dataset.")

        self.load(num_val)