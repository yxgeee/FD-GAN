from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class CUHK03(Dataset):

    def __init__(self, root, split_id=0, num_val=100):
        super(CUHK03, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "Please follow README.md to prepare CUHK03 dataset.")

        self.load(num_val)