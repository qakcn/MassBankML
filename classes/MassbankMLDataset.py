# Copyright 2023 qakcn
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
## This script contains pytorch dataset classes.                              ##
##                                                                            ##
## Author: qakcn                                                              ##
## Email: qakcn@hotmail.com                                                   ##
## Date: 2023-12-01                                                           ##
################################################################################

# PSL imports

# Third-party imports
import torch
from torch_geometric.data import Dataset
import pandas as pd

# Local imports

class MassbankMLDataset(Dataset):
    def __init__(self, dataset: pd.Series):
        super().__init__(None, transform=None, pre_transform=None, pre_filter=None)
        self.data=dataset

    def len(self):
        return self.data.shape[0]
    
    def indices(self):
        return range(self.data.shape[0])
    
    def get(self, idx: int):
        return self.data.iloc[idx]