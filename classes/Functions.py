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
## This script contains common functions.                                     ##
##                                                                            ##
## Author: qakcn                                                              ##
## Email: qakcn@hotmail.com                                                   ##
## Date: 2023-12-01                                                           ##
################################################################################

if __name__ == "__main__":
    raise SystemExit("This script is not meant to be run directly")

# PSL imports
from pathlib import Path
from statistics import mean

# Third-party imports
import torch
from torch import optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

import torchmetrics as tm
from tqdm import tqdm

# Local imports
from .Model import *
from .MassbankMLDataset import *

def prepare_row(row, element_list):
    nodes = row["nodes"]
    edges = row["edges"]
    fingerprints = row["fingerprints"]

    x=[]

    keys = sorted(nodes.keys())
    for key in keys:
        xn = []
        node = nodes[key]
        for e in element_list:
            if e in node["elems"]:
                xn.append(node["elems"][e])
            else:
                xn.append(0)
        xn.append(node["mz"])
        xn.append(node["rel_int"])
        x.append(xn)

    x = torch.tensor(x, dtype=torch.float)

    edge_index = list(edges.keys())
    edge_index_reverse = [(y,x) for x,y in edge_index]
    edge_index.extend(edge_index_reverse)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_attr = list(edges.values())*2
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).reshape([-1,1])

    fptype = [
        "FP2",
        "AtomPair",
        "Avalon",
        "MACCS",
        "Morgan",
        "TopologicalTorsion",
        "RDKitFingerprint",
        "CDKFingerprint",
        "PubChemFingerprint",
        "Klekota-Roth"
    ]

    y = []

    for fp in fptype:
        fplist = fingerprints[fp].tolist()
        y.extend(fplist)
    y = torch.tensor(y, dtype=torch.uint8)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
