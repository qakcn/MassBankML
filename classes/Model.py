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
## This script contains pytorch model.                                        ##
##                                                                            ##
## Author: qakcn                                                              ##
## Email: qakcn@hotmail.com                                                   ##
## Date: 2023-12-01                                                           ##
################################################################################

if __name__ == "__main__":
    raise SystemExit("This script is not meant to be run directly")

# PSL imports

# Third-party imports
import torch
from torch import nn
import torch_geometric.nn as pyg_nn

# Local imports
    
class Ftree2fpGAT(torch.nn.Module):
    def __init__(self, hidden_dim, num_features, num_layers, output_dim, heads=1, dropout=0.5, num_embeddings=0, classes_is_bit=False):
        super(Ftree2fpGAT, self).__init__()

        # self.net=pyg_nn.GAT(in_channels=num_features,
        #                     hidden_channels=hidden_dim,
        #                     out_channels=output_dim,
        #                     # dropout=dropout,
        #                     # v2=True,
        #                     heads=4,
        #                     edge_dim=1,
        #                     num_layers=num_layers)
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i==0:
                self.convs.append(pyg_nn.GATConv(num_features, hidden_dim//heads, heads=heads, edge_dim=1))
            else:
                self.convs.append(pyg_nn.GATConv(hidden_dim, hidden_dim//heads, heads=heads, edge_dim=1))
        
        if num_embeddings != 0:
            self.embed = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=num_features)
        
        self.classes_is_bit=classes_is_bit

        self.dropout = dropout

        self.lin1 = torch.nn.Linear(hidden_dim, output_dim)
        # self.lin2 = torch.nn.Linear(output_dim, output_dim)
        # self.lin3 = torch.nn.Linear(output_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if hasattr(self, 'embed'):
            x=self.embed(x).reshape([-1, self.embed.embedding_dim])
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = pyg_nn.pool.global_max_pool(x, batch)
        x = self.lin1(x)
        # x = self.lin2(x)
        # x = self.lin3(x)
        if self.classes_is_bit:
            x = torch.sigmoid(x)
        return x