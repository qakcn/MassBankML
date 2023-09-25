from imports import *

class Vocab:
    idxcnt=0
    instances={}

    @staticmethod
    def get(vtype):
        if vtype not in ('root', 'frag', 'link', 'elem'):
            raise ValueError('vtype must be \'root\', \'frag\', \'link\' or \'elem\'.')
        if vtype not in Vocab.instances:
            Vocab.instances[vtype] = Vocab(vtype)
        return Vocab.instances[vtype]
    
    @staticmethod
    def save():
        return dict(
            idxcnt=Vocab.idxcnt,
            instances=Vocab.instances
        )
    
    @staticmethod
    def load(data):
        Vocab.instances = data['instances']
        Vocab.idxcnt = data['idxcnt']
    
    @staticmethod
    def getAllVocabSize():
        return Vocab.idxcnt

    def __init__(self, vtype):
        self.type = vtype
        self.data=[]
        self.identifiers=[]
        self.indexes=[]
    
    def add(self, identifier, data=[]):
        if self.exists(identifier):
            return False
        self.identifiers.append(identifier)
        self.data.append(data)
        Vocab.idxcnt+=1
        self.indexes.append(Vocab.idxcnt)
        return True

    def exists(self, identifier):
        return identifier in self.identifiers
    
    def getIndex(self, identifier):
        if self.exists(identifier):
            return self.indexes[self.identifiers.index(identifier)]
    
    def getIdentifierByIndex(self, index):
        return self.identifiers[self.indexes.index(index)]
    
    def getDataByIndex(self, index):
        return self.data[self.indexes.index(index)]
    
    def setDataByIndex(self, index, data):
        self.data[self.indexes.index(index)]=data
        return True
    
    def getData(self, identifier):
        return self.getDataByIndex(self.getIndex(identifier))
    
    def setData(self, identifier, data):
        if not self.exists(identifier):
            return False
        else:
            return self.setDataByIndex(self.getIndex(identifier), data)
    
    def getType(self):
        return self.type
    
class Timer:
    timer = []
    named_timer={}

    @staticmethod
    def tick(name=None):
        t=time.perf_counter()
        if name is None:
            Timer.timer.append(t)
        else:
            Timer.named_timer[name]=t

    @staticmethod
    def tock(name=None):
        if name is None:
            b=Timer.timer.pop()
        else:
            b=Timer.named_timer[name]
        return time.perf_counter() - b

class Ftree2fpDataset(Dataset):
    def __init__(self, datalist, len_y_max):
        super().__init__(None, transform=None, pre_transform=None, pre_filter=None)
        self.data=datalist
        self.len_y_max=len_y_max

    def len(self):
        return len(self.data)
    
    def indices(self):
        return range(len(self.data))
    
    def get(self, idx):
        d = self.data[idx]
        y=d.y.reshape(-1)
        d.leny = len(y)
        pad = torch.zeros(self.len_y_max-d.leny, dtype=torch.long)
        d.y = torch.cat((y, pad)).reshape([1, -1])
        return d
    
class Ftree2fpGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(Ftree2fpGAT, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.conv3 = pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.lin = torch.nn.Linear(output_dim, output_dim)
        self.linlen = torch.nn.Linear(output_dim, 1)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = pyg_nn.pool.global_mean_pool(x, batch)
        x1 = self.lin(x)
        x1 = torch.sigmoid(x1)
        x2 = self.linlen(x)
        x2 = x2.reshape(-1)
        return x1,x2