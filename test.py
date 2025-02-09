# PSL imports
from pathlib import Path
from statistics import mean

# Third-party imports
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader

import torchmetrics as tm

# Local imports
from classes import *

##################################################
# Parameters that can be edited by the user      #
##################################################
## Paths
input_path = Path("inputs")
output_path = Path("outputs")
intermediate_path = Path("intermediates")

## Files
element_list_file = input_path / "element.nopd.pkl"

train_num = "1706325061"
epoch_num = 300


##################################################
# End of parameters, do not edit below this line #
##################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_output_path = output_path / train_num
eval_path = train_output_path / "eval_casmi"

TS("Loading data sets...").ip()
# eval_dataset = torch.load(train_output_path / "valid_dataset.pkl")
eval_dataset = pd.read_pickle(intermediate_path / "dataset.casmi.prepared.pkl")
eval_dataset = MassbankMLDataset(eval_dataset)
indices = list(eval_dataset.indices())
eval_dataset = Subset(eval_dataset, indices)
TS("Done.").green().p()

TS("Loading configurations..").ip()
configurations = torch.load(train_output_path / "configurations.pkl")
TS("Done.").green().p()

TS("Loading checkpoint parameters...").ip()
checkpoint = torch.load(train_output_path / "checkpoints" / f"checkpoint.{epoch_num}.pkl")
TS("Done.").green().p()

eval_path.mkdir(parents=True, exist_ok=True)

# eval_tfwriter = SummaryWriter(train_output_path / "tfevent/eval")

TS("Constructing model...").ip()
model = Ftree2fpGAT(**configurations["model"]).to(device)
loss_fn = torch.nn.BCELoss()
accuracy=tm.Accuracy(task="binary").to(device)
precision=tm.Precision(task="binary").to(device)
auc=tm.AUROC(task="binary").to(device)
recall=tm.Recall(task="binary").to(device)
f1=tm.F1Score(task="binary").to(device)
TS("Done.").green().p()

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

loss_list=[]
auccuracy_list=[]
precision_list=[]
auc_list=[]
recall_list=[]
f1_list=[]

results = {}

for idx in tqdm(eval_dataset.indices, desc="Evaluating", ncols=100):
    data = eval_dataset.dataset.get(idx)
    data = data.to(device)
    y_pred = model(data)
    y_true = data.y

    loss = loss_fn(y_pred, y_true.float())
    accuracy_score = accuracy(y_pred, y_true.float())
    precision_score = precision(y_pred, y_true.float())
    auc_score = auc(y_pred, y_true.float())
    recall_score = recall(y_pred, y_true.float())
    f1_score = f1(y_pred, y_true.float())

    loss_list.append(loss.item())
    auccuracy_list.append(float(accuracy_score))
    precision_list.append(float(precision_score))
    auc_list.append(float(auc_score))
    recall_list.append(float(recall_score))
    f1_list.append(float(f1_score))

    results[idx] = {
        "data": data,
        "y_pred": y_pred,
    }

indicator = dict(
    loss = {"list":loss_list, "mean":mean(loss_list)},
    accuracy = {"list":auccuracy_list, "mean":float(accuracy.compute())},
    precision = {"list":precision_list, "mean":float(precision.compute())},
    auc = {"list":auc_list, "mean":float(auc.compute())},
    recall = {"list":recall_list, "mean":float(recall.compute())},
    f1 = {"list":f1_list, "mean":float(f1.compute())},
)

torch.save(results, eval_path / "results.pkl")
torch.save(indicator, eval_path / "indicator.pkl")
