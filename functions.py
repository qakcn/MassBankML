import pickle

from torch import optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torchmetrics as tm

from tqdm import tqdm
from statistics import mean
from pathlib import Path

from classes import *

def load_vocab(input_path: Path):
    """
    Load the vocabulary from the input path.
    :param input_path: The path to the input data.
    :return: The loaded vocabulary.
    """
    vocab_data=pickle.loads((input_path/'vocabs_2.pkl').read_bytes())
    Vocab.load(vocab_data)

def fp2bits(datalist, dataname=''):
    """
    Convert the fingerprint to the index.
    :param fp: The fingerprint.
    :return: The index.
    """
    allnum = len(datalist)
    idxnum = 0
    print()
    for edata in datalist:
        fp_bits=[0]*1024
        fp=edata['fingerprint']
        for bit in fp:
            fp_bits[bit]=1
        idxnum = idxnum + 1
        edata['fingerprint_bits'] = fp_bits
        print('\rProcessed {} data {:>5d}/{:>5d}.'.format(dataname, idxnum, allnum), end='', flush=True)
    print()
    return datalist

def prepare_data(input_path):
    """
    Prepare the data for the model.
    :param input_path: The path to the input data.
    :param device: The device to use.
    :return: The prepared data.
    """
    ftree2fp_data=pickle.loads((input_path/'ftree2fp.dataset.pkl').read_bytes())
    ftree2fp_data_prepared = fp2bits(ftree2fp_data, 'ftree2fp')
    (input_path/'ftree2fp.dataset.prepared.pkl').write_bytes(pickle.dumps(ftree2fp_data_prepared, 4))

    mol2fp_data=pickle.loads((input_path/'mol2fp.dataset.pkl').read_bytes())
    mol2fp_data_prepared = fp2bits(mol2fp_data, 'mol2fp')    
    (input_path/'mol2fp.dataset.prepared.pkl').write_bytes(pickle.dumps(mol2fp_data_prepared, 4))

def gen_nodes_features(node_attr):
    node_features = []
    for node in node_attr:
        formula = node['molecularFormula']
        mz = node['mz']
        relint = node['relativeIntensity']
        elems = parseFormula(formula)
        ev = list(elems.values())
        ev.append(mz)
        ev.append(relint)
        node_features.append(ev)
    return node_features

def parseFormula(formula):
    formula=formula + '='
    elems=dict.fromkeys(Constant.Elements, 0)
    elemstr=''
    numstr=''
    for x in range(len(formula)):
        l=formula[x]
        if l == '[':
            continue
        if l in 'CHONPSIBFA]=':
            if elemstr != '':
                if numstr != '':
                    num = int(numstr)
                    numstr = ''
                else:
                    num = 1
                elems[elemstr] = num
            elemstr = l
            if l == ']':
                break
        if l in '0123456789':
            numstr += l
        if l in 'lris':
            elemstr += l
    return elems

def ftree2fp_load_dataset(input_path, node_is_elems=False, classes_is_bit=False):
    """
    Load the data for the ftree2fp model.
    :param input_path: The path to the input data.
    :return: The loaded data.
    """
    ftree2fp_dataset=pickle.loads((input_path/'ftree2fp.dataset.prepared.pkl').read_bytes())

    datalist=[]
    len_y_max=0

    print()
    for edata in ftree2fp_dataset:

        x = gen_nodes_features(edata['node_attr']) if node_is_elems else edata['node']

        ndata=Data(x=torch.tensor(x, dtype=torch.float32 if node_is_elems else torch.long),
                   edge_index=torch.tensor(edata['edge_index'], dtype=torch.long),
                   edge_attr=torch.tensor(edata['edge_attr'], dtype=torch.float32).reshape([-1,1]),
                   y=torch.tensor(edata['fingerprint_bits' if classes_is_bit else 'fingerprint'], dtype=torch.long).reshape([1,-1]))
        len_y_max=max(len_y_max, len(ndata.y[0]))
        datalist.append(ndata)
        print("\rLoad data: {:>5d}/{:>5d} max num_classes: {}".format(len(datalist), len(ftree2fp_dataset), len_y_max), end='', flush=True)

    ndataset=Ftree2fpDataset(datalist, len_y_max)
    print("\n\rFinished Loading Data.")
    return ndataset

def ftree2fp_learn(hidden_dim, num_features, num_layers, batch_size, num_epochs, save_epochs, input_path, output_path, device, gat_heads=1, num_embeddings=0, dropout=0.5, learning_rate=0.001, weight_decay=1e-4, node_is_elems=False, classes_is_bit=False, continuing=False, continue_from='last'):
    train_tfwriter=SummaryWriter(output_path/'tfevent/train')
    valid_tfwriter=SummaryWriter(output_path/'tfevent/valid')

    torch.save({
        'hidden_dim': hidden_dim,
        'num_features': num_features,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'save_epochs': save_epochs,
        'gat_heads': gat_heads,
        'num_embeddings': num_embeddings,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'node_is_elems': node_is_elems,
        'classes_is_bit': classes_is_bit,
    }, output_path/'ftree2fp.params.pkl', pickle_protocol=4)

    if not continuing:
        dataset=ftree2fp_load_dataset(input_path, node_is_elems, classes_is_bit)

        train_size = int(0.8 * len(dataset))
        valid_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - valid_size
        
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
        train_dataset.len_y_max=dataset.len_y_max
        valid_dataset.len_y_max=dataset.len_y_max
        test_dataset.len_y_max=dataset.len_y_max
        torch.save(train_dataset, output_path/'train_dataset.pkl', pickle_protocol=4)
        torch.save(valid_dataset, output_path/'valid_dataset.pkl', pickle_protocol=4)
        torch.save(test_dataset, output_path/'test_dataset.pkl', pickle_protocol=4)
    else:
        train_dataset=torch.load(output_path/'train_dataset.pkl')
        valid_dataset=torch.load(output_path/'valid_dataset.pkl')
        test_dataset=torch.load(output_path/'test_dataset.pkl')

    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model=Ftree2fpGAT(hidden_dim=hidden_dim,
                      num_features=num_features,
                      num_layers=num_layers,
                      output_dim=train_dataset.len_y_max,
                      dropout=dropout,
                      num_embeddings=num_embeddings,
                      classes_is_bit=classes_is_bit,
                      heads=gat_heads).to(device)

    lossf=torch.nn.BCELoss() if classes_is_bit else torch.nn.MSELoss()
    optimizer=optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    accuracy=tm.Accuracy(task="binary").to(device)
    precision=tm.Precision(task="binary").to(device)
    auc=tm.AUROC(task="binary").to(device)
    recall=tm.Recall(task="binary").to(device)
    f1=tm.F1Score(task="binary").to(device)

    if not continuing:
        indicators={
            'train': {
                'loss': [],
            },
            'validation': {
                'loss': [],
                'accuracy': [],
                'precision': [],
                'auc': [],
                'recall': [],
                'f1': [],
            }
        }
        start = 1
    else:
        checkpoint=torch.load(output_path/'ftree2fp.checkpoint.{}.pkl'.format(continue_from))
        indicators=torch.load(output_path/'ftree2fp.indicators.{}.pkl'.format(continue_from))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start=checkpoint['epoch']+1
        if continue_from == 'last':
            (output_path/'ftree2fp.checkpoint.last.pkl').rename(output_path/'ftree2fp.checkpoint.last.{}.pkl'.format(checkpoint['epoch']))
            (output_path/'ftree2fp.indicators.last.pkl').rename(output_path/'ftree2fp.indicators.last.{}.pkl'.format(checkpoint['epoch']))
        del checkpoint
        

    Timer.tick('all')
    for epoch in range(start, num_epochs + 1):
        Timer.tick()
        model.train()
        loss_list=[]

        for data in tqdm(train_loader, desc='Train {:d}'.format(epoch), ncols=100):
            data=data.to(device)
            optimizer.zero_grad()
            y=model(data)
            loss=lossf(y, data.y.float())

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        train_loss=mean(loss_list)
        indicators['train']['loss'].append(train_loss)

        train_tfwriter.add_scalar('loss', train_loss, epoch)

        print('Train {:d}/{:d} loss: {:.6} time: {:.6}/{:.6}\n'.format(epoch, num_epochs, train_loss, Timer.tock(), Timer.tock('all')))

        with torch.no_grad():
            Timer.tick()
            model.eval()
            loss_list=[]

            for data in tqdm(valid_loader, desc='Valid {:d}'.format(epoch), ncols=100):
                data=data.to(device)
                y=model(data)
                loss=lossf(y, data.y.float())

                accuracy.update(y, data.y)
                precision.update(y, data.y)
                auc.update(y, data.y)
                recall.update(y, data.y)
                f1.update(y, data.y)

                loss_list.append(loss.item())

            valid_loss=mean(loss_list)
            indicators['validation']['loss'].append(valid_loss)

            valid_accuracy = accuracy.compute()
            indicators['validation']['accuracy'].append(valid_accuracy)
            valid_precision = precision.compute()
            indicators['validation']['precision'].append(valid_precision)
            valid_auc = auc.compute()
            indicators['validation']['auc'].append(valid_auc)
            valid_recall = recall.compute()
            indicators['validation']['recall'].append(valid_recall)
            valid_f1 = f1.compute()
            indicators['validation']['f1'].append(valid_f1)

            valid_tfwriter.add_scalar('loss', valid_loss, epoch)
            valid_tfwriter.add_scalar('accuracy', valid_accuracy, epoch)
            valid_tfwriter.add_scalar('precision', valid_precision, epoch)
            valid_tfwriter.add_scalar('auc', valid_auc, epoch)
            valid_tfwriter.add_scalar('recall', valid_recall, epoch)
            valid_tfwriter.add_scalar('f1', valid_f1, epoch)

            print('Valid {:d}/{:d} loss: {:.6} accu: {:.6} prcs: {:.6} auc: {:.6} recl: {:.6} f1: {:.6} time: {:.6}/{:.6}\n'.format(epoch, num_epochs, valid_loss, valid_accuracy, valid_precision, valid_auc, valid_recall, valid_f1, Timer.tock(), Timer.tock('all')))

            if epoch % save_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_path/'ftree2fp.checkpoint.{}.pkl'.format(epoch), pickle_protocol=4)

                torch.save(indicators, output_path/'ftree2fp.indicators.{}.pkl'.format(epoch), pickle_protocol=4)

            # if mean(y_loss_list) > 10 or mean(leny_loss_list) > 10 or mean(total_loss_list) > 15:
            #     print('Loss is too large. Stop training.')
            #     break
        accuracy.reset()
        precision.reset()
        auc.reset()
        recall.reset()
        f1.reset()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_path/'ftree2fp.checkpoint.last.pkl'.format(epoch))
    torch.save(indicators, output_path/'ftree2fp.indicators.last.pkl'.format(epoch))

def make_path(output_path):
    """
    Make the path.
    """
    output_path=output_path/time.strftime('%Y%m%d %H%M%S %z')
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path/'tfevent').mkdir(parents=True, exist_ok=True)

    return output_path
