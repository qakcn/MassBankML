from classes import *

def load_vocab(input_path):
    """
    Load the vocabulary from the input path.
    :param input_path: The path to the input data.
    :return: The loaded vocabulary.
    """
    vocab_data=pickle.loads((input_path/'vocabs_2.pkl').read_bytes())
    Vocab.load(vocab_data)

def fp2idx(datalist, dataname=''):
    """
    Convert the fingerprint to the index.
    :param fp: The fingerprint.
    :return: The index.
    """
    tobinpattern=lambda x: '{:010b}'.format(x)
    toidxlist = lambda x: [int(c) for c in x]
    allnum = len(datalist)
    idxnum = 0
    y_max = 0
    print()
    for edata in datalist:
        y=edata['y']
        idxnum = idxnum + 1
        y_max = max(y_max, len(y))
        fpidx_list = []
        for bits in y:
            fpidx = toidxlist(tobinpattern(bits))
            fpidx_list.append(fpidx)
        edata['y_idx'] = fpidx_list
        print('\rProcessed {} data {:>5d}/{:>5d}. y_max = {:>3d}'.format(dataname, idxnum, allnum, y_max), end='', flush=True)
    return dict(data = datalist, y_max = y_max)

def prepare_data(input_path, device):
    """
    Prepare the data for the model.
    :param input_path: The path to the input data.
    :param device: The device to use.
    :return: The prepared data.
    """
    ftree2fp_data=pickle.loads((input_path/'ftree2fp.dataset.pkl').read_bytes())
    ftree2fp_data_prepared = fp2idx(ftree2fp_data, 'ftree2fp')
    (input_path/'ftree2fp.dataset.prepared.pkl').write_bytes(pickle.dumps(ftree2fp_data_prepared, 4))

    mol2fp_data=pickle.loads((input_path/'mol2fp.dataset.pkl').read_bytes())
    mol2fp_data_prepared = fp2idx(mol2fp_data, 'mol2fp')    
    (input_path/'mol2fp.dataset.prepared.pkl').write_bytes(pickle.dumps(mol2fp_data_prepared, 4))

def ftree2fp_load_dataset(input_path):
    """
    Load the data for the ftree2fp model.
    :param input_path: The path to the input data.
    :return: The loaded data.
    """
    ftree2fp_dataset=pickle.loads((input_path/'ftree2fp.dataset.prepared.pkl').read_bytes())

    datalist=[]
    len_y_max=0

    print()
    for edata in ftree2fp_dataset['data']:
        ndata=Data(x=torch.tensor(edata['x'], dtype=torch.float).reshape([-1,1]),
                   edge_index=torch.tensor(edata['edge_index'], dtype=torch.long),
                   edge_attr=torch.tensor(edata['edge_attr'], dtype=torch.float32).reshape([-1,1]),
                   y=torch.tensor(edata['y_idx'], dtype=torch.long).reshape([1,-1]))
        len_y_max=max(len_y_max, len(ndata.y[0]))
        datalist.append(ndata)
        print("\rLoad data: {:>5d}/{:>5d} max num_classes: {}".format(len(datalist), len(ftree2fp_dataset['data']), len_y_max), end='', flush=True)

    ndataset=Ftree2fpDataset(datalist, len_y_max)
    print("\n\rFinished Loading Data.")
    return ndataset

def ftree2fp_learn(num_features, hidden_dim, num_layers, batch_size, input_path, output_path, num_epochs, save_epochs, device):
    dataset=ftree2fp_load_dataset(input_path)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataset.len_y_max=dataset.len_y_max
    val_dataset.len_y_max=dataset.len_y_max
    test_dataset.len_y_max=dataset.len_y_max
    torch.save(test_dataset, output_path/'test_dataset.pkl', pickle_protocol=4)

    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model=Ftree2fpGAT(num_features, hidden_dim, num_layers, dataset.len_y_max).to(device)
    y_lossf=torch.nn.BCELoss()
    leny_lossf=torch.nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=0.005)

    train_losses=dict(loss1=[], loss2=[], total_loss=[])
    val_losses=dict(loss1=[], loss2=[], total_loss=[])

    Timer.tick('all')
    for epoch in range(num_epochs):
        Timer.tick()
        model.train()
        y_loss_list=[]
        leny_loss_list=[]
        total_loss_list=[]

        for data in tqdm(train_loader, desc='Epoch {:>3d}'.format(epoch)):
            data=data.to(device)
            optimizer.zero_grad()
            y,leny=model(data)
            y_loss=y_lossf(y, data.y.float())
            leny_loss=leny_lossf(leny, data.leny.float())
            total_loss=y_loss + leny_loss

            y_loss.backward(retain_graph=True)
            leny_loss.backward()
            optimizer.step()

            y_loss_list.append(y_loss.item())
            leny_loss_list.append(leny_loss.item())
            total_loss_list.append(total_loss.item())

        print('\nEpoch {:>4d}/{:>4d} y_loss: {:>10.6f} leny_loss: {:>10.6f} total_loss: {:>10.6f} time: {:>10.6f}/{:>10.6f}'.format(epoch, num_epochs, mean(y_loss_list), mean(leny_loss_list), mean(total_loss_list), Timer.tock(), Timer.tock('all')))

        train_losses['y_loss'].append(mean(y_loss_list))
        train_losses['leny_loss'].append(mean(leny_loss_list))
        train_losses['total_loss'].append(mean(total_loss_list))

        with torch.no_grad():
            Timer.tick()
            model.eval()
            y_loss_list=[]
            leny_loss_list=[]
            total_loss_list=[]

            for data in tqdm(val_loader, desc='Epoch {:>3d}'.format(epoch)):
                data=data.to(device)
                y,leny=model(data)
                y_loss=y_lossf(y, data.y.float())
                leny_loss=leny_lossf(leny, data.leny.float())
                total_loss=y_loss + leny_loss

                y_loss_list.append(y_loss.item())
                leny_loss_list.append(leny_loss.item())
                total_loss_list.append(total_loss.item())

            print('\nEpoch {:>4d}/{:>4d} y_loss: {:>10.6f} leny_loss: {:>10.6f} total_loss: {:>10.6f} time: {:>10.6f}/{:>10.6f}'.format(epoch, num_epochs, mean(y_loss_list), mean(leny_loss_list), mean(total_loss_list), Timer.tock(), Timer.tock('all')))

            val_losses['y_loss'].append(mean(y_loss_list))
            val_losses['leny_loss'].append(mean(leny_loss_list))
            val_losses['total_loss'].append(mean(total_loss_list))

            if(epoch+1) % save_epochs == 0:
                torch.save(model.state_dict(), output_path/'ftree2fp.model.{}.pkl'.format(epoch+1))
                torch.save(train_losses, output_path/'ftree2fp.train_losses.{}.pkl'.format(epoch+1))
                torch.save(val_losses, output_path/'ftree2fp.val_losses.{}.pkl'.format(epoch+1))

            if mean(y_loss_list) > 10 or mean(leny_loss_list) > 10 or mean(total_loss_list) > 15:
                print('Loss is too large. Stop training.')
                break

    torch.save(model.state_dict(), output_path/'ftree2fp.model.pkl')
    torch.save(train_losses, output_path/'ftree2fp.train_losses.pkl')
    torch.save(val_losses, output_path/'ftree2fp.val_losses.pkl')

def make_path(output_path):
    """
    Make the path.
    """
    output_path=output_path/time.strftime('%Y%m%d %H%M%S %z')
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path