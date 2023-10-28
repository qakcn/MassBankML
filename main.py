from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputpath = Path('./inputs')
outputpath = Path('./outputs')

load_vocab(inputpath)

# prepare_data(inputpath)

continuing=False
if continuing:
    cur_outputpath=outputpath/'20231021 204124 +0800'
else:
    cur_outputpath=make_path(outputpath)

ftree2fp_learn(hidden_dim=1024,
               gat_heads=4,
               num_features=14,
               num_layers=3,
               batch_size=64,
               num_epochs=3000,
               save_epochs=30,
               learning_rate=0.0001,
               weight_decay=1e-4,
               input_path=inputpath,
               output_path=cur_outputpath,
               continuing=continuing,
               node_is_elems=True,
               classes_is_bit=True,
               device=device)