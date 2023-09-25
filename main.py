from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputpath = Path('./inputs')
outputpath = Path('./outputs')

load_vocab(inputpath)

# prepare_data(inputpath, device)

cur_outputpath=make_path(outputpath)
ftree2fp_learn(num_features=1,
               hidden_dim=5120,
               num_layers=3,
               batch_size=1,
               input_path=inputpath,
               output_path=cur_outputpath,
               num_epochs=300,
               save_epochs=30,
               device=device)