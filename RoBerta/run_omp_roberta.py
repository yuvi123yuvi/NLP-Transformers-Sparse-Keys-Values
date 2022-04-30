import torch
from torch import nn
import pandas as pd
import numpy as np
from numpy import savetxt, loadtxt
import json
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.linear_model import OrthogonalMatchingPursuit

STATE_PATH = './roberta_state_w_finetuning'
RESULTS_PATH = "./roberta_omp_results/"

# load fine-tuned roberta
roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
roberta.load_state_dict(torch.load(STATE_PATH, map_location=torch.device('cpu')))

emb1 = roberta.roberta.get_input_embeddings() # embeddings
E = emb1.weight.detach()
roberta_st = roberta.roberta.state_dict()     # model state dictionary

for i in reversed(range(4)): # the curr layer
    print(f"####### Iteration {i} #######")
    # Compute sparse keys
    print("Computing keys...")
    keys = roberta_st[f"encoder.layer.{i}.intermediate.dense.weight"].detach() # keys of ff layer
    n_non_zero_keys = max(int(0.1 * keys.shape[0]), 1)
    reg_keys = OrthogonalMatchingPursuit(n_nonzero_coefs=n_non_zero_keys, normalize=False).fit(E.T.cpu(), keys.T.cpu())
    sparse_keys = reg_keys.predict(E.T)
    savetxt(RESULTS_PATH+f"layer{i}_keys.csv", sparse_keys, delimiter=',')

    # Compute sparse values
    print("Computing values...")
    values = roberta_st[f"encoder.layer.{i}.output.dense.weight"].detach() # values of ff layer
    n_non_zero_values = max(int(0.1 * values.shape[1]), 1)
    reg_values = OrthogonalMatchingPursuit(n_nonzero_coefs=n_non_zero_values, normalize=False).fit(E.T.cpu(), values.cpu())
    sparse_values = reg_values.predict(E.T)
    savetxt(RESULTS_PATH+f"layer{i}_values.csv", sparse_values, delimiter=',')