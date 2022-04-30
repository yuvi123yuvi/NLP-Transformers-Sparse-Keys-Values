from numpy import loadtxt
from transformers import RobertaTokenizer
import torch
import sys

KEYS_VALUES_PATH = "./roberta_omp_results/"
layer_num = 9

# load the sparse keys and values, that were computed by the OMP
sparse_keys = loadtxt(KEYS_VALUES_PATH+f"results_omp_roberta_layer11.csv", delimiter=',').astype("double")
sparse_keys = torch.from_numpy(sparse_keys)
# sparse_keys = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_keys.csv", delimiter=',').astype("double")
# sparse_keys = torch.from_numpy(sparse_keys)
# sparse_values = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_values.csv", delimiter=',').astype("double")
# sparse_values = torch.from_numpy(sparse_values)

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

for layer_num in range(11, 12):
  print(f"Results for Roberta - Layer {layer_num}")
  print(f"-------------------------------")

  for i in range(100):
    sorted_keys, indices_keys = torch.sort(torch.abs(sparse_keys[i]), descending=True)
    print("Key: ", roberta_tokenizer.batch_decode(indices_keys[0:10]))

    # sorted_values, indices_values = torch.sort(torch.abs(sparse_values[i]), descending=True)
    # print("Value: ", roberta_tokenizer.batch_decode(indices_values[0:10]))
    # print("#################################################")