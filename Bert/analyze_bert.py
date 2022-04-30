from numpy import loadtxt
from transformers import BertTokenizer
import torch

KEYS_VALUES_PATH = "./bert_omp_results/"
layer_num = 1

# load the sparse keys and values, that were computed by the OMP
sparse_keys = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_keys.csv", delimiter=',').astype("double")
sparse_keys = torch.from_numpy(sparse_keys)
sparse_values = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_values.csv", delimiter=',').astype("double")
sparse_values = torch.from_numpy(sparse_values)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print(f"Results for BERT - Layer {layer_num}")
print(f"-------------------------------")

for i in range(100):
  sorted_keys, indices_keys = torch.sort(torch.abs(sparse_keys[i]), descending=True)
  print("Key: ", bert_tokenizer.batch_decode(indices_keys[0:10]))

  sorted_values, indices_values = torch.sort(torch.abs(sparse_values[i]), descending=True)
  print("Value: ", bert_tokenizer.batch_decode(indices_values[0:10]))
  print("#################################################")