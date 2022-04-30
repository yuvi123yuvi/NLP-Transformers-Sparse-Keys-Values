from numpy import loadtxt
from transformers import GPT2Tokenizer
import torch

KEYS_VALUES_PATH = "./gpt2_omp_results/"
layer_num = 11

# load the sparse keys and values, that were computed by the OMP
sparse_keys = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_keys.csv", delimiter=',').astype("double")
sparse_keys = torch.from_numpy(sparse_keys)
sparse_values = loadtxt(KEYS_VALUES_PATH+f"layer{layer_num}_values.csv", delimiter=',').astype("double")
sparse_values = torch.from_numpy(sparse_values)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print(f"Results for GPT2 - Layer {layer_num}")
print(f"-------------------------------")

for i in range(100):
  sorted_keys, indices_keys = torch.sort(torch.abs(sparse_keys[i]), descending=True)
  print("Key: ", gpt2_tokenizer.batch_decode(indices_keys[0:10]))

  sorted_values, indices_values = torch.sort(torch.abs(sparse_values[i]), descending=True)
  print("Value: ", gpt2_tokenizer.batch_decode(indices_values[0:10]))
  print("#################################################")