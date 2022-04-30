from numpy import loadtxt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

STATE_PATH = './roberta_state_w_finetuning'
layer_num = 4
# load fine-tuned roberta
roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
roberta.load_state_dict(torch.load(STATE_PATH, map_location=torch.device('cpu')))

emb1 = roberta.roberta.get_input_embeddings() # embeddings
E = emb1.weight.detach()
roberta_st = roberta.roberta.state_dict()     # model state dictionary

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

keys = roberta_st[f"encoder.layer.{layer_num}.intermediate.dense.weight"].detach()
values = roberta_st[f"encoder.layer.{layer_num}.output.dense.weight"].detach()

print(f"Results for Roberta - Layer {layer_num} - Without Sparsity")
print(f"-----------------------------------------")

for i in range(100):
  sorted_keys, indices_keys = torch.sort(torch.abs(keys[i]), descending=True)
  print("Key: ", roberta_tokenizer.batch_decode(indices_keys[0:10]))

  sorted_values, indices_values = torch.sort(torch.abs(values[i]), descending=True)
  print("Value: ", roberta_tokenizer.batch_decode(indices_values[0:10]))
  print("#################################################")