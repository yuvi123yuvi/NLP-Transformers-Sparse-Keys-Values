from numpy import loadtxt
from transformers import BertModel, BertTokenizer
import torch
from torch import nn

STATE_PATH = './bert_state_w_finetuning'
layer_num = 0

class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        output = self.bert(tokens, attention_mask=masks)
        pooled_output = output.pooler_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

# load fine-tuned bert
bert = BertBinaryClassifier()
bert.load_state_dict(torch.load(STATE_PATH, map_location=torch.device('cpu')))

emb1 = bert.bert.get_input_embeddings() # embeddings
E = emb1.weight.detach()
bert_st = bert.bert.state_dict()     # model state dictionary

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

keys = bert_st[f"encoder.layer.{layer_num}.intermediate.dense.weight"].detach()
values = bert_st[f"encoder.layer.{layer_num}.output.dense.weight"].detach()

print(f"Results for BERT - Layer {layer_num} - Without Sparsity")
print(f"-----------------------------------------")

for i in range(100):
  sorted_keys, indices_keys = torch.sort(torch.abs(keys[i]), descending=True)
  print("Key: ", bert_tokenizer.batch_decode(indices_keys[0:10]))

  sorted_values, indices_values = torch.sort(torch.abs(values[i]), descending=True)
  print("Value: ", bert_tokenizer.batch_decode(indices_values[0:10]))
  print("#################################################")