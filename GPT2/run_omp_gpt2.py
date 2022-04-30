import torch
from torch import nn
from numpy import savetxt
from transformers import GPT2Model
from sklearn.linear_model import OrthogonalMatchingPursuit

STATE_PATH = './gpt2_state_w_finetuning.pt'
RESULTS_PATH = "./gpt2_omp_results/"

class GPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int):
        super(GPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained("gpt2")
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output

# load fine-tuned bert
gpt2 = GPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128)
gpt2.load_state_dict(torch.load(STATE_PATH, map_location=torch.device('cpu')))

emb1 = gpt2.gpt2model.get_input_embeddings() # embeddings
E = emb1.weight.detach()
gpt2_st = gpt2.gpt2model.state_dict()     # model state dictionary

# for i in reversed(range(11)): # the curr layer
i=11
print(f"####### Iteration {i} #######")
# Compute sparse keys
print("Computing keys...")
keys = gpt2_st[f"h.{i}.mlp.c_fc.weight"].detach() # keys of ff layer
n_non_zero_keys = max(int(0.1 * keys.shape[0]), 1)
reg_keys = OrthogonalMatchingPursuit(n_nonzero_coefs=n_non_zero_keys, normalize=False).fit(E.T.cpu(), keys.cpu())
sparse_keys = reg_keys.predict(E.T)
savetxt(RESULTS_PATH+f"layer{i}_keys.csv", sparse_keys, delimiter=',')

# Compute sparse values
print("Computing values...")
values = gpt2_st[f"h.{i}.mlp.c_proj.weight"].detach() # values of ff layer
n_non_zero_values = max(int(0.1 * values.shape[1]), 1)
reg_values = OrthogonalMatchingPursuit(n_nonzero_coefs=n_non_zero_values, normalize=False).fit(E.T.cpu(), values.T.cpu())
sparse_values = reg_values.predict(E.T)
savetxt(RESULTS_PATH+f"layer{i}_values.csv", sparse_values, delimiter=',')