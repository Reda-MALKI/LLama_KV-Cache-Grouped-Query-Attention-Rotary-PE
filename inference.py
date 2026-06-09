from model import Llama, Config
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from generate import generate_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

config = Config()
model = Llama(config).to(device)  
model.eval()

# reset cache before inference
for layer in model.layers:
    layer.groupedattention.init_cache(max_seq_len=512, device=device) 

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "World Cup"
max_new_tokens = 50

print(generate_inference(model, text, max_new_tokens, tokenizer, device=device))