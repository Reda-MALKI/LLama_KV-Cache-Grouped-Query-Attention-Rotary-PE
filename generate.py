from model import Llama
from model import Config
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn

config = Config()
model = Llama(config)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 3e-4)
print(model.eval())
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "World Cup"
text_tokenized = tokenizer.encode(text)
print(text_tokenized)

def generate_inference(model, prompt, max_tokens, tokenizer, device):
    for layer in model.layers:
        layer.groupedattention.init_cache(max_seq_len=512 , device=device)
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Step 1 — prefill
    tokens_cond = tokens[:, -128:]
    with torch.no_grad():
        logits = model(tokens_cond, start_pos=0, use_cache=True)
    
    start_pos = tokens_cond.shape[1]          
    probs = F.softmax(logits[:, -1, :], dim=-1)  
    next_token = torch.multinomial(probs, num_samples=1).to(device)
    tokens = torch.cat((tokens, next_token), dim=1) 

    # Step 2 — decode loop
    for _ in range(max_tokens - 1):
        tokens_cond = tokens[:, -128:]
        with torch.no_grad():
            logits = model(next_token, start_pos=start_pos, use_cache=True)  
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        start_pos += 1
        next_token = torch.multinomial(probs, num_samples=1).to(device)
        tokens = torch.cat((tokens, next_token), dim=1)  

    return tokenizer.decode(tokens[0].tolist()) 