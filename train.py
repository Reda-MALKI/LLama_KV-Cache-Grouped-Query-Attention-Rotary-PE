from model import Llama
from model import Config
from inference import generate_inference
from generate import tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from generate import 

# def train(model , epochs):
#     for epoch in range(epochs):
#         for x,y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
#             logits = model(x , start_pos = None , use_cache = False)
#             B , T , V = logits.shape
#             y_pred = logits.contiguous().view(B*T , V)
#             y = y.contiguous().view(B*T)
#             loss = 

# In this train.py file complete your trinaing pipline with your training loader
