import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

@dataclass
class Config():
    d_model:    int   = 768
    vocab_size: int   = 50257
    n_heads:    int   = 12
    n_kv_heads: int   = 4
    n_layers:   int   = 32
    eps:        float = 1e-5


class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, x):
        return self.embedding(x)


class RMSNorm(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.eps    = config.eps
        self.weight = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x):
        # RMSNorm is a simpler alternative to LayerNorm used in LLaMA
        # instead of computing mean and variance it only computes the root mean square of the input
        # this makes it faster and uses less memory while still stabilizing training
        # the formula is: x / sqrt( mean(x^2) + eps ) then scaled by a learnable weight
        # keepdim=True is critical here so the shape stays [B , T , 1] and broadcasts correctly against [B , T , d_model]
        # without keepdim the division would fail because the shapes would not align
        rmsx = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rmsx
        # the learnable weight gamma has shape [d_model] and rescales each feature independently
        # this gives the model the ability to undo the normalization if needed
        return self.weight * x


def compute_theta_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    # RoPE works by rotating pairs of values in the query and key vectors
    # each pair gets its own rotation angle that depends on both its dimension index and its position in the sequence
    # we need head_dim/2 angles in total because we work in pairs, so we step by 2
    theta_numerators = torch.arange(0, head_dim, 2).float()
    # shape is [head_dim/2]
    # this formula gives a geometric progression: low indices get high frequencies, high indices get low frequencies
    # so early dimensions rotate fast and later dimensions rotate slowly
    # this is the same intuition as sinusoidal encodings but applied as complex rotations
    theta_vals = 1.0 / (theta ** (theta_numerators / head_dim))
    # shape is still [head_dim/2]

    positions = torch.arange(seq_len)
    # shape is [seq_len]

    # outer product gives us a matrix where entry [m , i] = position_m * theta_i
    # meaning each token position gets its own set of rotation angles across all frequency dimensions
    freqs = torch.outer(positions, theta_vals)
    # shape is now [seq_len , head_dim/2]

    # we convert the angles to complex numbers using polar form: magnitude=1 and angle=freq
    # this gives us unit complex numbers e^(i * freq) = cos(freq) + i*sin(freq)
    # multiplying a complex number by this rotates it by the angle without changing its magnitude
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    # shape is [seq_len , head_dim/2] but now complex valued
    return freqs_complex


def apply_rotation(x: torch.Tensor, head_dim: int, seq_len: int, theta: float = 10000.0):
    # we reshape the last dimension into pairs of 2 then interpret each pair as a complex number
    # so head_dim real values become head_dim/2 complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape goes from [B , T , n_heads , head_dim] to [B , T , n_heads , head_dim/2] complex

    freqs_complex = compute_theta_frequencies(head_dim, seq_len, theta).to(x.device)
    # shape is [seq_len , head_dim/2] complex

    # we unsqueeze to make freqs_complex broadcastable with x_complex
    # after unsqueeze the shape becomes [1 , seq_len , 1 , head_dim/2]
    # the 1s align with the batch dimension and the head dimension so every head at every batch gets the same rotation
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # multiplying two complex numbers (a+ib)*(cos+i*sin) rotates the vector (a,b) by the angle
    # this is how RoPE injects positional information into Q and K without modifying the model architecture
    x_rotated = x_complex * freqs_complex
    # shape is still [B , T , n_heads , head_dim/2] complex

    # we convert back to real by viewing each complex number as two real values
    # then reshape back to the original shape [B , T , n_heads , head_dim]
    return torch.view_as_real(x_rotated).reshape(*x.shape)


def repeat_kv(x: torch.Tensor, n_reps: int):
    # in Grouped Query Attention we have fewer KV heads than Q heads
    # for example with n_heads=12 and n_kv_heads=4 each KV head is shared by 3 query heads
    # to compute attention we need K and V to have the same number of heads as Q
    # so we repeat each KV head group_size times along the head dimension
    # this is done logically using expand which does not allocate new memory
    B, T, H, C = x.shape
    # after expand shape is [B , T , n_kv_heads , group_size , head_dim]
    # after reshape shape becomes [B , T , n_kv_heads*group_size , head_dim] = [B , T , n_heads , head_dim]
    return (
        x[:, :, :, None, :]
        .expand(B, T, H, n_reps, C)
        .reshape(B, T, H * n_reps, C)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Config , max_seq_len : int =512):
        super().__init__()
        self.d_model    = config.d_model
        self.n_heads    = config.n_heads
        self.head_dim   = self.d_model // self.n_heads
        # n_kv_heads is smaller than n_heads, this is the core of Grouped Query Attention
        # instead of having one K and V per query head we have one K and V shared across a group of query heads
        # for example n_heads=12 and n_kv_heads=4 means every 3 query heads share the same K and V
        # this reduces the KV cache size by a factor of n_heads/n_kv_heads during inference
        self.n_kv_heads = config.n_kv_heads
        self.group_size = self.n_heads // self.n_kv_heads
        # Wq projects to the full query space: output shape [B , T , n_heads * head_dim]
        self.Wq = nn.Linear(self.d_model, self.n_heads    * self.head_dim, bias=False)
        # Wk and Wv project to the reduced KV space: output shape [B , T , n_kv_heads * head_dim]
        # this is smaller than Wq by a factor of n_kv_heads/n_heads
        self.Wk = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(self.d_model, self.d_model)
        # k_cache and v_cache store past K and V tensors during autoregressive generation
        # without KV cache we would recompute K and V for all previous tokens at every new step
        # with KV cache we compute K and V only for the new token and read the rest from memory
        # this reduces inference complexity from O(n^2) recomputation to O(n) per step

    def init_cache(self, max_seq_len: int, device):
        # we preallocate the full cache upfront with zeros
        # shape is [1 , n_kv_heads , max_seq_len , head_dim]
        # we use n_kv_heads not n_heads because the cache only stores the reduced KV heads
        # this is where the memory saving of GQA is most visible compared to standard MHA
        # in standard MHA the cache would be [1 , n_heads , max_seq_len , head_dim] which is much larger
        self.k_cache = torch.zeros(1, self.n_kv_heads, max_seq_len, self.head_dim).to(device)
        self.v_cache = torch.zeros(1, self.n_kv_heads, max_seq_len, self.head_dim).to(device)

    def forward(self, x, start_pos=None, use_cache=False):
        B, T, C = x.shape

        Q = self.Wq(x)  # shape [B , T , n_heads * head_dim]
        K = self.Wk(x)  # shape [B , T , n_kv_heads * head_dim]
        V = self.Wv(x)  # shape [B , T , n_kv_heads * head_dim]

        Q = Q.view(B, T, self.n_heads,    self.head_dim)  # shape [B , T , n_heads , head_dim]
        K = K.view(B, T, self.n_kv_heads, self.head_dim)  # shape [B , T , n_kv_heads , head_dim]
        V = V.view(B, T, self.n_kv_heads, self.head_dim)  # shape [B , T , n_kv_heads , head_dim]

        # we apply RoPE to Q and K so they encode positional information through rotation
        # we do not apply RoPE to V because values are not involved in the position-sensitive dot product
        # after rotation shapes remain the same but the values now carry positional context
        Q = apply_rotation(Q, self.head_dim, T)
        K = apply_rotation(K, self.head_dim, T)

        # transpose to move heads before sequence length for batched matrix multiplication
        Q = Q.transpose(1, 2)  # shape [B , n_heads    , T , head_dim]
        K = K.transpose(1, 2)  # shape [B , n_kv_heads , T , head_dim]
        V = V.transpose(1, 2)  # shape [B , n_kv_heads , T , head_dim]

        if not use_cache:
            # we expand K and V from n_kv_heads to n_heads using repeat_kv
            # transpose back to [B , T , n_kv_heads , head_dim] before repeat_kv because it expects that shape
            # then transpose again after to get [B , n_heads , T , head_dim] for attention computation
            K = repeat_kv(K.transpose(1, 2), self.group_size).transpose(1, 2)
            V = repeat_kv(V.transpose(1, 2), self.group_size).transpose(1, 2)
            # now K and V shape is [B , n_heads , T , head_dim] matching Q

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # shape is [B , n_heads , T , T]
            mask   = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))
            scores = F.softmax(scores, dim=-1)
            attn   = torch.matmul(scores, V)
            # shape is [B , n_heads , T , head_dim]

        else:
            # during inference we write the new K and V into the cache at position start_pos
            # start_pos tells us how many tokens have already been processed
            # so we write exactly T slots starting from start_pos
            self.k_cache[:, :, start_pos:start_pos + T, :] = K
            self.v_cache[:, :, start_pos:start_pos + T, :] = V

            # we read back all K and V up to and including the current token
            # this gives us the full context the attention can attend to
            K_cached = self.k_cache[:, :, :start_pos + T, :]
            V_cached = self.v_cache[:, :, :start_pos + T, :]
            # shape is [1 , n_kv_heads , start_pos+T , head_dim]

            # expand cached K and V from n_kv_heads to n_heads same as in the no-cache branch
            K_cached = repeat_kv(K_cached.transpose(1, 2), self.group_size).transpose(1, 2)
            V_cached = repeat_kv(V_cached.transpose(1, 2), self.group_size).transpose(1, 2)
            # shape is now [1 , n_heads , start_pos+T , head_dim]

            scores = torch.matmul(Q, K_cached.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # shape is [B , n_heads , T , start_pos+T]
            # note the second dimension is start_pos+T not T because Q attends over the full cached context
            mask   = torch.tril(torch.ones(T, K_cached.shape[2], device=x.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))
            scores = F.softmax(scores, dim=-1)
            attn   = torch.matmul(scores, V_cached)
            # shape is [B , n_heads , T , head_dim]

        # transpose back and merge all heads into d_model
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        # shape is [B , T , d_model]
        return self.Wo(attn)


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.SiLU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(self, x):
        return self.ff(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.rmsnorm          = RMSNorm(config)
        self.groupedattention = GroupedQueryAttention(config)
        self.ff               = FeedForward(config)

    def forward(self, x, start_pos=None, use_cache=False):
        rms               = self.rmsnorm(x)
        grouped_attention = self.groupedattention(rms, start_pos, use_cache)
        residual          = x + grouped_attention
        rms2              = self.rmsnorm(residual)
        ff                = self.ff(rms2)
        return residual + ff


class Llama(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embeddings = Embedding(config)
        self.layers     = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.rms = RMSNorm(config)
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, start_pos=0, use_cache=False):
        x = x.to(device)  
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, start_pos=start_pos, use_cache=use_cache)
        x = self.rms(x)
        return self.out(x)


# config = Config()
# model  = Llama(config).to(device)
# model.eval()

# x   = torch.randint(0, config.vocab_size, (1, 8)).to(device)
# out = model(x)
# print(out.shape)  # should be (1, 8, 50257)