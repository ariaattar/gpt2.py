import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import time
from safetensors.torch import save_file, load_file
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
        self.encoded = torch.tensor(self.encode(text), dtype=torch.long)
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def __len__(self):
        return max(0, len(self.encoded) - self.block_size)
    
    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.block_size]
        target = self.encoded[idx + 1:idx + self.block_size + 1]
        return chunk, target

def load_dataset(block_size, batch_size):
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    train_text = text[:int(0.9 * len(text))]
    val_text = text[int(0.9 * len(text)):]
    
    train_dataset = TextDataset(train_text, block_size)
    val_dataset = TextDataset(val_text, block_size)
    
    vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, vocab_size

def decode_tokens(tokens, dataset):
    return dataset.decode(tokens)

class MLP(nn.Module):

    def __init__(self, n_embd, ffwd_dim_mult):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffwd_dim_mult * n_embd),
            nn.GELU(),
            nn.Linear(ffwd_dim_mult * n_embd, n_embd),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)
    

class Attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size, head_size)
        self.lamb = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, v1=None):
        B,T,C = x.shape 
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        
        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        
        out = w @ v
        return self.proj(out), v1



class FlashAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(head_size, head_size)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.k(x)
        q = self.q(x) 
        v = self.v(x)
        
        k = k.unsqueeze(1)
        q = q.unsqueeze(1)
        v = v.unsqueeze(1)
        

        with torch.backends.cuda.sdp_kernel(enable_math=True):  
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout if self.training else 0.0,
                is_causal=True
            )
        out = out.squeeze(1)
        return self.proj(out)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, attention_type = 'base'):
        super().__init__()
        self.heads = nn.ModuleList([Attention(head_size) for _ in range(num_heads)]) if attention_type == 'base' else nn.ModuleList([FlashAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, ffwd_dim_mult, attention_type = 'flash_attn'):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, attention_type)
        self.mlp = MLP(n_embd, ffwd_dim_mult)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, ffwd_dim_mult, attention_type) for _ in range(n_blocks)])
        self.enc = nn.Embedding(vocab_size, n_embed)
        self.pos = nn.Embedding(block_size, n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.enc(idx)
        pos_emb = self.pos(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train(model, train_loader, val_loader, optimizer, num_epochs=3, max_steps=None):
    best_val_loss = float('inf')
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()
    if max_steps != None:
        total_steps = max_steps
    else:
        total_steps = num_epochs * len(train_loader)
    
    current_step = 0
    
    def evaluate(model, val_loader):
        model.eval()
        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits, loss = model(data, targets)
                total_val_loss += loss.item()
                val_steps += 1
        return total_val_loss / val_steps

    while current_step <= total_steps:
        last_step = (current_step == total_steps)
        
        if last_step or (current_step > 0 and current_step % eval_interval == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            
            val_loss = evaluate(model, val_loader)
            
            print(f'step:{current_step}/{total_steps} val_loss:{val_loss:.4f} '
                  f'train_time:{training_time_ms:.0f}ms '
                  f'step_avg:{training_time_ms/max(1,current_step-1):.2f}ms')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_file(model.state_dict(), 'best_model.safetensors')
                torch.save({
                    'step': current_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model_optimizer.pt')
                print("Saved new best model!")
            
            torch.cuda.synchronize()
            t0 = time.time()
            
        if last_step:
            break
            
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            logits, loss = model(data, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(f"step:{current_step+1}/{total_steps} train_loss:{train_loss:.4f} "
                  f"train_time:{approx_time:.0f}ms "
                  f"step_avg:{approx_time/max(1,current_step):.2f}ms")
            
            current_step += 1
            
            if current_step >= total_steps:
                break

    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    return model

def main(
    n_embed=64,
    n_heads=4,
    block_size=100,
    n_blocks=4,
    batch_size=32,
    learning_rate=1e-3,
    ffwd_dim_mult=4,
    dropout=0.0,
    num_epochs=3,
    max_steps=100,
    use_checkpoint=True,
    optimizer_type='adam',
    attention_type = 'flash_attn',
    seed=42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    globals().update(locals())
    
    global train_loader, val_loader, train_dataset, vocab_size
    train_loader, val_loader, train_dataset, vocab_size = load_dataset(block_size, batch_size)
    
    model = GPT().to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    if use_checkpoint:
        if os.path.exists('best_model.safetensors'):
            model.load_state_dict(load_file('best_model.safetensors'))
            if os.path.exists('best_model_optimizer.pt'):
                optimizer_checkpoint = torch.load('best_model_optimizer.pt')
                optimizer.load_state_dict(optimizer_checkpoint['optimizer_state_dict'])
            print("Loaded previous checkpoint!")
    
    try:
        train(model, train_loader, val_loader, optimizer, num_epochs, max_steps)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    return model
