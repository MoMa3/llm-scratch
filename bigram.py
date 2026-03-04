import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_intervals = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32


torch.manual_seed(1337)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train = data[:n]
val = data[n:]


def get_batch(split):
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # 65 x 32
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        #  idx = input
        #  target = expected output

        #  B = batch size
        #  T = time (context length)
        #  C = n_embed
        tok_emb = self.token_embedding_table(idx)
        logits = self.lm_head(tok_emb)  # B,T,vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            B, T = targets.shape
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx = (B,T)
        for _ in range(max_new_tokens):
            # get forward method
            logits, loss = self(idx)
            # focus only on last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to running seq
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BLM()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


batch_size = 32
for iter in range(max_iters):
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print(f"step {iter} train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
