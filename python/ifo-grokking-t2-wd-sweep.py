
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 定義 D12 群乘法表 (24x24)
# 0-11 是旋轉 (r^0 到 r^11), 12-23 是反射 (sr^0 到 sr^11)
def get_d12_table():
    table = np.zeros((24, 24), dtype=int)
    for i in range(24):
        for j in range(24):
            # 拆解 a
            a_is_ref = i >= 12
            a_rot = i % 12
            # 拆解 b
            b_is_ref = j >= 12
            b_rot = j % 12
            
            if not a_is_ref and not b_is_ref:
                c_rot = (a_rot + b_rot) % 12
                c_ref = False
            elif not a_is_ref and b_is_ref:
                c_rot = (b_rot - a_rot) % 12
                c_ref = True
            elif a_is_ref and not b_is_ref:
                c_rot = (a_rot + b_rot) % 12
                c_ref = True
            else: # a_is_ref and b_is_ref
                c_rot = (b_rot - a_rot) % 12
                c_ref = False
                
            c = c_rot + (12 if c_ref else 0)
            table[i, j] = c
    return table

# 2. 定義簡單 MLP
class GroupMLP(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(24, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 24)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [batch, 2]
        emb1 = self.embed(x[:, 0])
        emb2 = self.embed(x[:, 1])
        out = torch.cat([emb1, emb2], dim=1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

d12_table = get_d12_table()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立標準（未打亂）數據集
X, Y = [], []
for i in range(24):
    for j in range(24):
        X.append([i, j])
        Y.append(d12_table[i, j])
X = torch.tensor(X, dtype=torch.long).to(device)
Y = torch.tensor(Y, dtype=torch.long).to(device)

np.random.seed(42)
indices = np.random.permutation(576)
train_idx = indices[:403]
val_idx = indices[403:]

print("="*60)
print("RUNNING TRY 2: HIGH WD (0.4) COMPRESSION SWEEP")
print("="*60)

for seed in [1, 2]:
    torch.manual_seed(seed)
    model = GroupMLP().to(device)
    # 設置高 Weight Decay = 0.4
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.4)
    criterion = nn.CrossEntropyLoss()
    
    collapse_step = -1
    max_val_acc = 0.0
    
    for step in range(1, 15001):
        model.train()
        optimizer.zero_grad()
        out = model(X[train_idx])
        loss = criterion(out, Y[train_idx])
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_preds = model(X[val_idx]).argmax(dim=1)
                val_acc = (val_preds == Y[val_idx]).float().mean().item()
                
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                
            # 檢測崩塌：如果曾經達到過 90% 以上，後來掉到 80% 以下
            if max_val_acc > 0.90 and val_acc < 0.80 and collapse_step == -1:
                collapse_step = step
                
            if step % 2000 == 0:
                print(f"Seed {seed} | Step {step:5d} | Val Acc: {val_acc:.1%} | Max Val Acc: {max_val_acc:.1%}")
                
    print(f"--> Seed {seed} Finished. Max Val Acc: {max_val_acc:.1%}. Collapse detected at Step: {collapse_step if collapse_step != -1 else 'No Collapse'}")
    print("-" * 40)
