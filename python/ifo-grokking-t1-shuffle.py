
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

# 3. 實驗主循環
d12_table = get_d12_table()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("RUNNING TRY 1: INDEX SHUFFLE ABLATION (5 SEEDS)")
print("="*60)

for shuffle_seed in range(1, 6):
    # 生成隨機索引映射
    np.random.seed(shuffle_seed)
    shuffle_map = np.random.permutation(24)
    inverse_map = np.zeros(24, dtype=int)
    for original_id, shuffled_id in enumerate(shuffle_map):
        inverse_map[shuffled_id] = original_id
        
    # 建立打亂後的數據集
    X, Y = [], []
    for i in range(24):
        for j in range(24):
            shuffled_i = shuffle_map[i]
            shuffled_j = shuffle_map[j]
            shuffled_k = shuffle_map[d12_table[i, j]]
            X.append([shuffled_i, shuffled_j])
            Y.append(shuffled_k)
            
    X = torch.tensor(X, dtype=torch.long).to(device)
    Y = torch.tensor(Y, dtype=torch.long).to(device)
    
    # 劃分 Train/Val (70/30, 固定種子 42 確保分割一致)
    np.random.seed(42)
    indices = np.random.permutation(576)
    train_idx = indices[:403]
    val_idx = indices[403:]
    
    # 重新初始化網絡
    torch.manual_seed(shuffle_seed)
    model = GroupMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    # 訓練到 Step 500
    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(X[train_idx])
        loss = criterion(out, Y[train_idx])
        loss.backward()
        optimizer.step()
        
    # 在 Step 500 評估
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1).cpu().numpy()
        
    # 計算真實的 Parity Accuracy (還原打亂後的 ID，判斷是否原本 >= 12)
    correct_parity = 0
    for idx in range(576):
        pred_shuffled = preds[idx]
        pred_original = inverse_map[pred_shuffled]
        
        true_shuffled = Y[idx].item()
        true_original = inverse_map[true_shuffled]
        
        pred_parity = pred_original >= 12
        true_parity = true_original >= 12
        
        if pred_parity == true_parity:
            correct_parity += 1
            
    parity_acc = correct_parity / 576
    
    # 計算 Val Accuracy
    val_preds = preds[val_idx]
    val_trues = Y[val_idx].cpu().numpy()
    val_acc = (val_preds == val_trues).mean()
    
    print(f"Shuffle Seed {shuffle_seed} | Val Acc: {val_acc:.1%} | Step 500 Parity Acc: {parity_acc:.1%}")

print("="*60)
