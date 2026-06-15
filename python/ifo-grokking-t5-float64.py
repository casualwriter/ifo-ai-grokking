
import torch
import torch.nn as nn
import numpy as np
import random
import sys

# 強制不緩衝輸出，確保 print 立即顯示
sys.stdout.reconfigure(line_buffering=True)

# ==========================================
# 1. 核心配置：CPU 優化與強制雙精度 (Float64)
# ==========================================
DEVICE = torch.device("cpu") # 強制 CPU 運行
DTYPE = torch.float64        # 強制雙精度，對抗 NFI 質疑

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==========================================
# 2. 嚴格生成 D12 群乘法表與逆元表
# ==========================================
# 0-11: 旋轉 r^0 到 r^11
# 12-23: 反射 s*r^0 到 s*r^11
table = np.zeros((24, 24), dtype=np.int64)
for i in range(12):
    for j in range(12):
        table[i, j] = (i + j) % 12
        table[i, 12 + j] = 12 + (j - i) % 12
        table[12 + i, j] = 12 + (i + j) % 12
        table[12 + i, 12 + j] = (j - i) % 12

inverse_map = np.zeros(24, dtype=np.int64)
for i in range(12):
    inverse_map[i] = (12 - i) % 12  # r^i 的逆是 r^-i
    inverse_map[12 + i] = 12 + i    # s*r^i 的逆是它自己

# ==========================================
# 3. 雙精度 MLP 模型定義
# ==========================================
class DoubleMLP(nn.Module):
    def __init__(self, num_classes=24, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch, 2]
        emb = self.embed(x).view(x.size(0), -1) # Concat
        out = self.relu(self.fc1(emb))
        out = self.fc2(out)
        return out

# ==========================================
# 4. 向量化 CCC 計算 (CPU 飛速版)
# ==========================================
def compute_ccc_vectorized(model, inverse_map):
    model.eval()
    with torch.no_grad():
        # 一次性生成所有 576 個輸入組合
        all_a = []
        all_b = []
        for a in range(24):
            for b in range(24):
                all_a.append(a)
                all_b.append(b)
        
        inputs = torch.tensor(list(zip(all_a, all_b)), device=DEVICE, dtype=torch.long)
        pred_c = model(inputs).argmax(dim=-1) # [576]
        
        # 取得 b 的逆元
        b_invs = torch.tensor([inverse_map[b] for b in all_b], device=DEVICE, dtype=torch.long)
        
        # 第二步預測: Net(pred_c, b^-1)
        inputs_inv = torch.stack([pred_c, b_invs], dim=1) # [576, 2]
        pred_a = model(inputs_inv).argmax(dim=-1) # [576]
        
        target_a = torch.tensor(all_a, device=DEVICE, dtype=torch.long)
        correct_cycles = (pred_a == target_a).sum().item()
        
        return correct_cycles / 576.0

# ==========================================
# 5. 完整訓練與觀測循環
# ==========================================
def run_control_experiment(seed=42, use_wd=True):
    set_seed(seed)
    
    # 準備數據集 (70/30 固定劃分)
    pairs = []
    for a in range(24):
        for b in range(24):
            pairs.append((a, b, table[a, b]))
            
    random.seed(42) # 固定劃分 Seed
    random.shuffle(pairs)
    
    train_pairs = pairs[:403]
    val_pairs = pairs[403:]
    
    train_x = torch.tensor([[p[0], p[1]] for p in train_pairs], dtype=torch.long, device=DEVICE)
    train_y = torch.tensor([p[2] for p in train_pairs], dtype=torch.long, device=DEVICE)
    
    val_x = torch.tensor([[p[0], p[1]] for p in val_pairs], dtype=torch.long, device=DEVICE)
    val_y = torch.tensor([p[2] for p in val_pairs], dtype=torch.long, device=DEVICE)
    
    # 初始化雙精度模型
    model = DoubleMLP().to(DEVICE).to(DTYPE)
    
    wd = 0.2 if use_wd else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    print("=" * 85)
    print(f" 運行設定: DEVICE={DEVICE.type.upper()} | DTYPE={DTYPE} | Weight Decay={wd}")
    print("=" * 85)
    print(f"{'Step':<8} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Acc':<8} | {'CCC':<8} | {'Emb Norm':<8} | {'FC1 Norm':<8}")
    print("-" * 85)
    
    for step in range(1, 25001):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        
        # 打印策略：前期每 1000 步，靠近 22k 崩塌區間每 100 步
        is_print_step = (
            (step % 1000 == 0) or 
            (21000 <= step <= 23500 and step % 100 == 0) or 
            step == 1 or 
            step == 500
        )
        
        if is_print_step:
            model.eval()
            with torch.no_grad():
                # 計算 Train Acc
                train_preds = outputs.argmax(dim=-1)
                train_acc = (train_preds == train_y).float().mean().item()
                
                # 計算 Val Acc
                val_outputs = model(val_x)
                val_preds = val_outputs.argmax(dim=-1)
                val_acc = (val_preds == val_y).float().mean().item()
                
                # 計算 CCC (向量化)
                ccc = compute_ccc_vectorized(model, inverse_map)
                
                # 計算權重範數
                emb_norm = model.embed.weight.norm().item()
                fc1_norm = model.fc1.weight.norm().item()
                
                print(f"{step:<8} | {loss.item():<10.6f} | {train_acc:<9.4f} | {val_acc:<8.4f} | {ccc:<8.4f} | {emb_norm:<8.2f} | {fc1_norm:<8.2f}")

if __name__ == "__main__":
    # 執行預設實驗 (WD=0.2, Float64)
    run_control_experiment(seed=42, use_wd=True)