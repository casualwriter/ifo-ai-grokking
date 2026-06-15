
# ifo-grokking-t4-wd-sweep-extended.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define D12 Group Structure
def get_d12_table():
    table = np.zeros((24, 24), dtype=int)
    for i in range(24):
        for j in range(24):
            a_is_ref = i >= 12
            b_is_ref = j >= 12
            a_val = i % 12
            b_val = j % 12
            
            if not a_is_ref and not b_is_ref:
                res = (a_val + b_val) % 12
            elif not a_is_ref and b_is_ref:
                res = 12 + (b_val - a_val) % 12
            elif a_is_ref and not b_is_ref:
                res = 12 + (a_val + b_val) % 12
            else:
                res = (b_val - a_val) % 12
            table[i, j] = res
    return table

D12_TABLE = get_d12_table()
D12_INV = np.zeros(24, dtype=int)
for i in range(24):
    for j in range(24):
        if D12_TABLE[i, j] == 0:
            D12_INV[i] = j

# 2. Model
class GroupMLP(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(24, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 24)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        emb_a = self.emb(x[:, 0])
        emb_b = self.emb(x[:, 1])
        x_concat = torch.cat([emb_a, emb_b], dim=1)
        return self.fc2(self.relu(self.fc1(x_concat)))

def evaluate_metrics(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    acc = np.mean(preds == targets.cpu().numpy())
    
    # CCC
    ccc_correct = 0
    for idx in range(len(inputs)):
        c_prime = preds[idx]
        c_orig = targets[idx].item()
        c_inv = D12_INV[c_orig]
        
        test_input = torch.tensor([[c_prime, c_inv]], device=inputs.device)
        with torch.no_grad():
            test_pred = torch.argmax(model(test_input), dim=1).item()
        if test_pred == 0:
            ccc_correct += 1
    ccc = ccc_correct / len(inputs)
    
    # Weight Norms
    emb_norm = torch.norm(model.emb.weight, p='fro').item()
    fc1_norm = torch.norm(model.fc1.weight, p='fro').item()
    
    return acc, ccc, emb_norm, fc1_norm

# Setup Dataset
inputs_all = []
targets_all = []
for i in range(24):
    for j in range(24):
        inputs_all.append([i, j])
        targets_all.append(D12_TABLE[i, j])
inputs_all = np.array(inputs_all)
targets_all = np.array(targets_all)

np.random.seed(42)
indices = np.random.permutation(576)
train_size = int(0.7 * 576)
train_idx, val_idx = indices[:train_size], indices[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_in_tensor = torch.tensor(inputs_all[train_idx], dtype=torch.long, device=device)
train_tar_tensor = torch.tensor(targets_all[train_idx], dtype=torch.long, device=device)
val_in_tensor = torch.tensor(inputs_all[val_idx], dtype=torch.long, device=device)
val_tar_tensor = torch.tensor(targets_all[val_idx], dtype=torch.long, device=device)

# ============================================================
# RUN SWEEP FOR WD=0.4 UP TO 25,000 STEPS
# ============================================================
print("="*75)
print("RUNNING TEST 3: EXTENDED WD=0.4 SWEEP UP TO 25,000 STEPS (2 SEEDS)")
print("="*75)

for seed in [1, 2]:
    print(f"\n--- Starting Seed {seed} ---")
    torch.manual_seed(seed)
    model = GroupMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.4)
    criterion = nn.CrossEntropyLoss()
    
    max_val_acc = 0.0
    collapse_detected_step = "No Collapse"
    
    for step in range(1, 25001):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_in_tensor)
        loss = criterion(outputs, train_tar_tensor)
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0 or step == 500:
            val_acc, ccc, emb_n, fc1_n = evaluate_metrics(model, val_in_tensor, val_tar_tensor)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            
            # Simple collapse detection: if val_acc drops by more than 10% from its historic max
            if max_val_acc > 0.90 and val_acc < (max_val_acc - 0.10) and collapse_detected_step == "No Collapse":
                collapse_detected_step = f"Detected at Step {step} (Acc: {val_acc:.1%} vs Max: {max_val_acc:.1%})"
                
            print(f"Step {step:5d} | Val Acc: {val_acc:6.1%} | CCC: {ccc:6.1%} | Emb Norm: {emb_n:5.1f} | FC1 Norm: {fc1_n:5.1f}")
            
    print(f"--> Seed {seed} Finished. Max Val Acc: {max_val_acc:.1%}. Collapse: {collapse_detected_step}")
print("="*75)
