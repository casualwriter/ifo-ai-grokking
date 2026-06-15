
# ifo-grokking-t3-shuffle-baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define D12 Group Structure
def get_d12_table():
    # 0-11: rotations (r^0 to r^11), 12-23: reflections (s to sr^11)
    table = np.zeros((24, 24), dtype=int)
    for i in range(24):
        for j in range(24):
            a_is_ref = i >= 12
            b_is_ref = j >= 12
            a_val = i % 12
            b_val = j % 12
            
            if not a_is_ref and not b_is_ref:
                # rot * rot = rot
                res = (a_val + b_val) % 12
            elif not a_is_ref and b_is_ref:
                # rot * ref = ref
                res = 12 + (b_val - a_val) % 12
            elif a_is_ref and not b_is_ref:
                # ref * rot = ref
                res = 12 + (a_val + b_val) % 12
            else:
                # ref * ref = rot
                res = (b_val - a_val) % 12
            table[i, j] = res
    return table

D12_TABLE = get_d12_table()
D12_INV = np.zeros(24, dtype=int)
for i in range(24):
    for j in range(24):
        if D12_TABLE[i, j] == 0:  # 0 is identity
            D12_INV[i] = j

# 2. Network Architecture
class GroupMLP(nn.Module):
    def __init__(self, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(24, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 24)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, 2]
        emb_a = self.emb(x[:, 0])
        emb_b = self.emb(x[:, 1])
        x_concat = torch.cat([emb_a, emb_b], dim=1)
        h = self.relu(self.fc1(x_concat))
        return self.fc2(h)

# 3. Helper functions for evaluation
def get_shuffled_map(shuffle_seed):
    np.random.seed(shuffle_seed)
    perm = np.random.permutation(24)
    inv_perm = np.zeros(24, dtype=int)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return perm, inv_perm

def evaluate_metrics(model, inputs, targets, perm, inv_perm):
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    # Map back to original elements to compute algebraic metrics
    preds_orig = inv_perm[preds]
    targets_orig = inv_perm[targets.cpu().numpy()]
    inputs_orig = inv_perm[inputs.cpu().numpy()]
    
    # 1. Accuracy
    acc = np.mean(preds == targets.cpu().numpy())
    
    # 2. Parity Accuracy (Rotation < 12 vs Reflection >= 12)
    pred_parity = preds_orig >= 12
    true_parity = targets_orig >= 12
    parity_acc = np.mean(pred_parity == true_parity)
    
    # 3. Cycle Closure Count (CCC)
    # Test if Net(c', c^-1) == e
    ccc_correct = 0
    for idx in range(len(inputs)):
        c_prime = preds_orig[idx]
        c_orig = targets_orig[idx]
        c_inv = D12_INV[c_orig]
        
        # Map back to shuffled tokens to feed into network
        c_prime_shuf = perm[c_prime]
        c_inv_shuf = perm[c_inv]
        
        test_input = torch.tensor([[c_prime_shuf, c_inv_shuf]], device=inputs.device)
        with torch.no_grad():
            test_pred = torch.argmax(model(test_input), dim=1).item()
        test_pred_orig = inv_perm[test_pred]
        
        if test_pred_orig == 0:  # 0 is identity
            ccc_correct += 1
            
    ccc = ccc_correct / len(inputs)
    return acc, parity_acc, ccc

# ============================================================
# PART 1: Run 100 Untrained Models for Baseline
# ============================================================
print("="*60)
print("RUNNING TEST 1: 100 UNTRAINED MODELS BASELINE")
print("="*60)

# Generate full dataset
inputs_all = []
targets_all = []
for i in range(24):
    for j in range(24):
        inputs_all.append([i, j])
        targets_all.append(D12_TABLE[i, j])
inputs_all = np.array(inputs_all)
targets_all = np.array(targets_all)

# Train/Val Split (70/30, seed 42)
np.random.seed(42)
indices = np.random.permutation(576)
train_size = int(0.7 * 576)
train_idx, val_idx = indices[:train_size], indices[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

baseline_parity_accs = []
baseline_cccs = []

for shuffle_seed in range(1, 6):
    perm, inv_perm = get_shuffled_map(shuffle_seed)
    
    # Map dataset to shuffled tokens
    shuf_inputs = perm[inputs_all]
    shuf_targets = perm[targets_all]
    
    val_in_tensor = torch.tensor(shuf_inputs[val_idx], dtype=torch.long, device=device)
    val_tar_tensor = torch.tensor(shuf_targets[val_idx], dtype=torch.long, device=device)
    
    for init_seed in range(20):  # 20 seeds per shuffle configuration = 100 runs total
        torch.manual_seed(init_seed)
        model = GroupMLP().to(device)
        _, parity_acc, ccc = evaluate_metrics(model, val_in_tensor, val_tar_tensor, perm, inv_perm)
        baseline_parity_accs.append(parity_acc)
        baseline_cccs.append(ccc)

print(f"Untrained Baseline (N=100) over 5 Shuffled Encodings:")
print(f"-> Parity Accuracy Chance Level: {np.mean(baseline_parity_accs):.2%} ± {np.std(baseline_parity_accs):.2%}")
print(f"-> CCC Chance Level:             {np.mean(baseline_cccs):.2%} ± {np.std(baseline_cccs):.2%}")
print("\n")

# ============================================================
# PART 2: Train 5 Shuffled Seeds to Step 500
# ============================================================
print("="*60)
print("RUNNING TEST 2: STEP 500 SHUFFLED INDEX TRAIN (5 SEEDS)")
print("="*60)

for shuffle_seed in range(1, 6):
    perm, inv_perm = get_shuffled_map(shuffle_seed)
    
    # Map dataset to shuffled tokens
    shuf_inputs = perm[inputs_all]
    shuf_targets = perm[targets_all]
    
    train_in_tensor = torch.tensor(shuf_inputs[train_idx], dtype=torch.long, device=device)
    train_tar_tensor = torch.tensor(shuf_targets[train_idx], dtype=torch.long, device=device)
    val_in_tensor = torch.tensor(shuf_inputs[val_idx], dtype=torch.long, device=device)
    val_tar_tensor = torch.tensor(shuf_targets[val_idx], dtype=torch.long, device=device)
    
    # Train model
    torch.manual_seed(shuffle_seed + 100)  # Distinct init seed
    model = GroupMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(500):
        optimizer.zero_grad()
        outputs = model(train_in_tensor)
        loss = criterion(outputs, train_tar_tensor)
        loss.backward()
        optimizer.step()
        
    val_acc, parity_acc, ccc = evaluate_metrics(model, val_in_tensor, val_tar_tensor, perm, inv_perm)
    print(f"Shuffle Seed {shuffle_seed} | Step 500 | Val Acc: {val_acc:.1%} | Parity Acc: {parity_acc:.1%} | CCC: {ccc:.1%}")
print("="*60)
