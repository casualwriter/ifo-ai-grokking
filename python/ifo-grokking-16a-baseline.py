
"""
IFO Grokking Experiment V16+: CCC Random Baseline & Degeneracy Test
Measures CCC and prediction entropy across 100 randomly initialized networks.

CK Hung & Echo, 2026/6/2
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

# D12 Setup
def get_d12_table_and_inverses():
    table = np.zeros((24, 24), dtype=np.int64)
    inverses = np.zeros(24, dtype=np.int64)
    for i in range(24):
        s1 = i // 12; r1 = i % 12
        if s1 == 0:
            inverses[i] = (12 - r1) % 12
        else:
            inverses[i] = i
            
        for j in range(24):
            s2 = j // 12; r2 = j % 12
            if s1 == 0 and s2 == 0:
                out_s = 0; out_r = (r1 + r2) % 12
            elif s1 == 0 and s2 == 1:
                out_s = 1; out_r = (-r1 + r2) % 12
            elif s1 == 1 and s2 == 0:
                out_s = 1; out_r = (r1 + r2) % 12
            else:
                out_s = 0; out_r = (-r1 + r2) % 12
            table[i, j] = out_s * 12 + out_r
    return table, inverses

class DihedralNet(nn.Module):
    def __init__(self, num_elements=24, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_elements, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_elements)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        emb_a = self.embed(x[:, 0])
        emb_b = self.embed(x[:, 1])
        h = torch.cat([emb_a, emb_b], dim=1)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out

def test_baseline(num_seeds=100):
    table, inverses = get_d12_table_and_inverses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cccs = []
    step1_uniques = []
    step2_uniques = []
    step2_entropies = []
    
    all_pairs = torch.tensor([[a, b] for a in range(24) for b in range(24)], dtype=torch.long, device=device)
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        model = DihedralNet().to(device)
        model.eval()
        
        with torch.no_grad():
            # Step 1
            logits1 = model(all_pairs)
            preds1 = logits1.argmax(dim=1).cpu().numpy()
            
            # Step 2 inputs
            step2_inputs = []
            for idx, (a, b) in enumerate(zip(all_pairs[:, 0].cpu().numpy(), all_pairs[:, 1].cpu().numpy())):
                true_ab = table[a, b]
                inv_ab = inverses[true_ab]
                step2_inputs.append([preds1[idx], inv_ab])
                
            step2_inputs = torch.tensor(step2_inputs, dtype=torch.long, device=device)
            logits2 = model(step2_inputs)
            preds2 = logits2.argmax(dim=1).cpu().numpy()
            
            # Metrics
            ccc = np.sum(preds2 == 0) / 576.0
            cccs.append(ccc)
            
            u1 = len(np.unique(preds1))
            u2 = len(np.unique(preds2))
            step1_uniques.append(u1)
            step2_uniques.append(u2)
            
            # Calculate prediction distribution entropy for Step 2
            _, counts = np.unique(preds2, return_counts=True)
            pk = counts / len(preds2)
            s2_entropy = entropy(pk)
            s2_uniques = len(np.unique(preds2))

            
            
    print(f"=== Random Baseline Results ({num_seeds} Seeds) ===")
    print(f"CCC:         mean = {np.mean(cccs):.4f} | std = {np.std(cccs):.4f} | min = {np.min(cccs):.4f} | max = {np.max(cccs):.4f}")
    print(f"S1 Uniques:  mean = {np.mean(step1_uniques):.2f} | min = {np.min(step1_uniques)} | max = {np.max(step1_uniques)}")
    print(f"S2 Uniques:  mean = {np.mean(step2_uniques):.2f} | min = {np.min(step2_uniques)} | max = {np.max(step2_uniques)}")
    print(f"S2 Entropy:  mean = {np.mean(step2_entropies):.4f} | max possible = {np.log(24):.4f}")

if __name__ == "__main__":
    test_baseline(100)
