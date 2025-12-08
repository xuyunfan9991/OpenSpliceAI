import os, csv
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openspliceai.predict.predict import load_pytorch_models, prepare_rbp_tensor
from openspliceai.constants import SL

model_path = "/home1/xyf/project/github/OpenSpliceAI/runs/shared_film_train/SpliceAI_shared_film_train2_400_0_rs42/0/models/model_best.pt"
feature_dir = "/home1/xyf/data/openspliceai_tissue_data/tissue_feature"
tissue_files = {
    "blood": "blood_features.json",
    "craniofacial": "craniofacial_features.json",
    "endoderm": "endoderm_features.json",
    "endothelium": "endothelium_features.json",
    "epidermis": "epidermis_features.json",
    "head_mesoderm": "head_mesoderm_features.json",
    "IM": "IM_features.json",
    "limb": "limb_features.json",
    "neural_progenitor": "neural_progenitor_features.json",
    "neuron": "neuron_features.json",
    "schwann": "schwann_features.json",
    "sensory_neuron": "sensory_neuron_features.json",
    "somatic_LPM": "somatic_LPM_features.json",
    "somite": "somite_features.json",
    "splanchnic_LPM": "splanchnic_LPM_features.json",
}

paths = {k: os.path.join(feature_dir, v) for k, v in tissue_files.items()}
missing = [k for k, p in paths.items() if not os.path.exists(p)]
if missing:
    raise SystemExit(f"Missing feature files: {missing}")

# 先 import torch 再 import numpy，可避免 OMP 共享内存报错
device = torch.device("cpu")
models, params = load_pytorch_models(model_path, device, SL=SL, CL=400)
model = models[0]
if not getattr(model, "expression_film", None):
    raise SystemExit("Loaded model has no FiLM conditioning; cannot compute gamma/beta.")

records = []
for label, path in paths.items():
    rbp_tensor = prepare_rbp_tensor(models, rbp_expression_path=path)
    with torch.no_grad():
        gamma, beta = model.expression_film(rbp_tensor)
    strength = getattr(model, "film_strength", 1.0)
    if strength != 1.0:
        gamma = 1.0 + (gamma - 1.0) * strength
        beta = beta * strength
    gamma_vec = gamma[0].squeeze(-1).cpu().numpy()
    beta_vec = beta[0].squeeze(-1).cpu().numpy()
    records.append((label, gamma_vec, beta_vec))

# 拼接 gamma+beta 做 PCA
mat = np.stack([np.concatenate([g, b]) for _, g, b in records])
mat_std = StandardScaler().fit_transform(mat)
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(mat_std)
explained = pca.explained_variance_ratio_

os.makedirs("results", exist_ok=True)
ts_path = "results/film_gamma_beta_vectors.tsv"
with open(ts_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["tissue", "gamma_var", "beta_var", "pc1", "pc2"])
    for (label, gamma_vec, beta_vec), coord in zip(records, pca_coords):
        writer.writerow([
            label,
            float(np.var(gamma_vec)),
            float(np.var(beta_vec)),
            float(coord[0]),
            float(coord[1]),
        ])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="lightgray", linewidth=0.8)
ax.axvline(0, color="lightgray", linewidth=0.8)
for (label, _, _), (x, y) in zip(records, pca_coords):
    ax.scatter(x, y, s=35)
    ax.text(x + 0.03, y + 0.03, label.replace("_", " "), fontsize=8)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
ax.set_title("FiLM γ/β PCA across tissues")
fig.tight_layout()
plot_path = "results/film_gamma_beta_pca.png"
fig.savefig(plot_path, dpi=200)

print(f"Saved PCA plot to {plot_path}")
print(f"Saved vectors/variance table to {ts_path}")
print("Explained variance ratios:", explained)
sorted_by_gamma = sorted(records, key=lambda r: np.var(r[1]), reverse=True)
print("Top 5 gamma variance:")
for label, g, _ in sorted_by_gamma[:5]:
    print(label, f"var={np.var(g):.4f}")
