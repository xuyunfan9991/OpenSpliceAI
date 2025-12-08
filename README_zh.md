# OpenSpliceAI 中文工作流指南

OpenSpliceAI 是在 PyTorch 中重构的 SpliceAI，实现了从原始基因组数据到剪接位点预测/变异注释的全流程。本指南以“从零到组织特异模型”的视角，覆盖数据准备、基础模型训练、FiLM 条件化微调以及变异注释等全部步骤，帮助你在自己的物种/组织上复现完整实验。

---

## 1. 环境与资源

- **依赖**：Python ≥ 3.10、PyTorch（GPU 训练建议 CUDA≥11.7）、NumPy/Pandas/HDF5/pyfaidx 等；执行 `pip install -e .` 会自动安装需要的 Python 包。
- **硬件**：基础模型/FiLM 微调建议使用至少 16GB GPU；Variant 注释可在 CPU 上运行但会较慢。
- **基础数据**：
  - 参考基因组 FASTA（例：`data/genome.fa` + `.fai`）。
  - 组织/物种对应的 GTF/GFF 注释文件。
  - SpliceAI 官方 annotation（示例：`data/grch38.txt`）或自定义注释。
  - 组织表达矩阵：已拼接且标准化的 RBP+HVG 矩阵，如 `data/tissue_expression_features_scaled.csv`。

安装完成后可用下列命令快速检查：

```bash
pip install -e .
openspliceai --help
```

---

## 2. 完整流程概览

1. **create-data**：读取 GTF/GFF + FASTA，生成 HDF5 数据集（训练/验证/测试）。
2. **train**：在生成的 HDF5 上训练基础模型（无 FiLM，捕捉通用剪接规则）。
3. **prepare_rbp_expression**：为目标组织生成条件向量（RBP + HVG）。
4. **transfer**：加载基础模型 checkpoint，开启 FiLM，在组织特异数据上微调。
5. **predict**：使用组织特异模型 + 条件向量，对 FASTA 序列直接做剪接位点预测（输出 BED）。
6. **variant**：使用组织特异模型 + 条件向量对 VCF 进行剪接影响注释。

以下章节将详细说明每一步。

---

## 3. Step 1：生成 HDF5 数据 (`create-data`)

`create-data` 会执行两件事：1) `create_datafile` 将注释切成序列窗口；2) `create_dataset` 输出 HDF5 分片。典型命令：

```bash
openspliceai create-data \
  --annotation-gff data/limb_filtered.gff3 \
  --genome-fasta data/genome.fa \
  --output-dir /home1/xyf/data/openspliceai_data/dataset_limb \
  --parse-type canonical \
  --biotype protein-coding \
  --chr-split train-test \
  --split-method human \
  --split-ratio 0.8 \
  --val_split_ratio 0.1 \
  --flanking-size 10000 \
  --verify-h5 \
  --remove-paralogs \
  --min-identity 0.8 \
  --min-coverage 0.5
```

**要点**：

- `--flanking-size` 控制模型输入序列长度（越长越能捕捉远端上下文，在 10000 时需更多显存）。
- `--chr-split` + `--split-method` 决定染色体划分方式；`human` 预设以 SpliceAI 论文规则划分。
- `--remove-paralogs` 会调用 minimap2 检查训练/测试集之间的同源序列，避免信息泄漏。
- `--verify-h5` 可在生成后自动运行一致性检查。

输出目录一般包含 `dataset_train.h5`、`dataset_validation.h5`、`dataset_test.h5` 及相关日志/统计文件，后续 `train`、`transfer` 都直接引用（命名中需保留 `train`/`validation`/`test` 关键字）。

> **提示：组织特异 GFF3 预处理**  
> `workflow/step_all.sh` 将 `step0`~`step3` 脚本串联起来，可通过 `bash workflow/step_all.sh <tissue>` 一键生成 `<tissue>_step3.gff3`。每个 step 会过滤 isoform、转换 GFF3、重命名 transcript→mRNA、并补充 `gene_biotype`，供 `create-data` 使用。

---

## 4. Step 2：训练基础模型 (`train`)

基础模型学习“组织无关”的剪接规则，稍后 FiLM 微调会在此基础上注入条件信息。示例命令：

```bash
openspliceai train \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --flanking-size 10000 \
  --epochs 10 \
  --scheduler CosineAnnealingWarmRestarts \
  --loss cross_entropy_loss \
  --output-dir runs/base_model \
  --project-name limb_base \
  --random-seed 42
```

**提示**：

- `train-dataset` 名称需包含 `train`，程序会自动将同目录下的文件名替换为 `validation` 以加载验证集（`create-data` 默认输出 `dataset_train/validation/test.h5`）。
- `train` 模式默认不启用 FiLM，也不需要 RBP/HVG 输入。
- 输出目录包含每个 epoch 的 `model_{epoch}.pt`、best checkpoint、训练/验证日志（AUPRC、loss 曲线等）。
- 若训练多个物种，可在 `create-data` 与 `train` 中切换不同 HDF5，即可得到多条基础模型。

---

## 5. Step 3：生成组织条件向量 (`prepare_rbp_expression`)

FiLM 需要一个固定的条件向量。现在直接使用单个矩阵（已包含 RBP + HVG，且已标准化）例如 `data/tissue_expression_features_scaled.csv`，行是组织名、列是特征名。运行：

```bash
python -m openspliceai.scripts.prepare_rbp_expression \
  --matrix /home1/xyf/project/github/OpenSpliceAI/data/tissue_expression_features_scaled.csv \
  --tissue limb \
  --output data/limb_features.json \
  --format json \
  --standardize none
```

输出文件格式：

```json
{
  "values": [...],          # RBP + HVG 向量
  "rbp_names": ["feature1", "feature2", ...]
}
```

**务必在训练和推理时复用同一文件**，checkpoint 会记录 `rbp_dim` 与 `rbp_names` 用于校验。如果你有额外的组织特征，可直接追加到该矩阵列中，再用本脚本导出。

---

## 6. Step 4：FiLM 微调 (`transfer`)

`transfer` 在基础模型之上加载 FiLM 侧支 MLP，将组织向量注入主干末端（所有残差块之后、最终 1×1 卷积之前），对 32 个通道做一次乘加调制。根据数据量，可选择 **单组织模式**（一次只微调一个组织）或 **多组织共享模式**（一次加载多个组织，以便同一个 checkpoint 在推理时切换条件向量）。

### 6.1 单组织模式（与旧版用法一致）

```bash
openspliceai transfer \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --pretrained-model runs/base_model/model_best.pt \
  --flanking-size 10000 \
  --epochs 5 \
  --rbp-expression data/limb_features.json \
  --unfreeze 4 \
  --output-dir runs/limb_film \
  --project-name limb_film
```

关键参数说明（新版 FiLM 逻辑）：

- 与 `train` 相同，`train-dataset` 名称含 `train` 即可，程序会自动定位同目录 `dataset_validation.h5` 作为验证集。
- `--rbp-expression`：指向前一步生成的 JSON/NPY，内部包含 RBP + HVG 特征（示例维度 753，经 Z-score 标准化或已提供标准化矩阵）。
- FiLM 注入位置固定在主干末端（final 1×1 卷积前），无需 `--film-start-layer`。
- FiLM 侧支 MLP：`Linear(in_dim→128) → LayerNorm → ReLU → Dropout(0.2) → Linear(128→2*channels)`，输出层权重/偏置零初始化，确保初始 γ=1、β=0（热启动）。
- 训练时默认冻结主干，始终解冻 FiLM 侧支和最终 1×1 卷积头；`--unfreeze` 额外解冻末端若干 ResidualUnit，`--unfreeze-all` 可全模型联训。
- 训练日志结构与 `train` 类似，`model_best.pt` 中记录了 `rbp_dim`、`rbp_names` 等元信息。未提供 `--rbp-expression` 时，FiLM 会退化为 γ=1/β=0，表现等同基础模型。

### 6.2 多组织共享模式：`--tissue-config`

如果希望“训练阶段就让模型同时看到多个组织”，从而只保存一份 FiLM checkpoint，在 Variant 阶段通过更换条件向量实现 double-run，可以提供一个 JSON 配置列举所有组织的 HDF5 与特征：

```json
[
  {
    "name": "blood",
    "train_dataset": "/path/blood/dataset_train.h5",
    "valid_dataset": "/path/blood/dataset_validation.h5",
    "test_dataset":  "/path/blood/dataset_test.h5",
    "rbp_expression": "data/blood_features.json"
  },
  {
    "name": "neuron",
    "train_dataset": "/path/neuron/dataset_train.h5",
    "valid_dataset": "/path/neuron/dataset_validation.h5",
    "test_dataset":  "/path/neuron/dataset_test.h5",
    "rbp_expression": "data/neuron_features.json"
  }
]
```

命令示例：

```bash
openspliceai transfer \
  --tissue-config config/tissues.json \
  --pretrained-model runs/base_model/model_best.pt \
  --flanking-size 10000 \
  --epochs 5 \
  --unfreeze 4 \
  --output-dir runs/shared_film \
  --project-name shared_film
```

注意事项：

- `--tissue-config` 与 `--rbp-expression` 互斥；前者表示一次性加载多个组织，每个组织都需要 train/valid/test HDF5。
- 配置中的所有 `rbp_expression` 向量必须维度一致、列名顺序相同（脚本会自动校验）。
- 训练过程中 dataloader 会混合不同组织的 batch，FiLM γ/β 由同一侧支 MLP 生成，但使用对应组织的向量。
- 训练完成后，Variant 阶段只需要这一份模型：运行多次 `openspliceai variant`，更换 `--rbp-expression` 指向 blood/neuron 等向量，即可得到可比较的组织特异预测。
- 多组织模式下不再自动缩放 batch_size，仍用单组织基准批量（flanking=400 默认每卡 18×GPU 数）。梯度累积步数默认为组织数。若显存吃紧，可手动调低基准批量或 `--unfreeze`。

---

## 7. Step 5：序列级预测 (`predict`)

`predict` 现在支持 RBP/HVG 条件向量，可直接对 FASTA 序列输出组织特异的剪接位点 BED。示例：

```bash
openspliceai predict \
  --input-sequence data/neuron_genes.fa \
  --model runs/shared_film/model_best.pt \
  --flanking-size 10000 \
  --rbp-expression data/neuron_features.json \
  --output-dir predict_out/neuron/ \
  --threshold 1e-6 \
  --predict-all
```

说明与注意：

- `--rbp-expression` 与 `variant` 用法一致，必须与 FiLM checkpoint 中记录的 `rbp_dim`/`rbp_names` 对齐；缺失时会报错，维度或顺序不匹配也会报错。
- 若加载的是无 FiLM 的基础模型，可省略 `--rbp-expression`，此时预测等同于标准 SpliceAI。
- `--predict-all` 会先写中间 HDF5/pt，再生成 BED；关闭该选项则直接边预测边写 BED，节省磁盘。
- 输出的 `acceptor_predictions.bed`、`donor_predictions.bed` 可按组织对比（如 limb vs neuron）。

---

## 8. Step 6：Variant 注释 (`variant`)

最后，将 VCF 输入组织特异模型即可得到 delta 分数和剪接位点位移：

```bash
openspliceai variant \
  --input data/decipher_variants_all.vcf \
  --output results/annotated_limb.vcf \
  --model runs/limb_film/model_best.pt \
  --ref-genome data/genome.fa \
  --annotation data/grch38_chr.txt \
  --flanking-size 10000 \
  --rbp-expression data/limb_features.json
```

**注意**：

- 若 checkpoint 包含 FiLM/条件元数据，`variant` 会检查输入向量维度与名称；缺失或顺序错误会直接报错，避免预测偏差。
- 多组织共享模型只需运行 `variant` 多次，替换 `--rbp-expression` 即可输出 blood/neuron 等结果；由于 checkpoint 相同，分数直接可比。

---

## 8. 结果与目录结构示例

```
OpenSpliceAI/
├── data/
│   ├── genome.fa / genome.fa.fai
│   ├── grch38.txt
│   ├── tissue_rbp_matrix.csv
│   └── developmental_system_hvg.csv
├── runs/
│   ├── base_model/
│   │   ├── model_best.pt
│   │   └── metrics/*.txt
│   └── limb_film/
│       ├── model_best.pt
│       └── metrics/*.txt
├── results/
│   └── annotated_limb.vcf
└── scripts/prepare_rbp_expression.py
```

建议将训练日志（TensorBoard/自定义可视化）与 checkpoint 同步归档，方便比较不同组织或参数设置。

---

## 9. 常见问题

1. **FiLM 模型可以在没有条件向量时运行吗？**  
   - 可以；若不传 `--rbp-expression`，FiLM γ=1、β=0，相当于标准 SpliceAI。仅当 checkpoint 中声明 `rbp_dim>0` 且你忘记提供向量时，程序才会报错。

2. **如何自定义组织特征？**  
   - 将任意组织级别特征（如 RBP TPM、高变基因表达、UMAP 坐标等）拼成 CSV，行名与 `tissue_rbp_matrix.csv` 保持一致，再传给 `--hvg-matrix` 或直接替换原矩阵。`prepare_rbp_expression` 会自动拼接、标准化。

3. **一次训练能覆盖多个组织吗？**  
   - 可以，使用 `--tissue-config` 提供多个组织的 train/valid/test HDF5 与各自表达向量，`transfer` 会混合训练并共享同一 FiLM 侧支（详见 6.2）。

4. **Variant 结果如何解释？**  
   - `variant` 输出与官方 SpliceAI 相同的 delta scores（ΔAG、ΔAL、ΔDG、ΔDL）及位置偏移，可直接用于筛选可能影响剪接的突变。若对多个组织运行，可比较不同组织的分数差异。

---

## 10. 更多资源

- 文档/教程（英文）：`README.md` 与 `docs/` 目录。
- 相关脚本：
  - `openspliceai/create_data/*`：HDF5 生成与验证。
  - `openspliceai/train_base/*`：SpliceAI 主体、FiLM 层、训练/验证循环。
  - `openspliceai/rbp/expression.py`：条件向量读写及标准化工具。
  - `openspliceai/variant/variant.py`：VCF 注释入口。
- 若遇到问题或希望贡献功能，欢迎在 GitHub Issues 中提问。

祝你在 OpenSpliceAI 的研究中取得好结果！ 😊
