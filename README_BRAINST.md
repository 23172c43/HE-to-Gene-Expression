# Hướng dẫn chạy BRAIN-ST (V1_Adult_Mouse_Brain)

Dataset: mouse brain coronal section, 10x Visium, 1 section (~2702 spots, 250 gene
Top250). Tải tự động qua `scanpy.datasets.visium_sge` (cần mạng lần đầu).

Không có tissue-type label → ARI/NMI = NaN. Vẫn đánh giá được PCC / Spearman /
RMSE / MAE / Moran's I.

## 1. Cài đặt (chỉ lần đầu)
```bash
pip install --only-binary :all: scprep opencv-python-headless albumentations einops
```
(Riêng Kaggle: scanpy đã có sẵn; nếu conflict xem notebook `hest-1k`).

## 2. Light-HGGEP trên BRAIN-ST
```bash
python run_pipeline.py --datasets brainst
```
- Tự động: download V1_Adult_Mouse_Brain, build KNN graph, chia spot-level 80/20
  (train 2162 / val 540), train 100 epoch, eval PCC/RMSE/Moran's I.
- Kết quả lưu: `Light-HGGEP_results.csv`, `gene_predictions_stats.csv`,
  `figures/Light-HGGEP_*_fold5.png`.

## 3. Baseline (STNet + HisToGene) trên BRAIN-ST
```bash
# Chạy cả 2 baseline
python run_baselines.py --mode all --datasets brainst

# Hoặc từng cái
python run_baselines.py --mode stnet --datasets brainst
python run_baselines.py --mode histogene --datasets brainst
```
- `--max_epochs`, `--batch_size`, `--lr`, `--n_gpus` tương tự HER2ST.
- Kết quả lưu: `baselines_results.csv`, `figures/*_PCC_fold5.png`.

## 4. So sánh HER2ST ↔ BRAIN-ST
```bash
python run_pipeline.py --datasets her2st              # Light-HGGEP HER2ST
python run_pipeline.py --datasets brainst             # Light-HGGEP BRAIN-ST
python run_baselines.py --mode all --datasets her2st # baseline HER2ST
python run_baselines.py --mode all --datasets brainst # baseline BRAIN-ST
```
Đọc PCC / RMSE / Moran's I từ các file CSV để so sánh công bằng.

## 5. Lưu ý
- BRAIN-ST chỉ 1 section → không có LOOCV. Split là spot-level 80/20 cố định
  (seed=42), không phải fold.
- `n_genes` tự override = 250 (Top250 của section), không dùng `her_hvg_cut_1000.npy`.
- Không vẽ FASN nếu gene đó không nằm trong Top250 (tự động fallback gene đầu).
- Ảnh H&E load từ `adata.uns['spatial'][...]['images']['hires']` (ndarray),
  crop 224×224 quanh tọa độ full-res × `tissue_hires_scalef`.

## 6. Files liên quan
- `dataset.py`: `LightHGGEP_BRAINST`, `BRAINSTSpotDataset`, `BRAINSTSlideDataset`
- `run_pipeline.py`: `--datasets brainst`, `SectionBatchSampler(section_indices_override=...)`
- `run_baselines.py`: `--datasets brainst`, `brainst_split_train_val`,
  `brainst_stnet_predict`, `brainst_histogene_predict`
