"""Shared, leakage-free HER2ST evaluation protocol."""

import numpy as np

from predict import (get_R, get_MSE, get_MAE, get_Spearman, get_MoransI_all,
                     cluster_with_nmi)
from utils import comp_tsne_km

PROTOCOL_NAME = "her2st-loocv-v1-raw-lognorm"


def evaluate_her2st_predictions(adata_pred, adata_gt, genes, label=None,
                                 n_clusters=4, section_ids=None):
    """Evaluate all models on raw, log-library-normalised HER2ST expression.

    A standardised copy is used solely for PCA/t-SNE/K-means visualisation.
    Thus scaling cannot alter PCC, RMSE/MAE, Spearman, or Moran's I.

    section_ids: [MỚI - sửa lỗi Moran's I đa lát cắt] None hoặc array[str] độ dài
        N spot, lấy từ predict.get_section_ids(test_dataset), CÙNG THỨ TỰ với
        adata_pred/adata_gt. Bắt buộc truyền khi tập test của 1 fold gồm nhiều
        lát cắt (patient-level/LOPO split) -- nếu để None trong trường hợp đó,
        Moran's I sẽ bị tính sai do nối láng giềng xuyên lát cắt (xem
        predict.get_MoransI để biết chi tiết).
    """
    genes = list(genes)
    if adata_pred.shape != adata_gt.shape:
        raise ValueError(f"Prediction/ground-truth shapes differ: "
                         f"{adata_pred.shape} vs {adata_gt.shape}")
    if adata_pred.shape[1] != len(genes):
        raise ValueError(f"Expected {len(genes)} genes, got {adata_pred.shape[1]}")

    adata_pred.var_names = genes
    adata_gt.var_names = genes
    R, p_values = get_R(adata_pred, adata_gt, section_ids=section_ids)
    Spearman, spearman_pvalues = get_Spearman(adata_pred, adata_gt, section_ids=section_ids)
    MSE = get_MSE(adata_pred, adata_gt, section_ids=section_ids)
    MAE = get_MAE(adata_pred, adata_gt, section_ids=section_ids)
    RMSE = np.sqrt(MSE)
    morans = get_MoransI_all(adata_pred, adata_gt, top_k=50, section_ids=section_ids)

    adata_visual = adata_pred.copy()
    # Thay sc.pp.scale bằng z-score numpy (ko phụ thuộc scanpy). Chuẩn hóa từng
    # gene (cột) theo mean/std; gene ko biến thiên (std=0) giữ nguyên 0.
    X = np.asarray(adata_visual.X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    adata_visual.X = ((X - mean) / std).astype(np.float32)
    adata_visual = comp_tsne_km(adata_visual, n_clusters)
    if label is None:
        ARI = NMI = float("nan")
    else:
        _, ARI, NMI = cluster_with_nmi(adata_visual, label)

    return adata_visual, {
        "R": R, "p_values": p_values,
        "Spearman": Spearman, "spearman_pvalues": spearman_pvalues,
        "MSE": MSE, "MAE": MAE, "RMSE": RMSE, "morans": morans,
        "ARI": ARI, "NMI": NMI,
        "pearson": np.nanmean(R),
        "median_pearson": np.nanmedian(R),
        "spearman": np.nanmean(Spearman),
        "rmse": np.nanmean(RMSE), "mae": np.nanmean(MAE),
        "morans_i_pred": np.nanmean(morans["pred"]),
        "morans_i_gt": np.nanmean(morans["gt"]),
    }
