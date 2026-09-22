"""run_hest_pipeline.py -- External validation Light-HGGEP trên HEST (.h5ad).

Dùng model Light-HGGEP ĐÃ TRAIN trên HER2ST (785 gene) để predict biểu hiện gene
trên từng mẫu HEST (mỗi file .h5ad = 1 mẫu = 1 graph riêng). Metric: gene-wise
PCC/Spearman/RMSE/MAE, per-section average (section_ids) — HEST không có nhãn mô
học nên ARI/NMI = NaN.

Nguyên tắc gene:
  - Khi load (dataset LighHGGEP_HEST_h5py): giữ ĐỦ gene_list (785) — gene vắng
    mặt trong mẫu được điền 0 để khớp shape (N, 785) với ckpt.
  - Khi tính điểm: các cột toàn-0 (gene vắng mặt hoặc không phát hiện được ở mọi
    spot) bị LOẠI khỏi tính toán, chỉ tính trên gene có mặt -> tránh PCC bị kéo
    về 0 tự nhiên do cột hằng số.

Cách chạy:
  python run_hest_pipeline.py --ckpt model_ckpts/lighthggep_fold6_epoch=86_val_loss=0.6360.ckpt
"""
import argparse
import glob
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from dataset import LightHGGEP_HEST_h5py
from models.LightHGGEP import LightHGGEP
from predict import (lighthggep_predict, get_section_ids,
                     get_R, get_Spearman, get_MSE, get_MAE)
from sampler_utils import SectionBatchSampler, section_collate_fn


ABLATION_FLAGS = {
    'full':           dict(use_graph=True,  use_cross_scale=True,  depthwise=True),
    'no_graph':       dict(use_graph=False, use_cross_scale=True,  depthwise=True),
    'no_cross_scale': dict(use_graph=True,  use_cross_scale=False, depthwise=True),
    'no_depthwise':   dict(use_graph=True,  use_cross_scale=True,  depthwise=False),
}


def main():
    _p = argparse.ArgumentParser(description="External validation Light-HGGEP trên HEST (.h5ad)")
    _p.add_argument('--hest-dir', type=str, default='data/hest/st',
                    help="Thư mục chứa các file .h5ad (mặc định data/hest/st)")
    _p.add_argument('--ckpt', type=str, required=True,
                    help="Đường dẫn checkpoint Light-HGGEP đã train HER2ST")
    _p.add_argument('--k', type=int, default=4, help="Số láng giềng K-NN cho graph (mặc định 4)")
    _p.add_argument('--ablation', choices=list(ABLATION_FLAGS), default='full',
                    help="Cấu hình ablation (phải khớp lúc train ckpt)")
    _p.add_argument('--patch-size', type=int, default=224, help="Kích thước patch resize (mặc định 224)")
    _p.add_argument('--gene-list', type=str, default='data/her_hvg_cut_1000.npy',
                    help="File gene list (.npy) dùng làm target (785)")
    _p.add_argument('--num-workers', type=int, default=2)
    _p.add_argument('--batch-size', type=int, default=64,
                    help="Chỉ dùng cho CNN chunk bên trong forward (test vẫn full-section 1 batch)")
    args = _p.parse_args()

    # 1) Gene list + N_GENES
    gene_list = list(np.load(args.gene_list, allow_pickle=True))
    N_GENES = len(gene_list)
    print(f"Gene list: {args.gene_list} -> N_GENES = {N_GENES}")

    # 2) Load model từ checkpoint (khớp hparams lúc train)
    abl = ABLATION_FLAGS[args.ablation]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | ablation={args.ablation} | k={args.k}")

    model = LightHGGEP.load_from_checkpoint(
        args.ckpt,
        n_genes=N_GENES,
        k_neighbors=args.k,
        cnn_chunk=args.batch_size,
        **abl,
    )
    model.to(device)

    # Verify hparams trong ckpt khớp (an toàn nếu load thiếu/trùng)
    ckpt_hparams = torch.load(args.ckpt, map_location='cpu', weights_only=False).get('hyper_parameters', {})
    if ckpt_hparams.get('n_genes') not in (None, N_GENES):
        raise ValueError(f"ckpt n_genes={ckpt_hparams.get('n_genes')} != gene_list {N_GENES}")

    # 3) Loop từng mẫu HEST
    files = sorted(glob.glob(os.path.join(args.hest_dir, '*.h5ad')))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy *.h5ad trong {args.hest_dir}")
    print(f"Tìm thấy {len(files)} mẫu HEST.")

    results = []
    for path in files:
        sample = os.path.basename(path).split('.')[0]
        print(f"\n{'='*72}\nSample: {sample}\n{'='*72}")

        ds = LightHGGEP_HEST_h5py(path, gene_list, k_neighbors=args.k,
                                  patch_size=args.patch_size)
        # Gắn graph riêng cho mẫu
        for name, A_norm in ds.A_norm_cache.items():
            model.set_graph(name, torch.from_numpy(A_norm).float())

        sampler = SectionBatchSampler(ds, batch_size=args.batch_size,
                                      shuffle=False, full_section=True)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=section_collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())

        adata_pred, adata_gt = lighthggep_predict(model, loader, device=device)

        # 4) Loại gene toàn-0 khỏi tính điểm
        mask = (np.abs(adata_gt.X).max(axis=0) > 0)
        mask = np.array(mask).flatten()
        n_genes_eval = int(mask.sum())
        subset = [g for g, keep in zip(gene_list, mask) if keep]
        adata_pred_sub = adata_pred[:, mask].copy()
        adata_gt_sub = adata_gt[:, mask].copy()

        # 4b) Tính metric trực tiếp (section-aware) — HEST không có label,
        # skip scanpy clustering (avoid PCA/sklearn version mismatch).
        section_ids = get_section_ids(ds)
        R, _ = get_R(adata_pred_sub, adata_gt_sub, section_ids=section_ids)
        spearman_rho, _ = get_Spearman(adata_pred_sub, adata_gt_sub, section_ids=section_ids)
        mse = get_MSE(adata_pred_sub, adata_gt_sub, section_ids=section_ids)
        mae = get_MAE(adata_pred_sub, adata_gt_sub, section_ids=section_ids)

        results.append({
            'sample': sample,
            'n_spots': ds.__len__(),
            'n_genes_total': N_GENES,
            'n_genes_eval': n_genes_eval,
            'pearson': float(np.nanmean(R)),
            'spearman': float(np.nanmean(spearman_rho)),
            'rmse': float(np.nanmean(np.sqrt(mse))),
            'mae': float(np.nanmean(mae)),
            'ablation': args.ablation,
            'k': args.k,
        })
        print(f"  n_spots={ds.__len__()} | n_genes_eval={n_genes_eval} | "
              f"pearson={float(np.nanmean(R)):.4f} spearman={float(np.nanmean(spearman_rho)):.4f} "
              f"rmse={float(np.nanmean(np.sqrt(mse))):.4f} mae={float(np.nanmean(mae)):.4f}")

    # 5) Tổng hợp
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = f"hest_validation_{args.ablation}_k{args.k}.csv"
    if os.path.isfile(out_csv):
        old = pd.read_csv(out_csv)
        old = old[~old['sample'].isin(df['sample'])]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print("\n" + "="*72)
    print("HEST EXTERNAL VALIDATION")
    print("="*72)
    show = ['sample', 'n_spots', 'n_genes_eval', 'pearson', 'spearman', 'rmse', 'mae']
    print(df[show].to_string(index=False))
    print(f"\nMean ± STD: pearson {df['pearson'].mean():.4f}±{df['pearson'].std():.4f}, "
          f"spearman {df['spearman'].mean():.4f}±{df['spearman'].std():.4f}")
    print(f"\nĐã lưu: {out_csv}")


if __name__ == '__main__':
    main()