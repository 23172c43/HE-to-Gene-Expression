"""run_hest_train.py -- TRAIN Light-HGGEP từ đầu trên HEST (.h5ad) theo LOPO.

Mỗi file .h5ad = 1 mẫu (patient). LOPO: mỗi fold chọn 1 mẫu làm test, phần còn
lại train; trong tập train lấy mẫu alphabet đầu làm validation (early stopping).
— Giống hệt cách HER2ST làm (run_pipeline.py), chỉ khác dataset là HEST h5py.

Metric: gene-wise PCC/Spearman/RMSE/MAE, per-section average (section_ids).
HEST không có nhãn mô học → không tính ARI/NMI (label=None).

Cách chạy:
  python run_hest_train.py --hest-dir data/hest/st --gene-list data/her_hvg_cut_1000.npy
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from dataset import LightHGGEP_HEST_h5py
from models.LightHGGEP import LightHGGEP
from predict import lighthggep_predict, get_section_ids, get_R, get_Spearman, get_MSE, get_MAE
from sampler_utils import SectionBatchSampler, section_collate_fn

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger


ABLATION_FLAGS = {
    'full':           dict(use_graph=True,  use_cross_scale=True,  depthwise=True),
    'no_graph':       dict(use_graph=False, use_cross_scale=True,  depthwise=True),
    'no_cross_scale': dict(use_graph=True,  use_cross_scale=False, depthwise=True),
    'no_depthwise':   dict(use_graph=True,  use_cross_scale=True,  depthwise=False),
}


class HestEpochCallback(Callback):
    """In 1 dòng tiến độ mỗi epoch (train/val mse+pcc, lr, thời gian, eta).
    Hỗ trợ cả cặp key Light-HGGEP (val_*) lẫn STNet/HisToGene (valid_*)."""
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started_at = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m  = trainer.callback_metrics
        train_mse = m.get('train_mse', m.get('train_loss', 0.0))
        train_pcc = m.get('train_pcc', float('nan'))
        val_mse   = m.get('val_mse',   m.get('valid_loss', m.get('val_loss', 0.0)))
        val_pcc   = m.get('val_pcc',   m.get('valid_pcc', float('nan')))
        ep    = trainer.current_epoch + 1
        total = trainer.max_epochs
        opt   = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]['lr']
        elapsed  = time.perf_counter() - getattr(self, 'epoch_started_at', time.perf_counter())
        remaining = max(total - ep, 0) * elapsed
        print(f"[ep {ep}/{total}] "
              f"train_mse={float(train_mse):.4f} train_pcc={float(train_pcc):.4f} "
              f"val_mse={float(val_mse):.4f} val_pcc={float(val_pcc):.4f} lr={lr:.4e} "
              f"epoch_time={elapsed:.1f}s eta={remaining / 60:.1f}m", flush=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_fold(fold, args, gene_list):
    N_GENES = len(gene_list)
    abl = ABLATION_FLAGS[args.ablation]

    train_dataset = LightHGGEP_HEST_h5py(
        args.hest_dir, gene_list, train=True, fold=fold,
        k_neighbors=args.k, patch_size=args.patch_size)
    VAL_SECTION = sorted(train_dataset.names)[0]   # mẫu train alphabet đầu làm val

    # test mẫu cho fold này = mẫu fold (tên file)
    all_samples = sorted(os.path.basename(p).split('.')[0]
                         for p in os.listdir(args.hest_dir) if p.endswith('.h5ad'))
    TEST_SECTION = all_samples[fold % len(all_samples)]
    print(f"\n{'='*72}\nFOLD {fold} | TEST={TEST_SECTION} | VAL={VAL_SECTION} | "
          f"train={len(train_dataset.names)}\n{'='*72}")

    train_sampler = SectionBatchSampler(train_dataset, batch_size=args.batch_size,
                                        shuffle=True, exclude_sections=[VAL_SECTION])
    val_sampler = SectionBatchSampler(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, include_sections=[VAL_SECTION])
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=section_collate_fn, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(train_dataset, batch_sampler=val_sampler,
                            collate_fn=section_collate_fn, num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())

    # Model train từ đầu (ngẫu nhiên init)
    model = LightHGGEP(
        n_genes=N_GENES, k_neighbors=args.k,
        learning_rate=args.lr, max_epochs=args.max_epochs,
        cnn_chunk=args.batch_size, **abl)
    for section, A_norm in train_dataset.A_norm_cache.items():
        model.set_graph(section, torch.from_numpy(A_norm).float())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Light-HGGEP params: {total_params:,} (ablation={args.ablation})")

    early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir, filename=f'hest_fold{fold}_{{epoch:02d}}_{{val_loss:.4f}}',
        save_top_k=1, monitor='val_loss', mode='min', save_last=False)
    logger = CSVLogger("logs", name=f"hest_fold{fold}")

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, max_epochs=args.max_epochs,
        callbacks=[early_stop, ckpt_cb, HestEpochCallback()], logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    best_val = ckpt_cb.best_model_score
    best_val = float(best_val) if best_val is not None else float('nan')
    print(f"\nBest ckpt: {best_ckpt} | val_loss={best_val:.4f}")
    best_model = LightHGGEP.load_from_checkpoint(
        best_ckpt, n_genes=N_GENES, k_neighbors=args.k,
        learning_rate=args.lr, max_epochs=args.max_epochs,
        cnn_chunk=args.batch_size, **abl)

    # ---- Test ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = LightHGGEP_HEST_h5py(
        args.hest_dir, gene_list, train=False, fold=fold,
        k_neighbors=args.k, patch_size=args.patch_size)
    for section, A_norm in test_dataset.A_norm_cache.items():
        best_model.set_graph(section, torch.from_numpy(A_norm).float())

    test_sampler = SectionBatchSampler(test_dataset, batch_size=args.batch_size,
                                       shuffle=False, full_section=True)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,
                             collate_fn=section_collate_fn, num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available())

    adata_pred, adata_gt = lighthggep_predict(best_model, test_loader, device=device)

    # Loại cột toàn-0 (gene vắng / không phát hiện) khỏi metric
    mask = np.array((np.abs(adata_gt.X).max(axis=0) > 0)).flatten()
    n_genes_eval = int(mask.sum())
    subset = [g for g, keep in zip(gene_list, mask) if keep]
    ap = adata_pred[:, mask].copy()
    ag = adata_gt[:, mask].copy()

    section_ids = get_section_ids(test_dataset)
    R, _ = get_R(ap, ag, section_ids=section_ids)
    rho, _ = get_Spearman(ap, ag, section_ids=section_ids)
    mse = get_MSE(ap, ag, section_ids=section_ids)
    mae = get_MAE(ap, ag, section_ids=section_ids)

    return {
        'fold': fold,
        'test_sample': TEST_SECTION,
        'val_sample': VAL_SECTION,
        'n_train_samples': len(train_dataset.names),
        'n_test_spots': adata_pred.shape[0],
        'n_genes_eval': n_genes_eval,
        'pearson': float(np.nanmean(R)),
        'spearman': float(np.nanmean(rho)),
        'rmse': float(np.nanmean(np.sqrt(mse))),
        'mae': float(np.nanmean(mae)),
        'best_val_loss': best_val,
        'ablation': args.ablation,
        'k': args.k,
        'n_genes': N_GENES,
    }


def main():
    _p = argparse.ArgumentParser(description="Train Light-HGGEP từ đầu trên HEST (LOPO)")
    _p.add_argument('--hest-dir', type=str, default='data/hest/st')
    _p.add_argument('--gene-list', type=str, default='data/her_hvg_cut_1000.npy')
    _p.add_argument('--k', type=int, default=4)
    _p.add_argument('--ablation', choices=list(ABLATION_FLAGS), default='full')
    _p.add_argument('--patch-size', type=int, default=224)
    _p.add_argument('--num-workers', type=int, default=2)
    _p.add_argument('--batch-size', type=int, default=32)
    _p.add_argument('--lr', type=float, default=1e-4)
    _p.add_argument('--max-epochs', type=int, default=100)
    _p.add_argument('--patience', type=int, default=15)
    _p.add_argument('--fold-start', type=int, default=0)
    _p.add_argument('--fold-end', type=int, default=None,
                    help="Số fold chạy (mặc định = số mẫu HEST, 1 fold/mẫu LOPO)")
    _p.add_argument('--ckpt-dir', type=str, default='model_ckpts')
    _p.add_argument('--seed', type=int, default=42)
    args = _p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    gene_list = list(np.load(args.gene_list, allow_pickle=True))
    print(f"Gene list: {args.gene_list} -> {len(gene_list)} gene")

    all_samples = sorted(os.path.basename(p).split('.')[0]
                         for p in os.listdir(args.hest_dir) if p.endswith('.h5ad'))
    if not all_samples:
        raise FileNotFoundError(f"Không tìm thấy *.h5ad trong {args.hest_dir}")
    n_fold = args.fold_end if args.fold_end is not None else len(all_samples)
    n_max = len(all_samples)
    if n_fold > n_max:
        print(f"[WARN] --fold-end={n_fold} vượt số mẫu ({n_max}). Giới hạn xuống {n_max} "
              f"(và fold-start lên {args.fold_start}..{n_max - 1}).")
        n_fold = n_max
    if args.fold_start >= n_max:
        raise SystemExit(f"[LỖI] --fold-start={args.fold_start} >= số mẫu {n_max} — không có gì để chạy.")
    print(f"Tổng {len(all_samples)} mẫu → chạy fold {args.fold_start}..{n_fold - 1}")

    import pandas as pd
    rows = []
    for fold in range(args.fold_start, n_fold):
        rows.append(run_fold(fold, args, gene_list))

    df = pd.DataFrame(rows)
    out = f"hest_train_{args.ablation}_k{args.k}.csv"
    if os.path.isfile(out):
        old = pd.read_csv(out)
        old = old[~old['fold'].isin(df['fold'])]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)

    print("\n" + "="*72)
    print("HEST TRAIN (LOPO) RESULTS")
    print("="*72)
    show = ['fold', 'test_sample', 'n_train_samples', 'n_test_spots', 'n_genes_eval',
            'pearson', 'spearman', 'rmse', 'mae', 'best_val_loss']
    print(df[show].to_string(index=False))
    for c in ['pearson', 'spearman', 'rmse', 'mae']:
        col = df[c].dropna()
        if len(col):
            print(f"  {c:10s}: mean={col.mean():.4f} std={col.std():.4f} (n={len(col)})")
    print(f"Đã lưu: {out}")


if __name__ == '__main__':
    main()