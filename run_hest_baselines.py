"""run_hest_baselines.py -- Train + eval STNet / HisToGene trên HEST (.h5ad) theo LOPO.

Giống run_hest_train.py (Light-HGGEP) nhưng dành cho baseline STNet và HisToGene,
để so sánh công bằng trên cùng tập HEST cùng phương pháp LOPO (1 mẫu test/fold,
val = mẫu train alphabet đầu, early stopping theo 'valid_loss').

- STNet    : patch 224, spot-level, feature extractor ResNet-pretrained.
- HisToGene: patch crop 112 (từ 224), slide-level attention, gộp toàn bộ spot
             của 1 mẫu thành 1 slide (batch=1).
- n_genes = 785 (khớp Light-HGGEP và her_hvg_cut_1000.npy).
- Không dùng graph / SGC; không cần SectionBatchSampler.

Cách chạy:
  python run_hest_baselines.py --mode stnet --hest-dir data/hest/st
  python run_hest_baselines.py --mode histogene --hest-dir data/hest/st
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate

warnings.filterwarnings('ignore')

from dataset import LightHGGEP_HEST_h5py
from predict import stnet_predict, histogene_predict, get_section_ids
from predict import get_R, get_Spearman, get_MSE, get_MAE

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from run_hest_train import HestEpochCallback

_HISTO_PATCH = 112   # HisToGene dùng patch 112 (template HER2ST)


class HestBaselineDataset(Dataset):
    """Adapter: bọc LightHGGEP_HEST_h5py để __getitem__ trả 4 phần tử
    (patch, loc, exp, center) — khớp kỳ vọng của stnet_predict / histogene_predict
    (HER2ST train=False trả 4 phần tử). HEST dataset test trả 6, cần drop name, idx.
    """
    def __init__(self, h5_dir, gene_list, train, fold, k, patch_size):
        self.ds = LightHGGEP_HEST_h5py(h5_dir, gene_list, train=train, fold=fold,
                                       k_neighbors=k, patch_size=patch_size)
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        if self.ds.train:
            # HEST train: (patch, loc, exp, name, idx) → STNet training_step
            # cần (patch, center, exp); center ko quan trọng với STNet forward,
            # dùng loc thay (chỉ pass qua interface).
            patch, loc, exp, _name, _idx = item
            return patch, loc, exp      # 3 phần tử: (patch, center=loc, exp)
        else:
            # HEST test: (patch, loc, exp, center, name, idx) → stnet_predict
            # unpack 4 phần tử (patch, loc, exp, center).
            patch, loc, exp, center, _name, _idx = item
            return patch, loc, exp, center


class HisToGeneSlideDataset(Dataset):
    """Gộp toàn bộ spot của 1 section thành 1 item (slide-level) cho HisToGene.
    Patch crop 112 trung tâm, loc clamp(0,63)."""
    def __init__(self, spot_dataset, section_indices, patch_size=_HISTO_PATCH):
        self.spot_dataset = spot_dataset     # HestBaselineDataset hoặc LightHGGEP
        self.section_indices = section_indices
        self.patch_size = patch_size

    def __len__(self):
        return len(self.section_indices)

    def __getitem__(self, index):
        patches, locations, expressions = [], [], []
        for spot_index in self.section_indices[index]:
            patch, location, expression = self.spot_dataset[spot_index][:3]
            h, w = patch.shape[-2:]
            ps = self.patch_size
            top, left = (h - ps) // 2, (w - ps) // 2
            patch = patch[:, top:top + ps, left:left + ps]
            patches.append(patch.flatten())
            locations.append(location.long().clamp(0, 63))
            expressions.append(expression)
        return torch.stack(patches), torch.stack(locations), torch.stack(expressions)


def split_train_val(ds):
    """Tách mẫu alphabet đầu làm val, còn lại train (spot-level). Đầu vào là
    HestBaselineDataset (train=True). Dùng cumlen/id2name của dataset bên trong."""
    inner = ds.ds
    val_name = sorted(inner.names)[0]
    name2idx = {n: i for i, n in inner.id2name.items()}
    val_i = name2idx[val_name]
    val_start = int(inner.cumlen[val_i - 1]) if val_i > 0 else 0
    val_end = int(inner.cumlen[val_i])
    val_idx = list(range(val_start, val_end))
    train_idx = [i for i in range(len(inner)) if i not in set(val_idx)]
    return train_idx, val_idx, val_name


def _collate_drop_center(batch):
    """Test trả 4 phần tử (patch, loc, exp, center) nhưng model nhận 3
    (patch, center, exp cho STNet / patches, centers, exp cho HisToGene).
    Drop center cho train_step 3 phần tử."""
    return default_collate([item[:3] for item in batch])


def main():
    _p = argparse.ArgumentParser(description="Train+eval STNet/HisToGene trên HEST (LOPO)")
    _p.add_argument('--mode', choices=['stnet', 'histogene', 'all'], default='stnet')
    _p.add_argument('--hest-dir', type=str, default='data/hest/st')
    _p.add_argument('--gene-list', type=str, default='data/her_hvg_cut_1000.npy')
    _p.add_argument('--k', type=int, default=4,
                    help="Chỉ dùng để build graph dataset (baseline ko dùng graph, giữ tham số cho nhất quán)")
    _p.add_argument('--patch-size', type=int, default=224)
    _p.add_argument('--num-workers', type=int, default=0)
    _p.add_argument('--batch-size', type=int, default=None,
                    help="Mặc định: stnet=32, histogene=1 (slide)")
    _p.add_argument('--lr', type=float, default=1e-4)
    _p.add_argument('--max-epochs', type=int, default=None,
                    help="Mặc định: stnet=100, histogene=100")
    _p.add_argument('--patience', type=int, default=15)
    _p.add_argument('--fold-start', type=int, default=0)
    _p.add_argument('--fold-end', type=int, default=None,
                    help="Mặc định = số mẫu HEST (1 fold/mẫu LOPO)")
    _p.add_argument('--ckpt-dir', type=str, default='model_ckpts')
    _p.add_argument('--seed', type=int, default=42)
    args = _p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gene_list = list(np.load(args.gene_list, allow_pickle=True))
    n_genes = len(gene_list)
    print(f"Gene list: {args.gene_list} -> {n_genes} gene")

    all_samples = sorted(os.path.basename(p).split('.')[0]
                         for p in os.listdir(args.hest_dir) if p.endswith('.h5ad'))
    if not all_samples:
        raise FileNotFoundError(f"Không có *.h5ad trong {args.hest_dir}")
    n_fold = args.fold_end if args.fold_end is not None else len(all_samples)
    n_max = len(all_samples)
    if n_fold > n_max:
        print(f"[WARN] --fold-end={n_fold} vượt số mẫu ({n_max}). Giới hạn xuống {n_max}.")
        n_fold = n_max
    if args.fold_start >= n_max:
        raise SystemExit(f"[LỖI] --fold-start={args.fold_start} >= số mẫu {n_max} — không có gì để chạy.")
    modes = ['stnet', 'histogene'] if args.mode == 'all' else [args.mode]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | modes={modes} | n_samples={len(all_samples)}")

    import pandas as pd
    rows = []

    for mode in modes:
        bs = args.batch_size if args.batch_size is not None else (
            1 if mode == 'histogene' else 32)
        max_ep = args.max_epochs if args.max_epochs is not None else 100

        for fold in range(args.fold_start, n_fold):
            test_sample = all_samples[fold]
            print(f"\n{'='*66}\nMODE {mode.upper()} | FOLD {fold} | TEST={test_sample}\n{'='*66}")

            # ---- TRAIN (LOPO) ----
            ds_train = HestBaselineDataset(args.hest_dir, gene_list, train=True,
                                           fold=fold, k=args.k, patch_size=args.patch_size)
            train_idx, val_idx, val_name = split_train_val(ds_train)

            # Bản train có augment; bản val không augment (train=False)
            ds_noaug = HestBaselineDataset(args.hest_dir, gene_list, train=False,
                                           fold=fold, k=args.k, patch_size=args.patch_size)
            ds_noaug.ds.train = False   # tắt augment cho val

            if mode == 'histogene':
                # slide-level
                secs = []
                start = 0
                for i, end in enumerate(ds_train.ds.cumlen):
                    secs.append((ds_train.ds.id2name[i], list(range(start, int(end)))))
                    start = int(end)
                train_sec = [indices for name, indices in secs if name != val_name]
                val_sec = [indices for name, indices in secs if name == val_name]
                train_sub = HisToGeneSlideDataset(ds_train, train_sec)
                val_sub = HisToGeneSlideDataset(ds_noaug, val_sec)
                train_loader = DataLoader(train_sub, batch_size=1, shuffle=True,
                                          num_workers=args.num_workers)
                val_loader = DataLoader(val_sub, batch_size=1, shuffle=False,
                                        num_workers=args.num_workers)
                from models.HisToGene_model import HisToGene
                model = HisToGene(patch_size=_HISTO_PATCH, n_layers=8, n_genes=n_genes,
                                  learning_rate=args.lr, max_epochs=max_ep)
            else:
                # STNet spot-level
                def _apply_index(dsbase, indices):
                    return Subset(dsbase, indices)
                train_sub = _apply_index(ds_train, train_idx)
                val_sub = _apply_index(ds_noaug, val_idx)
                cache = len(ds_train.ds.names)
                ds_train.ds.img_cache_size = cache
                ds_noaug.ds.img_cache_size = cache
                train_loader = DataLoader(train_sub, batch_size=bs, shuffle=True,
                                          num_workers=args.num_workers)
                val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False,
                                        collate_fn=_collate_drop_center,
                                        num_workers=args.num_workers)
                from models.STNet_model import STModel
                model = STModel(n_genes=n_genes, learning_rate=args.lr, max_epochs=max_ep)

            ckpt_dir = os.path.join(args.ckpt_dir, mode)
            os.makedirs(ckpt_dir, exist_ok=True)
            early_stop = EarlyStopping(monitor='valid_loss', patience=args.patience,
                                       mode='min', verbose=False)
            ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,
                                      filename=f"{mode}_hest_fold{fold}_" + "{epoch:02d}",
                                      save_top_k=1, monitor='valid_loss', mode='min',
                                      save_last=False)
            logger = CSVLogger("logs", name=f"hest_{mode}_fold{fold}")

            trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1, max_epochs=max_ep,
                callbacks=[early_stop, ckpt_cb, HestEpochCallback()], logger=logger,
                gradient_clip_val=1.0,
                precision='16-mixed' if torch.cuda.is_available() else '32-true',
                enable_progress_bar=False, enable_model_summary=False,
            )
            if train_loader is not None:
                trainer.fit(model, train_loader, val_loader)

            best_ckpt = ckpt_cb.best_model_path
            best_val = ckpt_cb.best_model_score
            best_val = float(best_val) if best_val is not None else float('nan')
            print(f"  Best ckpt: {best_ckpt} | val_loss={best_val:.4f}")

            # ---- TEST ----
            ds_test = HestBaselineDataset(args.hest_dir, gene_list, train=False,
                                          fold=fold, k=args.k, patch_size=args.patch_size)
            ds_test_raw = ds_test.ds

            if mode == 'histogene':
                from models.HisToGene_model import HisToGene
                m = HisToGene.load_from_checkpoint(best_ckpt, patch_size=_HISTO_PATCH,
                                                   n_layers=8, n_genes=n_genes,
                                                   learning_rate=args.lr, max_epochs=max_ep)
                test_loader = DataLoader(ds_test, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers)
                adata_pred, adata_gt = histogene_predict(m, test_loader, device=device)
            else:
                from models.STNet_model import STModel
                m = STModel.load_from_checkpoint(best_ckpt, n_genes=n_genes,
                                                 learning_rate=args.lr, max_epochs=max_ep)
                test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False,
                                         collate_fn=_collate_drop_center,
                                         num_workers=args.num_workers)
                adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)

            # ---- METRIC ----
            mask = np.abs(adata_gt.X).max(axis=0) > 0
            mask = np.array(mask).flatten()
            n_genes_eval = int(mask.sum())
            subset = [g for g, keep in zip(gene_list, mask) if keep]
            ap = adata_pred[:, mask].copy()
            ag = adata_gt[:, mask].copy()

            section_ids = get_section_ids(ds_test_raw)
            R, _ = get_R(ap, ag, section_ids=section_ids)
            rho, _ = get_Spearman(ap, ag, section_ids=section_ids)
            mse = get_MSE(ap, ag, section_ids=section_ids)
            mae = get_MAE(ap, ag, section_ids=section_ids)

            rows.append({
                'mode': mode, 'fold': fold, 'test_sample': test_sample,
                'val_sample': val_name, 'n_train_samples': len(ds_train.ds.names),
                'n_test_spots': adata_pred.shape[0], 'n_genes_eval': n_genes_eval,
                'pearson': float(np.nanmean(R)),
                'spearman': float(np.nanmean(rho)),
                'rmse': float(np.nanmean(np.sqrt(mse))),
                'mae': float(np.nanmean(mae)),
                'best_val_loss': best_val,
            })
            print(f"  n_spots={adata_pred.shape[0]} | n_genes_eval={n_genes_eval} | "
                  f"pearson={float(np.nanmean(R)):.4f} spearman={float(np.nanmean(rho)):.4f}")

            # giải phóng
            del ds_train, ds_noaug, ds_test, model, adata_pred, adata_gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = "hest_baselines.csv"
    if os.path.isfile(out):
        old = pd.read_csv(out)
        idx = old.set_index(['mode', 'fold']).index
        df = df[~df.set_index(['mode', 'fold']).index.isin(idx)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)

    print("\n" + "="*66)
    print("HEST BASELINES (LOPO) RESULTS")
    print("="*66)
    show = ['mode', 'fold', 'test_sample', 'n_genes_eval', 'pearson', 'spearman', 'rmse', 'mae']
    print(df.to_string(index=False))
    for mode in df['mode'].unique():
        sub = df[df['mode'] == mode]
        for c in ['pearson', 'spearman', 'rmse', 'mae']:
            col = sub[c].dropna()
            if len(col):
                print(f"  [{mode}] {c:9s}: mean={col.mean():.4f} std={col.std():.4f} (n={len(col)})")
    print(f"Đã lưu: {out}")


if __name__ == '__main__':
    main()