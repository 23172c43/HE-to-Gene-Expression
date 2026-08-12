"""
run_baselines.py -- Pipeline train / predict / eval cho các mô hình baseline trên HER2ST.

Tất cả baseline dùng HER2ST dataset để so sánh fair với LightHGGEP.

Cách dùng:
    python run_baselines.py --mode all          # chạy các baseline tương thích liên tiếp
    python run_baselines.py --mode stnet        # chạy riêng 1 model
    python run_baselines.py --mode histogene

Tùy chọn:
    --fold        : LOOCV fold (default: 5)
    --n_genes     : số gene dự đoán (default: 785)
    --max_epochs  : số epoch tối đa
    --batch_size  : batch size
    --lr          : learning rate (default: 1e-4, shared budget)
    --ckpt_dir    : thư mục lưu checkpoint (default: model_ckpts)
    --ckpt_path   : load checkpoint sẵn, bỏ qua train (chỉ dùng khi mode != all)
    --skip_train  : chỉ predict+eval (chỉ dùng khi mode != all)
    --n_gpus      : số GPU dùng (default: 1, giống Light-HGGEP)
"""
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"   # thêm cùng lúc, cùng họ nguyên nhân
import scanpy as sc
import argparse
import os
import pathlib
import random
import time
import warnings

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ann

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR = str(pathlib.Path(__file__).parent.resolve())
os.chdir(WORKDIR)

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Baseline pipeline cho HER2ST")
parser.add_argument("--mode",       type=str, required=True,
                    choices=["histogene", "stnet", "all"],
                    help="Model muốn chạy. 'all' = chạy các baseline tương thích giao thức chung.")
parser.add_argument("--datasets",   type=str, default="her2st",
                    choices=["her2st", "brainst"],
                    help="Dataset (mac dinh: her2st). brainst = V1_Adult_Mouse_Brain (scanpy).")
parser.add_argument("--fold",       type=int,   default=5)
parser.add_argument("--n_genes",    type=int,   default=785)
parser.add_argument("--max_epochs", type=int,   default=None)
parser.add_argument("--batch_size", type=int,   default=None)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--histogene_lr", type=float, default=None,
                    help="LR riêng cho HisToGene (mặc định: dùng chung --lr nếu không set). "
                         "Lý do cần tách: HisToGene train 1 slide/batch (~15 bước cập nhật/epoch/rank), "
                         "khác hẳn LightHGGEP (batch=32 patch trong section, nhiều bước cập nhật/epoch hơn) "
                         "-- dùng chung LR không tính đến chênh lệch này.")
parser.add_argument("--ckpt_dir",   type=str,   default="model_ckpts")
parser.add_argument("--ckpt_path",  type=str,   default=None,
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--skip_train", action="store_true",
                    help="Chỉ dùng khi --mode không phải 'all'")
parser.add_argument("--n_gpus",     type=int,   default=None,
                    help="Số GPU dùng. Mặc định: 1, giống Light-HGGEP.")
parser.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers trên mỗi DDP rank (default: 2).")
args = parser.parse_args()

FOLD    = args.fold
N_GENES = args.n_genes
LR      = args.lr
NUM_WORKERS = args.num_workers

# Số GPU
n_available = torch.cuda.device_count()
if args.n_gpus is not None:
    N_GPUS = min(args.n_gpus, n_available)
else:
    N_GPUS = 1
N_GPUS = max(N_GPUS, 1)           # ít nhất 1

print("=" * 60)
print(f"BASELINE PIPELINE  mode={args.mode.upper()}  fold={FOLD}")
print("=" * 60)
print(f"  GPU available : {n_available}  →  dùng {N_GPUS} GPU")
print(f"  n_genes       : {N_GENES}")
print(f"  lr            : {LR}")
print(f"  num_workers   : {NUM_WORKERS} / rank")
print("=" * 60)

# ── Imports chung ─────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger

from dataset import HER2ST, BRAINSTSpotDataset, BRAINSTSlideDataset
from evaluation import PROTOCOL_NAME, evaluate_her2st_predictions
from predict import stnet_predict, histogene_predict

# ── Callback: log mỗi epoch ra stdout ────────────────────────────────────────
class EpochProgressBar(Callback):
    """In MSE/PCC train-validation và learning rate sau mỗi epoch."""
    def on_train_start(self, trainer, pl_module):
        # Lightning shards the ordinary DataLoaders with DistributedSampler.
        # Print the effective per-rank work so DDP duplication is immediately
        # visible in Kaggle logs.
        print(f"DDP data shard: rank {trainer.global_rank}/{trainer.world_size}; "
              f"train batches={trainer.num_training_batches}", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started_at = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m        = trainer.callback_metrics
        ep       = trainer.current_epoch + 1
        total    = trainer.max_epochs
        t_mse    = m.get("train_mse", m.get("train_loss_epoch", m.get("train_loss", float("nan"))))
        v_mse    = m.get("val_mse", m.get("val_loss", m.get("valid_loss", float("nan"))))
        t_pcc    = m.get("train_pcc", float("nan"))
        v_pcc    = m.get("val_pcc", float("nan"))
        opt      = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        elapsed = time.perf_counter() - getattr(self, "epoch_started_at", time.perf_counter())
        remaining = max(total - ep, 0) * elapsed
        print(
            f"[{trainer.logger.name}] "
            f"Epoch {ep:3d}/{total}  "
            f"train_mse={float(t_mse):.4f}  train_pcc={float(t_pcc):.4f}  "
            f"val_mse={float(v_mse):.4f}  val_pcc={float(v_pcc):.4f}  "
            f"lr={lr:.2e}  epoch_time={elapsed:.1f}s  eta={remaining / 60:.1f}m",
            flush=True,
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
_default_epochs = {"histogene": 100, "stnet": 100, "uni": 50, "wsuni": 50}
_default_bs     = {"histogene": 1,   "stnet": 32,  "uni": 16, "wsuni": 16}


class HisToGeneSlideDataset(Dataset):
    """Adapt ``HER2ST`` from spot-level samples to HisToGene slide samples.

    HisToGene applies self-attention across all spots in a section, so one dataset
    item must represent one section.  The original HER2ST loader returns a 224 px
    patch per spot, whereas this implementation of HisToGene was built with
    112 px patches.  We take the centred 112 px crop before flattening it.
    """
    def __init__(self, spot_dataset, section_indices):
        self.spot_dataset = spot_dataset
        self.section_indices = section_indices

    def __len__(self):
        return len(self.section_indices)

    def __getitem__(self, index):
        patches, locations, expressions = [], [], []
        for spot_index in self.section_indices[index]:
            item = self.spot_dataset[spot_index]
            patch, location, expression = item[:3]
            # HER2ST patches are (3, 224, 224); HisToGene expects 3 * 112 * 112.
            h, w = patch.shape[-2:]
            top, left = (h - 112) // 2, (w - 112) // 2
            patch = patch[:, top:top + 112, left:left + 112]
            patches.append(patch.flatten())
            locations.append(location.long().clamp(0, 63))
            expressions.append(expression)
        return torch.stack(patches), torch.stack(locations), torch.stack(expressions)

def split_train_val(ds_aug, ds_noaug):
    """
    Tách slide đầu alphabet làm val, còn lại làm train.
    ds_aug   : HER2ST(train=True)  → có augmentation → train_loader
    ds_noaug : HER2ST(train=True) với .train=False → không augment → val_loader
    """
    val_name  = sorted(ds_aug.names)[0]
    name2idx  = {name: i for i, name in ds_aug.id2name.items()}
    val_i     = name2idx[val_name]
    val_start = int(ds_aug.cumlen[val_i - 1]) if val_i > 0 else 0
    val_end   = int(ds_aug.cumlen[val_i])
    val_idx   = list(range(val_start, val_end))
    train_idx = [i for i in range(len(ds_aug)) if i not in set(val_idx)]
    print(f"  Val slide : {val_name} ({len(val_idx)} spots) | "
          f"Train spots: {len(train_idx)}")
    return Subset(ds_aug, train_idx), Subset(ds_noaug, val_idx)


def split_histo_train_val(ds_aug, ds_noaug):
    """Return slide-level train/validation datasets for HisToGene."""
    val_name = sorted(ds_aug.names)[0]
    sections = []
    start = 0
    for i, end in enumerate(ds_aug.cumlen):
        sections.append((ds_aug.id2name[i], list(range(start, int(end)))))
        start = int(end)
    train_sections = [indices for name, indices in sections if name != val_name]
    val_sections = [indices for name, indices in sections if name == val_name]
    print(f"  Val slide : {val_name} ({len(val_sections[0])} spots) | "
          f"Train slides: {len(train_sections)}")
    return (HisToGeneSlideDataset(ds_aug, train_sections),
            HisToGeneSlideDataset(ds_noaug, val_sections))


def collate_drop_center(batch):
    """
    HER2ST(train=False) trả về 4 phần tử (patch, loc, exp, center).
    validation_step của HisToGene/STNet unpack 3 phần tử → drop center.
    """
    return default_collate([item[:3] for item in batch])


# ── BRAIN-ST helpers (V1_Adult_Mouse_Brain, 1 section duy nhất) ────────────────
def brainst_split_train_val(spot_ds, val_frac=0.2, seed=42):
    """Chia spot-level 80/20 trong 1 section duy nhất (BRAINST chỉ có 1 section)."""
    rng = random.Random(seed)
    idx = list(range(len(spot_ds)))
    rng.shuffle(idx)
    n_val = max(1, int(val_frac * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(spot_ds, train_idx), Subset(spot_ds, val_idx), val_idx, train_idx


def brainst_stnet_predict(model, test_loader, device=torch.device("cpu")):
    """Predict cho STNet tren BRAINST. test_loader yields (patch, center, exp)."""
    model.eval(); model = model.to(device)
    preds, gts, centers = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            patch, center, exp = batch
            patch, center = patch.to(device), center.to(device)
            pred = model(patch, center)
            preds.append(pred.cpu()); gts.append(exp); centers.append(center.cpu())
    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    centers = torch.cat(centers).numpy()
    adata_pred = ann.AnnData(preds); adata_pred.obsm["spatial"] = centers
    adata_gt = ann.AnnData(gts); adata_gt.obsm["spatial"] = centers
    return adata_pred, adata_gt


def brainst_histogene_predict(model, test_loader, device=torch.device("cpu")):
    """Predict cho HisToGene tren BRAINST. test_loader yields slide-level
    (patches_flat, centers_clamped, exp) voi batch dim san co (1, N, ...)."""
    model.eval(); model = model.to(device)
    all_p, all_c, all_e = [], [], []
    with torch.no_grad():
        for patch, loc, exp in test_loader:
            all_p.append(patch); all_c.append(loc); all_e.append(exp)
    # Moi phan tu da co batch dim (1, N, ...) -> cat dim0 tao (B, N, ...)
    patches = torch.cat(all_p, 0).to(device)
    locs = torch.cat(all_c, 0).to(device)
    exps = torch.cat(all_e, 0).squeeze(0).numpy()
    centers = locs.squeeze(0).cpu().numpy().astype(float)
    with torch.no_grad():
        pred = model(patches, locs).squeeze(0).cpu().numpy()
    adata_pred = ann.AnnData(pred); adata_pred.obsm["spatial"] = centers
    adata_gt = ann.AnnData(exps); adata_gt.obsm["spatial"] = centers
    return adata_pred, adata_gt


# ─────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH: chạy train + predict + eval cho 1 mode
# ─────────────────────────────────────────────────────────────────────────────
def run_one(mode, fold, n_genes, lr, max_epochs, batch_size,
            ckpt_dir, ckpt_path, skip_train, n_gpus, num_workers):

    from models.HisToGene_model import HisToGene
    from models.STNet_model import STModel

    max_ep = max_epochs if max_epochs is not None else _default_epochs[mode]
    bs     = batch_size if batch_size is not None else _default_bs[mode]
    loader_options = dict(num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=num_workers > 0)
    eval_loader_options = dict(num_workers=num_workers,
                               pin_memory=torch.cuda.is_available(),
                               persistent_workers=False)
    ckpt_out_dir = os.path.join(ckpt_dir, mode)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    os.makedirs("figures/kmeans", exist_ok=True)
    os.makedirs("figures/FASN",   exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MODEL : {mode.upper()}")
    print(f"  epochs={max_ep}  batch={bs}  lr={lr}  gpus={n_gpus}")
    print(f"{'='*60}")

    # ── Strategy cho multi-GPU ────────────────────────────────────────────────
    # DataParallel (dp): chạy trên 1 process, chia batch sang các GPU.
    # Đơn giản, không cần spawn, không conflict với num_workers=0.
    # DDP sẽ nhanh hơn nhưng cần multi-process → phức tạp hơn khi chạy từ script.
    if n_gpus > 1 and not skip_train:
        # HisToGene chứa 1 module normalization cố ý không dùng -> CẦN
        # find_unused_parameters=True để DDP không báo lỗi. Các mode khác (STNet...)
        # không có tham số thừa nào -> dùng "ddp" thường, tránh tốn thêm 1 lượt duyệt
        # toàn bộ đồ thị autograd mỗi bước (đúng cảnh báo PyTorch đã in ra trong log
        # khi chạy STNet với cờ này).
        strategy = "ddp_find_unused_parameters_true" if mode == "histogene" else "ddp"
        accelerator = "gpu"
        devices = n_gpus
    elif torch.cuda.is_available():
        strategy = "auto"
        accelerator = "gpu"
        devices = 1
    else:
        strategy = "auto"
        accelerator = "cpu"
        devices = 1

    # ── Monitor key ───────────────────────────────────────────────────────────
    _monitor = {
        "histogene": "valid_loss",
        "stnet":     "valid_loss",
        "uni":       "val_loss",
        "wsuni":     "val_loss",
    }
    monitor = _monitor[mode]

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    # [BRAIN-ST] Tạo dataset + n_genes override TRƯỚC block train (cần cả khi
    # --skip_train để load đúng kích thước head của model từ checkpoint).
    if args.datasets == "brainst":
        print(f"  [BRAIN-ST] Dung V1_Adult_Mouse_Brain thay vi HER2ST.")
        ds_aug   = BRAINSTSpotDataset(train=True, k_neighbors=4)
        ds_noaug = BRAINSTSpotDataset(train=False, k_neighbors=4)
        # BRAIN-ST chi co 250 gene (Top250) -> ghi de n_genes de khop head cua model
        n_genes = len(ds_aug.brain.gene_set)
        print(f"  [BRAIN-ST] n_genes override: {n_genes}")
        train_subset, val_subset, _, _ = brainst_split_train_val(
            ds_aug, val_frac=0.2, seed=fold)

    if not skip_train:
        logger = CSVLogger("logs", name=f"baseline_{mode}")

        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_out_dir,
            filename=f"{mode}_fold{fold}_" + "{epoch:02d}",
            save_top_k=3,
            monitor=monitor,
            mode="min",
            save_last=True,
        )
        early_stop_cb = EarlyStopping(
            monitor=monitor,
            patience=15,
            mode="min",
            verbose=False,
        )

        if args.datasets == "brainst":
            # Slide-level cho HisToGene, spot-level cho STNet (subset da tinh o tren)
            if mode == "histogene":
                train_loader = DataLoader(BRAINSTSlideDataset(ds_aug, list(range(len(train_subset)))),
                                          batch_size=1, shuffle=True, **loader_options)
                val_loader = DataLoader(BRAINSTSlideDataset(ds_noaug, list(range(len(val_subset)))),
                                        batch_size=1, shuffle=False, **eval_loader_options)
            else:
                train_loader = DataLoader(train_subset, batch_size=bs,
                                          shuffle=True, **loader_options)
                val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False,
                                        collate_fn=collate_drop_center,
                                        **eval_loader_options)
        else:
            ds_aug   = HER2ST(train=True, fold=fold)
            ds_noaug = HER2ST(train=True, fold=fold)
            ds_noaug.train = False

            if mode == "histogene":
                train_subset, val_subset = split_histo_train_val(ds_aug, ds_noaug)
                # Sections have different spot counts, therefore they cannot be
                # stacked together.  One complete section is one training sample.
                train_loader = DataLoader(train_subset, batch_size=1,
                                          shuffle=True, **loader_options)
                val_loader = DataLoader(val_subset, batch_size=1,
                                        shuffle=False, **eval_loader_options)
            else:
                train_subset, val_subset = split_train_val(ds_aug, ds_noaug)
            # [MỚI - fix "treo"] HER2ST._get_img_cached() mặc định img_cache_size=1
            # (dataset.py, dùng chung mọi mode -- KHÔNG sửa file đó để tránh ảnh hưởng
            # HisToGene/LightHGGEP). HisToGeneSlideDataset gộp nguyên 1 section/lần gọi
            # nên cache=1 vẫn khớp (không bị trục xuất giữa chừng); nhưng STNet lấy mẫu
            # theo spot rồi shuffle=True xáo trộn phẳng qua TẤT CẢ section, nên gần như
            # mỗi __getitem__ kế tiếp rơi vào 1 section khác -- cache=1 bị trục xuất và
            # giải mã lại ảnh WSI (Image.open().convert("RGB")) liên tục, cực chậm (trông
            # như treo, không phải deadlock thật). Tăng cache riêng cho 2 instance của
            # nhánh STNet, đủ giữ hết số slide train trong fold này (loại bỏ thrashing),
            # chặn trên để tránh tốn RAM nếu sau này dùng fold có nhiều section hơn.
            # [SỬA - fix vẫn treo] Cache=16 (giới hạn trước) vẫn KHÔNG đủ: với shuffle=True
            # + chỉ 31 slide train, riêng 1 batch=32 spot đã có xác suất rất cao chạm gần
            # hết cả 31 slide (bài toán kiểu birthday-paradox: 32 lần rút ngẫu nhiên trên
            # 31 giá trị). Cache=16 vẫn bị trục xuất NGAY TRONG 1 BATCH, không chỉ giữa các
            # batch -- vẫn giải mã lại ảnh WSI liên tục. RAM còn dư (theo bạn kiểm tra,
            # 14.1/30GiB đang dùng) nên bỏ hẳn giới hạn, cache đủ toàn bộ slide train.
            n_train_slides = len(set(ds_aug.names))
            cache_size = n_train_slides
            ds_aug.img_cache_size = cache_size
            ds_noaug.img_cache_size = cache_size
            print(f"  [Fix cache thrashing] img_cache_size: 1 -> {cache_size} "
                  f"(= toàn bộ {n_train_slides} slide train trong fold này)")
            train_loader = DataLoader(train_subset, batch_size=bs,
                                      shuffle=True, **loader_options)
            val_loader   = DataLoader(val_subset, batch_size=bs,
                                      shuffle=False, collate_fn=collate_drop_center,
                                      **eval_loader_options)

        if mode == "histogene":
            model = HisToGene(patch_size=112, n_layers=8, n_genes=n_genes,
                              learning_rate=lr, max_epochs=max_ep)
        elif mode == "stnet":
            model = STModel(n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            # Explicitly keep Lightning's correct DistributedSampler wrapping
            # for the ordinary baseline DataLoaders under DDP.
            use_distributed_sampler=True,
            max_epochs=max_ep,
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            precision="16-mixed" if accelerator == "gpu" else "32-true",
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint_cb, early_stop_cb, EpochProgressBar()],
        )
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = checkpoint_cb.best_model_path
        print(f"  Best checkpoint: {ckpt_path}")
        print(f"  Best val loss  : {checkpoint_cb.best_model_score:.4f}"
              if hasattr(checkpoint_cb, "best_model_score") else "")

    else:
        if ckpt_path is None:
            raise ValueError(f"--skip_train yêu cầu --ckpt_path cho mode={mode}")

    # Every DDP rank trains; only rank zero may perform the canonical inference
    # and write CSV/figures.  Other ranks wait so modes stay in lockstep.
    if n_gpus > 1 and not skip_train:
        trainer.strategy.barrier()
        if not trainer.is_global_zero:
            return None

    # ── PREDICT ───────────────────────────────────────────────────────────────
    print(f"\n  [PREDICT] Load: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.datasets == "brainst":
        # BRAIN-ST: test = toan bo section (khong LOOCV). Khong co label -> ARI/NMI NaN.
        test_dataset = BRAINSTSpotDataset(train=False, k_neighbors=4)
        label = None
        if mode == "histogene":
            m = HisToGene.load_from_checkpoint(
                ckpt_path, patch_size=112, n_layers=8, n_genes=n_genes,
                learning_rate=lr, max_epochs=max_ep)
            test_loader = DataLoader(BRAINSTSlideDataset(test_dataset),
                                     batch_size=1, shuffle=False, **eval_loader_options)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            _t0 = time.perf_counter()
            adata_pred, adata_gt = brainst_histogene_predict(m, test_loader, device=device)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            inference_time_total_s = time.perf_counter() - _t0
        elif mode == "stnet":
            m = STModel.load_from_checkpoint(
                ckpt_path, n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
            test_loader = DataLoader(test_dataset, batch_size=bs,
                                     shuffle=False, **eval_loader_options)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            _t0 = time.perf_counter()
            adata_pred, adata_gt = brainst_stnet_predict(m, test_loader, device=device)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            inference_time_total_s = time.perf_counter() - _t0
    else:
        test_dataset = HER2ST(train=False, fold=fold)
        label        = test_dataset.label[test_dataset.names[0]]

        if mode == "histogene":
            m = HisToGene.load_from_checkpoint(
                ckpt_path, patch_size=112, n_layers=8, n_genes=n_genes,
                learning_rate=lr, max_epochs=max_ep)
            test_loader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, **eval_loader_options)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            adata_pred, adata_gt = histogene_predict(m, test_loader, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_total_s = time.perf_counter() - _t0

        elif mode == "stnet":
            m = STModel.load_from_checkpoint(
                ckpt_path, n_genes=n_genes, learning_rate=lr, max_epochs=max_ep)
            test_loader = DataLoader(test_dataset, batch_size=bs,
                                     shuffle=False, **eval_loader_options)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            adata_pred, adata_gt = stnet_predict(m, test_loader, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time_total_s = time.perf_counter() - _t0

    n_test_spots = adata_pred.shape[0]
    inference_time_per_spot_ms = 1000.0 * inference_time_total_s / max(n_test_spots, 1)
    print(f"  [INFER TIME] total={inference_time_total_s:.3f}s "
          f"({n_test_spots} spot) -> {inference_time_per_spot_ms:.3f} ms/spot")

    # ── Common, fair evaluation ───────────────────────────────────────────────
    # [BRAIN-ST] dung gene_set cua chinh dataset do (khong co her_hvg_cut_1000.npy)
    if args.datasets == "brainst":
        g = list(test_dataset.brain.gene_set)
    else:
        g = list(np.load("data/her_hvg_cut_1000.npy", allow_pickle=True))
    adata_visual, metrics = evaluate_her2st_predictions(
        adata_pred, adata_gt, g, label=label, n_clusters=4)
    R, p_values = metrics["R"], metrics["p_values"]
    Spearman, spearman_pvalues = metrics["Spearman"], metrics["spearman_pvalues"]
    MSE, MAE, RMSE, morans = metrics["MSE"], metrics["MAE"], metrics["RMSE"], metrics["morans"]
    mean_pcc, median_pcc = metrics["pearson"], metrics["median_pearson"]
    mean_spearman, mean_rmse, mean_mae = metrics["spearman"], metrics["rmse"], metrics["mae"]
    mean_mi_pred, mean_mi_gt = metrics["morans_i_pred"], metrics["morans_i_gt"]
    ARI, NMI = metrics["ARI"], metrics["NMI"]

    print(f"\n  [EVAL] {mode.upper()} fold={fold}")

    print(f"  PCC={mean_pcc:.4f}  Spearman={mean_spearman:.4f}  "
          f"ARI={ARI:.4f}  NMI={NMI:.4f}")
    print(f"  RMSE={mean_rmse:.4f}  MAE={mean_mae:.4f}")
    print(f"  Moran's I pred={mean_mi_pred:.4f}  gt={mean_mi_gt:.4f}")

    # ── VISUALIZE ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(R, bins=30, color="tab:blue", alpha=0.75, edgecolor="black")
    ax.axvline(mean_pcc,   color="red",  ls="--", lw=2,
               label=f"Mean={mean_pcc:.3f}")
    ax.axvline(median_pcc, color="blue", ls="-.", lw=2,
               label=f"Median={median_pcc:.3f}")
    ax.set_xlabel("PCC"); ax.set_ylabel("Gene count")
    ax.set_title(f"{mode.upper()} fold{fold} PCC distribution")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{mode.upper()}_PCC_fold{fold}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.spatial(adata_visual, img=None, color="kmeans", spot_size=112,
                  frameon=False, legend_loc=None, title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/kmeans/{mode.upper()}_kmeans_fold{fold}.png",
                dpi=300, bbox_inches="tight", transparent=True)
    plt.clf(); plt.close()

    # [BRAIN-ST] FASN co the khong co trong Top250 -> chi ve neu co trong var_names
    fasn_gene = "FASN" if "FASN" in adata_visual.var_names else (
        g[0] if g else None)
    if fasn_gene is not None and fasn_gene in adata_visual.var_names:
        sc.pl.spatial(adata_visual, img=None, color=fasn_gene, spot_size=112,
                      color_map="magma", frameon=False, legend_loc=None,
                      title=None, show=False)
        plt.gca().set_title("")
        plt.savefig(f"figures/FASN/{mode.upper()}_FASN_fold{fold}.png",
                    dpi=300, bbox_inches="tight", transparent=True)
        plt.clf(); plt.close()

    # ── LƯU KẾT QUẢ ──────────────────────────────────────────────────────────
    gene_stats = pd.DataFrame({
        "gene":          g,
        "pcc":           R,
        "pcc_pvalue":    p_values,
        "spearman":      Spearman,
        "spearman_pval": spearman_pvalues,
        "mse":           MSE,
        "rmse":          RMSE,
        "mae":           MAE,
    })
    gene_stats.to_csv(f"{mode}_gene_stats_fold{fold}.csv", index=False)

    total_params = sum(p.numel() for p in m.parameters())
    result = {
        "model":         mode.upper(),
        "fold":          fold,
        "pearson":       mean_pcc,
        "spearman":      mean_spearman,
        "ari":           ARI,
        "nmi":           NMI,
        "rmse":          mean_rmse,
        "mae":           mean_mae,
        "morans_i_pred": mean_mi_pred,
        "morans_i_gt":   mean_mi_gt,
        "params":        total_params,
        "inference_time_total_s":   inference_time_total_s,
        "inference_time_per_spot_ms": inference_time_per_spot_ms,
        "n_test_spots":  n_test_spots,
        "ckpt":          ckpt_path,
        "eval_protocol": PROTOCOL_NAME,
        "split_rule":    "LOOCV test=fold; validation=first alphabetical train slide",
        "n_genes":       n_genes,
        "max_epochs":    max_ep,
        "learning_rate": lr,
        "optimizer":     "AdamW(weight_decay=1e-4)",
        "scheduler":     "CosineAnnealingLR(T_max=max_epochs,eta_min=1e-6)",
        "batch_size":    1 if mode == "histogene" else bs,
        "seed":          42,
        "n_gpus":        n_gpus,
        "precision":     "16-mixed" if torch.cuda.is_available() else "32-true",
    }

    summary_csv = "baselines_results.csv"
    new_row = pd.DataFrame([result])
    if os.path.isfile(summary_csv):
        existing = pd.read_csv(summary_csv)
        existing = existing[~((existing["model"] == mode.upper()) &
                               (existing["fold"]  == fold))]
        summary = pd.concat([existing, new_row], ignore_index=True)
    else:
        summary = new_row
    summary.to_csv(summary_csv, index=False)
    print(f"  Saved summary → {summary_csv}")

    if n_gpus > 1 and not skip_train:
        trainer.strategy.barrier()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
ALL_MODES = ["histogene", "stnet"]  # UNI/WSUNI require a distinct multi-scale cached dataset.
modes_to_run = ALL_MODES if args.mode == "all" else [args.mode]

all_results = []
for mode in modes_to_run:
    max_ep = args.max_epochs
    bs     = args.batch_size
    # Khi chạy all, ckpt_path và skip_train không áp dụng
    ckpt_p     = args.ckpt_path  if args.mode != "all" else None
    skip_train = args.skip_train if args.mode != "all" else False

    # [MỚI] HisToGene dùng LR riêng nếu được set qua --histogene_lr, không thì fallback về LR chung
    mode_lr = args.histogene_lr if (mode == "histogene" and args.histogene_lr is not None) else LR

    result = run_one(
        mode       = mode,
        fold       = FOLD,
        n_genes    = N_GENES,
        lr         = mode_lr,
        max_epochs = max_ep,
        batch_size = bs,
        ckpt_dir   = args.ckpt_dir,
        ckpt_path  = ckpt_p,
        skip_train = skip_train,
        n_gpus     = N_GPUS,
        num_workers = NUM_WORKERS,
    )
    if result is not None:
        all_results.append(result)

# ── Bảng tổng kết cuối ───────────────────────────────────────────────────────
if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("TỔNG KẾT TẤT CẢ BASELINE")
    print(f"{'='*70}")
    cols = ["model", "pearson", "spearman", "ari", "nmi", "rmse", "mae",
            "morans_i_pred", "params"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("pearson", ascending=False).to_string(index=False))
    print(f"{'='*70}")