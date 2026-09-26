"""
run_pipeline.py -- Toàn bộ pipeline train/predict/eval/visualize Light-HGGEP trên HER2ST.

File này được TÁCH RA NGUYÊN VẸN từ các cell code của LightHGGEP.ipynb (PHẦN 1, 2, 4, 5,
6, 7, 8, 9 -- KHÔNG bao gồm PHẦN 3, vốn đã tách thành utils.py / predict.py / dataset.py /
models/LightHGGEP.py / models/__init__.py riêng), giữ NGUYÊN THỨ TỰ và NỘI DUNG từng cell.

Thay đổi DUY NHẤT so với notebook gốc: các dòng lệnh IPython bắt đầu bằng "!" (không phải
cú pháp Python hợp lệ trong file .py) được dịch sang subprocess.run(..., shell=True) --
CÙNG một câu lệnh shell, cùng hành vi, không đổi logic. Mỗi vị trí dịch đều có comment
"[DỊCH TỪ IPYTHON]" đánh dấu, kèm câu lệnh gốc để đối chiếu.

CÁCH CHẠY (đã mở rộng cho k-fold CV):
  - Mặc định chạy 5-fold LOOCV (fold 0..4) -- test section = samples[fold] trong
    names[1:33] của dataset.
  - --fold-start / --fold-end: chạy 1 lát cắt (exclusive) -- ví dụ
    `--fold-start 0 --fold-end 2` chỉ chạy fold 0 và 1. Đổi --fold-end=32 để full 32-fold.
  - --datasets: 'her2st' (785 gen) hoặc 'her2st_top250' (250 gen).
  Kết quả TẤT CẢ fold được gộp vào Light-HGGEP_results.csv (1 dòng / fold) và in ra màn hình
  cả từng fold lẫn bảng tổng hợp.
"""
import subprocess  # [MỚI - chỉ để dịch các dòng "!..." của notebook, xem docstring trên]


# ============================================================================
# ---- Cell 3 (notebook gốc) ----
# ============================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
import numpy as np
import torch
import warnings
import pandas as pd  # [MỚI - cần cho bảng tổng hợp kết quả 32-fold]
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
N_GPUS = 1  # LightHGGEP is I/O-bound and faster/more stable on one T4.
print("GPUs used for Light-HGGEP:", N_GPUS)

# ============================================================================
# ---- Cell 4 (notebook gốc) ----
# ============================================================================
# ============================================================
# (Tuy chon) Dang nhap Weights & Biases
# ============================================================
USE_WANDB = False  # doi thanh True neu ban muon dung W&B cua rieng minh

# [MỚI] Logger được tạo RIÊNG trong mỗi fold (tên kèm fold) để không đè lên nhau.
# W&B giữ nguyên logic gốc nếu bật.
wandb_logger = None
if USE_WANDB:
    import wandb
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        from pytorch_lightning.loggers import WandbLogger
        print("Da bat W&B logging.")
    except Exception as e:
        print("Khong the bat W&B (thieu secret WANDB_API_KEY?), tiep tuc voi CSVLogger.", e)
        USE_WANDB = False

from pytorch_lightning.loggers import CSVLogger

# ============================================================================
# ---- Cell 6 (notebook gốc) ----
# ============================================================================
import pathlib

# [SỬA lỗi #6] WORKDIR luôn = thư mục chứa file này (repo root) -- đúng trên mọi môi
# trường (Kaggle, local, Colab). Trước đây Kaggle dùng /kaggle/working nhưng clone repo
# vào /kaggle/working/HE-to-Gene-Expression/ nên path data/... resolve sai.
WORKDIR = str(pathlib.Path(__file__).parent.resolve())
os.chdir(WORKDIR)

# Cac thu muc se duoc tao trong qua trinh chay:
os.makedirs("data", exist_ok=True)
os.makedirs("model_ckpts", exist_ok=True)
os.makedirs("cache_features_train", exist_ok=True)
os.makedirs("cache_features_test", exist_ok=True)
os.makedirs("figures/kmeans", exist_ok=True)
os.makedirs("figures/FASN", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("Working dir:", os.getcwd())

# ============================================================================
# ---- Cell 7 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.1 Clone du lieu HER2ST (chi can chay 1 LAN)
# ----------------------------------------------------------
if not os.path.isdir("data/her2st/.git"):
    # [SỬA lỗi #7] Dùng tham số cwd thay vì "cd data && ..." -- lệnh cd trong subprocess
    # shell=True chạy trong tiến trình con riêng nên không ảnh hưởng thư mục làm việc
    # của Python; dùng cwd= là cách đúng và nhất quán trên cả Linux lẫn Windows.
    # [SỬA đầy ổ đĩa] shallow clone --depth 1 để .git không tốn ~1GB như full clone.
    # [DỊCH TỪ IPYTHON] gốc: !cd data && git clone https://github.com/almaan/her2st.git
    subprocess.run("git clone --depth 1 https://github.com/almaan/her2st.git", shell=True, cwd="data")
else:
    print("data/her2st da ton tai, bo qua clone.")

# ============================================================================
# ---- Cell 8 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.2 Giai nen cac file .tsv.gz trong ST-cnts (chi can chay 1 lan)
# ----------------------------------------------------------
cnt_dir = "data/her2st/data/ST-cnts"
gz_files = [f for f in os.listdir(cnt_dir) if f.endswith(".gz")]
if gz_files:
    subprocess.run(f"cd {cnt_dir} && gunzip -f *.gz", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !cd {cnt_dir} && gunzip -f *.gz
    print(f"Da giai nen {len(gz_files)} file.")
else:
    print("Khong con file .gz (co the da giai nen roi).")

print("So file .tsv trong ST-cnts:", len([f for f in os.listdir(cnt_dir) if f.endswith(".tsv")]))

# ============================================================================
# ---- Cell 10 (notebook gốc) ----
# ============================================================================
# ----------------------------------------------------------
# 2.4 Khoi phuc model_ckpts tu Kaggle Dataset cua session truoc (neu co)
# ----------------------------------------------------------
import glob
import shutil

def restore_dir_from_input(dirname):
    matches = glob.glob(f"/kaggle/input/*/{dirname}")
    if matches:
        src = matches[0]
        dst = os.path.join(WORKDIR, dirname)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Da khoi phuc {dirname} tu {src}")
        return True
    return False

for d in ["model_ckpts"]:
    if not restore_dir_from_input(d):
        print(f"Khong tim thay {d} trong /kaggle/input (binh thuong neu day la lan chay dau tien).")

# ============================================================================
# ---- Cell 19 (notebook gốc, MỞ RỘNG: FOLD -> FOLDS range) ----
# ============================================================================
import argparse

# Chon dataset qua CLI: 'her2st' (785 gen her_hvg_cut_1000) hoac 'her2st_top250'
# (250 gen co muc bieu hien trung binh cao nhat, chon tu count matrix truoc LOOCV split).
_p = argparse.ArgumentParser()
_p.add_argument('--datasets', choices=['her2st', 'her2st_top250'], default='her2st',
                help="Dataset dung cho training/eval (mac dinh: her2st)")
_p.add_argument('--fold-start', type=int, default=0,
                help="Fold dau tien (inclusive). Mac dinh 0.")
_p.add_argument('--fold-end', type=int, default=5,
                help="Fold cuoi (exclusive). Mac dinh 5 = 5-fold LOOCV (fold 0..4).")
_p.add_argument('--skip-train', action='store_true',
                help="Bo qua huan luyen, dung model vua khoi tao de do peak memory / "
                     "thoi gian suy luan (KHONG dung cot pearson/rmse... cua fold nay).")
_p.add_argument('--cnn-chunk', type=int, default=None,
                help="So patch xu ly dong thoi qua CNN (Stage1-3). Mac dinh = BATCH_SIZE "
                     "neu khong truyen. Giam gia tri nay de giam peak memory.")
_p.add_argument('--ablation', choices=['full', 'no_graph', 'no_cross_scale', 'no_depthwise'],
                default='full',
                help="Ablation study: 'full' = day du 3 thanh phan; "
                     "'no_graph' = bo Spatial SGC; "
                     "'no_cross_scale' = bo cross-scale fusion (chi dung scale cuoi); "
                     "'no_depthwise' = thay depthwise separable bang Conv2d thuong.")
_p.add_argument('--k', type=int, default=4,
                help="So lang gieng gan nhat (k) cho do thi K-NN trong Spatial SGC. "
                     "Mac dinh 4. Loi khuyen: chay k trong {2,4,8,12,16} de khao sat.")
_p.add_argument('--batch-sampler', choices=['random', 'spatial_cluster'], default='random',
                help="'random' (mac dinh, hanh vi cu). 'spatial_cluster': gom moi batch "
                     "thanh 1 cum khong gian lien ke (K-means) -- sua loi lech pha "
                     "train/test cua Spatial SGC, cung nguyen nhan da sua ben HEST, muc "
                     "do nhe hon vi section HER2ST it spot hon.")
_p.add_argument('--test-batch-sampler', choices=['spatial_cluster', 'full_section'],
                default='spatial_cluster',
                help="Cách gom batch khi TEST/predict: 'spatial_cluster' (mặc định, "
                     "nhanh + ít VRAM): mỗi batch = cụm không gian liền kề → gần bằng "
                     "full_section. 'full_section': cả section = 1 batch (chính xác tuyệt "
                     "đối nhưng chậm hơn — hành vi cũ).")
_p.add_argument('--fast-eval', action='store_true',
                help="Chỉ tính PCC/Spearman/RMSE/MAE (bỏ Moran's I và t-SNE/PCA — 2 bước "
                     "O(N²) rất chậm). Nhanh hơn nhiều; ARI/NMI và morans_i_* = NaN.")
_args = _p.parse_args()
DATASET = _args.datasets
BATCH_SAMPLER = _args.batch_sampler
TEST_BATCH_SAMPLER = _args.test_batch_sampler
FAST_EVAL = _args.fast_eval
SKIP_TRAIN = _args.skip_train
ABLATION = _args.ablation
# Map ablation name -> 3 co cua model
ABLATION_FLAGS = {
    'full':           dict(use_graph=True,  use_cross_scale=True,  depthwise=True),
    'no_graph':       dict(use_graph=False, use_cross_scale=True,  depthwise=True),
    'no_cross_scale': dict(use_graph=True,  use_cross_scale=False, depthwise=True),
    'no_depthwise':   dict(use_graph=True,  use_cross_scale=True,  depthwise=False),
}
ABL_FLAGS = ABLATION_FLAGS[ABLATION]
N_GENES = None  # tu dong lay tu dataset gene_set neu de None
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
K_NEIGHBORS = _args.k
BATCH_SIZE = 32  # Light-HGGEP rat nhe nen co the tang batch size
CNN_CHUNK = _args.cnn_chunk if _args.cnn_chunk is not None else BATCH_SIZE
NUM_WORKERS = 0  # 0: dùng _SingleProcessDataLoaderIter tránh treo fork on Kaggle/Colab

CKPT_DIR = "model_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

# [MỚI] Lát cắt fold chạy trong lần gọi này. Dataset dùng names[1:33] => đúng 32 fold.
FOLDS = range(_args.fold_start, _args.fold_end)

print(f"Configuration:")
print(f"  FOLDS = {list(FOLDS)} (mac dinh 5-fold, --fold-end=32 de full 32-fold)")
print(f"  DATASET = {DATASET}")
print(f"  N_GENES = {N_GENES}")
print(f"  MAX_EPOCHS = {MAX_EPOCHS}")
print(f"  PATIENCE = {PATIENCE}")
print(f"  LEARNING_RATE = {LEARNING_RATE}")
print(f"  BATCH_SIZE = {BATCH_SIZE}")
print(f"  NUM_WORKERS = {NUM_WORKERS} per DDP rank")
print(f"  BATCH_SAMPLER = {BATCH_SAMPLER}")

# ============================================================================
# ---- Cell 22 (notebook gốc) ----
# ============================================================================
# ==== [MOI - vá 2 lỗi đã phát hiện khi rà soát] ====
# Lỗi 1: DataLoader mac dinh (shuffle=True, khong Sampler/collate_fn rieng) tron lan cac
# spot tu NHIEU section khac nhau vao 1 batch -- vi pham dieu kien bat buoc cua Spatial SGC
# (Eq.2): A_norm_full[local_indices][:, local_indices] chi co y nghia khi TOAN BO
# local_indices trong 1 batch thuoc CUNG 1 do thi/section. Ngoai ra default_collate cua
# PyTorch LUON goi truong section_name (kieu str) thanh 1 LIST (ke ca batch_size=1), khien
# `section_name in self.A_norm_cache` (trong forward()) nem TypeError: unhashable type
# 'list' -- crash ngay batch dau tien, ca luc train LAN luc predict (test_loader).
#
# Lỗi 2: trainer.fit(model, train_loader) khong truyen val_dataloader nao, trong khi
# EarlyStopping/ModelCheckpoint lai theo doi 'val_loss' -- chi so nay khong bao gio duoc
# log vi validation_step() khong bao gio duoc goi.
#
# Cach va: KHONG doi Dataset (dataset.py)/Model (models/LightHGGEP.py) -- chi them 1
# Sampler dam bao moi batch CHI chua 1 section, va 1 collate_fn giu section_name la 1
# CHUOI DUY NHAT thay vi list. Vi Model doc dung 1 chuoi section_name tu batch (khong doi
# forward()/training_step()/validation_step()/test_step()), day la cach va dung o dung lop
# DataLoader, khong dung vao logic model/dataset.

from sampler_utils import SectionBatchSampler, SpatialClusterBatchSampler, section_collate_fn

from pytorch_lightning.callbacks import Callback

class SimpleProgressBar(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started_at = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Validation runs after training in each epoch, so this reports the
        # current epoch's MSE/PCC rather than stale validation metrics.
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        train_mse = trainer.callback_metrics.get('train_mse', trainer.callback_metrics.get('train_loss', 0.0))
        val_mse = trainer.callback_metrics.get('val_mse', trainer.callback_metrics.get('val_loss', 0.0))
        train_pcc = trainer.callback_metrics.get('train_pcc', float('nan'))
        val_pcc = trainer.callback_metrics.get('val_pcc', float('nan'))
        current_epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs
        # [SỬA lỗi #10] trainer.optimizers có thể là [] trong một số phiên bản PL vì
        # optimizers được lazy-init; pl_module.optimizers() luôn trả về optimizer đang
        # hoạt động một cách an toàn hơn.
        opt = pl_module.optimizers()
        # optimizers() có thể trả về list hoặc optimizer đơn tùy phiên bản PL
        if isinstance(opt, list):
            opt = opt[0]
        lr = opt.param_groups[0]['lr']
        elapsed = time.perf_counter() - getattr(self, 'epoch_started_at', time.perf_counter())
        remaining = max(total_epochs - (current_epoch + 1), 0) * elapsed

        # In đúng format bạn muốn
        print(f"[ep {current_epoch + 1}/{total_epochs}] "
              f"train_mse={train_mse:.4f} train_pcc={train_pcc:.4f} "
              f"val_mse={val_mse:.4f} val_pcc={val_pcc:.4f} lr={lr:.4e} "
              f"epoch_time={elapsed:.1f}s eta={remaining / 60:.1f}m")


print("Đã định nghĩa SectionBatchSampler / section_collate_fn (vá lỗi batching + section_name).")


# ============================================================================
# ---- Imports chung (notebook gốc Cell 25/27) -- 1 lần, ngoài loop ----
# ============================================================================
from dataset import LightHGGEP_HER2ST, LightHGGEP_HER2ST_Top250
from models.LightHGGEP import LightHGGEP
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
import scanpy as sc
import matplotlib.pyplot as plt


# Chon class dataset theo DATASET
DATASET_CLASS = LightHGGEP_HER2ST if DATASET == 'her2st' else LightHGGEP_HER2ST_Top250
# N_GENES lay 1 lan (bo gen giong nhau moi fold; base=785 tu file, top250=250 tu count).
if N_GENES is None:
    N_GENES = len(DATASET_CLASS(train=True, fold=0, k_neighbors=K_NEIGHBORS).gene_set)
    print(f"  N_GENES auto = {N_GENES}")


# ============================================================================
# ---- [MỚI] Hàm chạy 1 fold (LOOCV) -- train + eval + figure + results row ----
# ============================================================================
def run_fold(fold):
    import scanpy as sc  # dam bao co san trong scope
    FOLD = fold

    # ----- Cell 25 (notebook gốc): build dataset / loader / model / trainer -----
    train_dataset = DATASET_CLASS(train=True, fold=FOLD, k_neighbors=K_NEIGHBORS)
    # [SỬA - vá lỗi 1+2] Tách 1 slide CỐ ĐỊNH trong 31 slide train làm validation (KHÔNG
    # đụng test_dataset -- giữ đúng nguyên tắc LOOCV: test chỉ dùng 1 lần duy nhất lúc
    # đánh giá cuối, xem PHẦN 6). Chọn theo alphabet cho tái lập được, có thể đổi thủ công
    # nếu muốn slide khác.
    VAL_SECTION = sorted(train_dataset.names)[0]
    # test section cho fold nay = samples[fold], giong logic dataset (names[1:33][fold]).
    _all = sorted(os.listdir('data/her2st/data/ST-cnts'))
    _all = [n[:2] for n in _all]
    _samples = _all[1:33]
    TEST_SECTION = _samples[FOLD]
    print(f"\n{'='*72}\nFOLD {FOLD} | TEST_SECTION={TEST_SECTION} | "
          f"VAL_SECTION={VAL_SECTION} | train_sections={len(train_dataset.names)}\n{'='*72}")

    DDP_RANK = int(os.environ.get("LOCAL_RANK", 0))
    # Kaggle's parent DDP process can construct the rank-0 loader before it
    # exports WORLD_SIZE.  Fall back to the configured device count so rank 0 also
    # receives only its own section shard rather than processing the full dataset.
    DDP_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", N_GPUS))
    if BATCH_SAMPLER == 'spatial_cluster':
        # SpatialClusterBatchSampler chua ho tro rank/num_replicas/shard_sections (xem
        # sampler_utils.py) -- chi an toan vi N_GPUS = 1 co dinh cho HER2ST (Cell 3:
        # "N_GPUS = 1  # LightHGGEP is I/O-bound..."). Neu sau nay tang N_GPUS > 1 cho
        # HER2ST, PHAI bo sung sharding cho class nay truoc, khong duoc dung truc tiep.
        assert N_GPUS == 1, "SpatialClusterBatchSampler chua ho tro multi-GPU sharding."
        train_sampler = SpatialClusterBatchSampler(train_dataset, batch_size=BATCH_SIZE,
                                                   shuffle=True, exclude_sections=[VAL_SECTION])
        val_sampler = SpatialClusterBatchSampler(train_dataset, batch_size=BATCH_SIZE,
                                                 shuffle=False, include_sections=[VAL_SECTION])
    else:
        train_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            exclude_sections=[VAL_SECTION], rank=DDP_RANK,
                                            num_replicas=DDP_WORLD_SIZE, shard_sections=True)
        val_sampler = SectionBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                          include_sections=[VAL_SECTION], rank=DDP_RANK,
                                          num_replicas=DDP_WORLD_SIZE, shard_sections=False)
    print(f"DDP data shard: rank {DDP_RANK}/{DDP_WORLD_SIZE}; "
          f"train sections={len(train_sampler.section_names)}, "
          f"train batches={len(train_sampler)}", flush=True)
    # [SỬA] timeout chỉ set khi num_workers>0: num_workers=0 dùng
    # _SingleProcessDataLoaderIter, PyTorch yêu cầu timeout==0.
    _to = (180 if NUM_WORKERS > 0 else 0)
    loader_options = dict(num_workers=NUM_WORKERS,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=NUM_WORKERS > 0,
                          timeout=_to)
    eval_loader_options = dict(num_workers=NUM_WORKERS,
                               pin_memory=torch.cuda.is_available(),
                               # Avoid keeping train and validation worker caches
                               # alive simultaneously on every DDP rank.
                               persistent_workers=False,
                               timeout=_to)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              collate_fn=section_collate_fn, **loader_options)
    val_loader = DataLoader(train_dataset, batch_sampler=val_sampler,
                            collate_fn=section_collate_fn, **eval_loader_options)

    # Model
    model = LightHGGEP(
        n_genes=N_GENES,
        k_neighbors=K_NEIGHBORS,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        cnn_chunk=CNN_CHUNK,
        **ABL_FLAGS,
    )

    # Set graph cho model
    for section, A_norm in train_dataset.A_norm_cache.items():
        model.set_graph(section, torch.from_numpy(A_norm).float())

    # Tinh so tham so
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Light-HGGEP total parameters: {total_params:,}")
    assert total_params < 300000, f"Light-HGGEP should have <300K params, got {total_params:,}"

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename='lighthggep_fold' + str(FOLD) + '_{epoch:02d}_{val_loss:.4f}',
        # [SỬA đầy ổ đĩa] chỉ giữ 1 best + không save_last để tiết dung lượng Kaggle.
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=False
    )

    # [MỚI] Logger riêng từng fold để không đè lên nhau.
    if USE_WANDB and wandb_logger is not None:
        logger = wandb_logger
    else:
        logger = CSVLogger("logs", name=f"lighthggep_fold{FOLD}")

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=N_GPUS,
        # LightHGGEP uses every parameter in its forward loss path; plain DDP avoids
        # the unnecessary autograd traversal warned about by find_unused_parameters.
        strategy='ddp' if N_GPUS > 1 else 'auto',
        # SectionBatchSampler shards whole spatial sections itself.
        use_distributed_sampler=False,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback, SimpleProgressBar()],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        enable_progress_bar=False,
        enable_model_summary=False,     # Tắt bảng tóm tắt model
    )

    if not SKIP_TRAIN:
        # Train
        trainer.fit(model, train_loader, val_loader)

        # Only rank zero performs the single canonical test evaluation and writes files.
        trainer.strategy.barrier()
        if not trainer.is_global_zero:
            trainer.strategy.barrier()
            raise SystemExit(0)

        # Load best checkpoint
        best_ckpt_path = checkpoint_callback.best_model_path
        print(f"\nBest checkpoint: {best_ckpt_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

        # ----- Cell 27 (notebook gốc): predict + evaluate -----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load best model
        best_model = LightHGGEP.load_from_checkpoint(
            best_ckpt_path,
            n_genes=N_GENES,
            k_neighbors=K_NEIGHBORS,
            learning_rate=LEARNING_RATE,
            max_epochs=MAX_EPOCHS,
            cnn_chunk=CNN_CHUNK,
            **ABL_FLAGS,
        )
    else:
        # [MỚI] Chỉ đo peak memory / thời gian suy luận: dùng đúng kiến trúc + đúng
        # shape dữ liệu thật, KHÔNG train. Bộ nhớ đỉnh và thời gian forward không phụ
        # thuộc giá trị trọng số, chỉ phụ thuộc kiến trúc và shape tensor.
        print("  [SKIP-TRAIN] Dung model vua khoi tao (chua train) de do memory/toc do.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_model = model
        checkpoint_callback.best_model_score = float('nan')

    # Set graph cho model (cho test set)
    test_dataset = DATASET_CLASS(train=False, fold=FOLD, k_neighbors=K_NEIGHBORS)
    for section, A_norm in test_dataset.A_norm_cache.items():
        best_model.set_graph(section, torch.from_numpy(A_norm).float())

    # [SỬA predict nhanh] Test thay vì full_section (cả section = 1 batch rất chậm/nặng
    # với HEST/Visium vài nghìn spot) → dùng SpatialClusterBatchSampler: mỗi batch = 1
    # cụm KHÔNG GIAN liền kề (K-means). Vì SGC chỉ nối k-NN gần, cạnh xuyên cụm gần như
    # không có → kết quả gần bằng full_section nhưng nhanh + tốn ít VRAM hơn nhiều.
    # Chọn test sampler theo --test-batch-sampler (xem help):
    #  'spatial_cluster' (mặc định): mỗi batch = cụm không gian liền kề → nhanh, ít VRAM.
    #  'full_section'     : cả section = 1 batch (chính xác tuyệt đối, chậm hơn).
    # lighthggep_predict vẫn torch.cat các batch lại đúng thứ tự section (shuffle=False).
    if TEST_BATCH_SAMPLER == 'spatial_cluster':
        test_sampler = SpatialClusterBatchSampler(test_dataset, batch_size=BATCH_SIZE,
                                                  shuffle=False)
    else:
        test_sampler = SectionBatchSampler(test_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, full_section=True)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,
                             collate_fn=section_collate_fn, **eval_loader_options)

    # Predict
    from predict import lighthggep_predict
    from evaluation import PROTOCOL_NAME, evaluate_her2st_predictions
    label = test_dataset.get_test_labels()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    adata_pred, adata_gt = lighthggep_predict(best_model, test_loader, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time_total_s = time.perf_counter() - _t0
    peak_inference_memory_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                                 if torch.cuda.is_available() else float('nan'))

    # Common fair evaluation: metrics are always computed on raw log-normalised
    # expression.  Only the visualisation/clustering copy is standardised.
    g = test_dataset.gene_set  # dung dung bo gen cua dataset da chon (785 hoac 250)
    # [SỬA - lỗi Moran's I đa lát cắt] test_dataset (LOPO) co the gom nhieu lat cat cua
    # 1 benh nhan -> phai bao section_ids de get_MoransI khong noi lang gieng xuyen lat
    # cat (xem predict.get_section_ids / predict.get_MoransI).
    from predict import get_section_ids, get_R, get_MSE, get_MAE, get_Spearman
    section_ids = get_section_ids(test_dataset)
    # --fast-eval: bỏ Moran's I (O(N²×top_k)) + t-SNE/PCA (O(N²×iter)) — chỉ giữ 4 metric
    # chính (nhanh, đủ cho so sánh cơ bản). Ngược lại dùng evaluate_her2st_predictions đầy đủ.
    if FAST_EVAL:
        mask_genes = np.array((np.abs(adata_gt.X).max(axis=0) > 0)).flatten()
        n_genes_eval = int(mask_genes.sum())
        ap = adata_pred[:, mask_genes].copy()
        ag = adata_gt[:, mask_genes].copy()
        R, p_values = get_R(ap, ag, section_ids=section_ids)
        Spearman, spearman_pvalues = get_Spearman(ap, ag, section_ids=section_ids)
        MSE = get_MSE(ap, ag, section_ids=section_ids)
        MAE = get_MAE(ap, ag, section_ids=section_ids)
        RMSE = np.sqrt(MSE)
        morans = None          # không tính
        ARI = NMI = float('nan')
    else:
        adata_pred, metrics = evaluate_her2st_predictions(
            adata_pred, adata_gt, g, label=label, n_clusters=4, section_ids=section_ids)
        R, p_values = metrics['R'], metrics['p_values']
        Spearman, spearman_pvalues = metrics['Spearman'], metrics['spearman_pvalues']
        MSE, MAE, RMSE, morans = metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['morans']
        ARI, NMI = metrics['ARI'], metrics['NMI']

    # ==================== IN KẾT QUẢ CHI TIẾT ====================

    print("="*70)
    print(f"KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG - Light-HGGEP trên HER2ST (FOLD {FOLD}, tập test)")
    print("="*70)

    # Thông tin tổng quan
    n_spots = adata_pred.shape[0]
    inference_time_per_spot_ms = 1000.0 * inference_time_total_s / max(n_spots, 1)
    print(f"  [INFER TIME] total={inference_time_total_s:.3f}s "
          f"({n_spots} spot) -> {inference_time_per_spot_ms:.3f} ms/spot")
    print(f"  [PEAK MEM]   {peak_inference_memory_mb:.1f} MB")
    # FAST_EVAL: metrics dict ko tồn tại → tính trực tiếp từ R/Spearman/RMSE/MAE.
    # morans có thể là None (--fast-eval) → guard NaN.
    mean_pcc      = float(np.nanmean(R))
    median_pcc    = float(np.nanmedian(R))
    std_pcc       = float(np.nanstd(R))
    mean_spearman = float(np.nanmean(Spearman))
    mean_rmse     = float(np.nanmean(RMSE))
    mean_mae      = float(np.nanmean(MAE))
    mean_mi_pred  = float(np.nanmean(morans['pred'])) if morans is not None else float('nan')
    mean_mi_gt    = float(np.nanmean(morans['gt']))   if morans is not None else float('nan')

    print(f"  Số spot test đã đánh giá  : {n_spots}")
    print(f"  Số gene đánh giá          : {len(R)}")
    print(f"")
    print(f"  [Correlation]")
    print(f"  Mean Gene-wise PCC        : {mean_pcc:.4f}")
    print(f"  Median Gene-wise PCC      : {median_pcc:.4f}")
    print(f"  Std Gene-wise PCC         : {std_pcc:.4f}")
    print(f"  Mean Gene-wise Spearman   : {mean_spearman:.4f}")
    print(f"")
    print(f"  [Error]")
    print(f"  Mean RMSE                 : {mean_rmse:.4f}")
    print(f"  Mean MAE                  : {mean_mae:.4f}")
    print(f"")
    print(f"  [Spatial structure - top-50 high-var genes]")
    print(f"  Mean Moran's I (pred)     : {mean_mi_pred:.4f}")
    print(f"  Mean Moran's I (gt)       : {mean_mi_gt:.4f}")

    # Tạo DataFrame với thông tin các gene
    gene_stats = pd.DataFrame({
        'gene':          g,
        'pcc':           R,
        'pcc_pvalue':    p_values,
        'spearman':      Spearman,
        'spearman_pval': spearman_pvalues,
        'mse':           MSE,
        'rmse':          RMSE,
        'mae':           MAE,
    })

    # Top-10 gene tốt nhất (PCC cao nhất)
    print("\nTop-10 gen dự đoán TỐT NHẤT (PCC cao nhất):")
    top10_best = gene_stats.nlargest(10, 'pcc')[['gene', 'pcc', 'spearman']]
    print("  " + top10_best.to_string(index=False).replace('\n', '\n  '))

    # Top-10 gene kém nhất (PCC thấp nhất)
    print("\nTop-10 gen dự đoán KÉM NHẤT (PCC thấp nhất):")
    top10_worst = gene_stats.nsmallest(10, 'pcc')[['gene', 'pcc', 'spearman']]
    print("  " + top10_worst.to_string(index=False).replace('\n', '\n  '))

    # Thống kê bổ sung
    print("\n" + "="*70)
    print("THỐNG KÊ BỔ SUNG")
    print("="*70)
    print(f"  Số gene có PCC > 0:  {np.sum(R > 0):,}/{len(R)} ({100*np.sum(R > 0)/len(R):.1f}%)")
    print(f"  Số gene có PCC > 0.2: {np.sum(R > 0.2):,}/{len(R)} ({100*np.sum(R > 0.2)/len(R):.1f}%)")
    print(f"  Số gene có PCC > 0.3: {np.sum(R > 0.3):,}/{len(R)} ({100*np.sum(R > 0.3)/len(R):.1f}%)")

    # ARI + NMI (already computed by the shared protocol on the visualisation copy).
    if label is not None:
        print(f"\n  [Global structure]")
        print(f"  ARI (Adjusted Rand Index) : {ARI:.4f}")
        print(f"  NMI (Norm. Mutual Info)   : {NMI:.4f}")
    else:
        ARI = float('nan')
        NMI = float('nan')
        print("\nARI/NMI: N/A (section này không có ground-truth label)")
    print("="*70)

    # ==================== VẼ HISTOGRAM PCC ====================

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(R, bins=30, color="tab:purple", alpha=0.75, edgecolor="black")
    ax.axvline(mean_pcc, color="red", linestyle="--", linewidth=2, label=f"Mean PCC = {mean_pcc:.3f}")
    ax.axvline(median_pcc, color="blue", linestyle="-.", linewidth=2, label=f"Median PCC = {median_pcc:.3f}")
    ax.set_xlabel("PCC (Pearson Correlation Coefficient)")
    ax.set_ylabel("Số lượng gen")
    ax.set_title(f"Phân bố PCC của Light-HGGEP trên {len(R)} gene (HER2ST test set, fold {FOLD})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    # [MỚI] tên file kèm fold để không đè lên nhau
    plt.savefig(f"figures/Light-HGGEP_PCC_distribution_fold{FOLD}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nĐã lưu biểu đồ PCC distribution vào figures/Light-HGGEP_PCC_distribution_fold{FOLD}.png")

    # Lưu toàn bộ kết quả gene stats vào CSV (kèm fold)
    gene_stats.to_csv(f"gene_predictions_stats_fold{FOLD}.csv", index=False)
    print(f"Đã lưu thống kê chi tiết từng gene vào gene_predictions_stats_fold{FOLD}.csv")

    # ----- Cell 29 (notebook gốc): kmeans / FASN figures (đã có fold trong tên) -----
    # K-means clusters
    sc.pl.spatial(adata_pred, img=None, color="kmeans", spot_size=112,
                  frameon=False, legend_loc=None, title=None, show=False)
    plt.gca().set_title("")
    plt.savefig(f"figures/kmeans/Light-HGGEP_kmeans_fold{FOLD}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.clf()
    plt.close()
    print(f"Saved: figures/kmeans/Light-HGGEP_kmeans_fold{FOLD}.png")

    # FASN gene expression (chỉ khi FASN nằm trong bộ gen đang dùng)
    if "FASN" in adata_pred.var_names:
        sc.pl.spatial(adata_pred, img=None, color="FASN", spot_size=112,
                      color_map="magma", frameon=False, legend_loc=None, title=None, show=False)
        plt.gca().set_title("")
        plt.savefig(f"figures/FASN/Light-HGGEP_FASN_fold{FOLD}.png", dpi=300, bbox_inches="tight", transparent=True)
        plt.clf()
        plt.close()
        print(f"Saved: figures/FASN/Light-HGGEP_FASN_fold{FOLD}.png")

    # ----- Cell 31 (notebook gốc): results row (TRẢ VỀ, không ghi đè) -----
    row = {
        'model':          'Light-HGGEP',
        'ablation':       ABLATION,
        'k_neighbors':    K_NEIGHBORS,
        'fold':           FOLD,
        'val_section':    VAL_SECTION,
        'test_section':   test_dataset.names[0],
        'pearson':        np.nanmean(R),
        'spearman':       np.nanmean(Spearman),
        'ari':            ARI,
        'nmi':            NMI,
        'rmse':           np.nanmean(RMSE),
        'mae':            np.nanmean(MAE),
        'morans_i_pred':  (np.nanmean(morans['pred']) if morans is not None else float('nan')),
        'morans_i_gt':    (np.nanmean(morans['gt'])   if morans is not None else float('nan')),
        'params':         total_params,
        'inference_time_total_s':     inference_time_total_s,
        'inference_time_per_spot_ms': inference_time_per_spot_ms,
        'peak_inference_memory_mb':   peak_inference_memory_mb,
        'cnn_chunk':      CNN_CHUNK,
        'batch_sampler':  BATCH_SAMPLER,
        'n_test_spots':   n_spots,
        'best_val_loss':  float(checkpoint_callback.best_model_score),
        'eval_protocol':  PROTOCOL_NAME,
        'split_rule':     'LOOCV test=fold; validation=first alphabetical train slide',
        'n_genes':        N_GENES,
        'max_epochs':     MAX_EPOCHS,
        'learning_rate':  LEARNING_RATE,
        'optimizer':      'AdamW(weight_decay=1e-4)',
        'scheduler':      'CosineAnnealingLR(T_max=max_epochs,eta_min=1e-6)',
        'batch_size':     BATCH_SIZE,
        'seed':           42,
        'n_gpus':         N_GPUS,
        'precision':      '16-mixed' if torch.cuda.is_available() else '32-true',
    }

    # [MỚI] Giải phóng VRAM giữa các fold để tránh leak.
    del model, best_model, train_dataset, test_dataset, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # [SỬA đầy ổ đĩa] Xóa ckpt của CÁC FOLD TRƯỚC fold hiện tại, GIỮ lại ckpt fold này
    # (best đã load vào best_model, nhưng giữ file để dùng lại predict sau này). Vẫn
    # giải phóng đĩa cho fold tiếp theo trên Kaggle mà không mất trọng số fold vừa chạy.
    if os.path.isdir(CKPT_DIR):
        for fn in os.listdir(CKPT_DIR):
            if fn.startswith("lighthggep_fold") and fn.endswith(".ckpt"):
                try:
                    fnum = int(fn.replace("lighthggep_fold", "").split("_")[0])
                except (IndexError, ValueError):
                    continue
                if fnum < FOLD:
                    try:
                        os.remove(os.path.join(CKPT_DIR, fn))
                    except OSError:
                        pass
        kept = [f for f in os.listdir(CKPT_DIR)
                if f.startswith(f"lighthggep_fold{FOLD}_")]
        print(f"  [space] kept fold-{FOLD} ckpt(s): {kept}")

    return row


# ============================================================================
# ---- [MỚI] Main loop: chạy từng fold, gộp CSV, in bảng tổng hợp ----
# ============================================================================
all_rows = []
for fold in FOLDS:
    all_rows.append(run_fold(fold))

results = pd.DataFrame(all_rows)
# [SỬA ghi đè] Append an toàn: giữ các fold cũ trong CSV, chỉ nối fold mới (xóa trùng
# theo cột 'fold'), để chạy lẻ từng fold (vd --fold-start 5 --fold-end 6) không mất
# kết quả các fold đã chạy trước đó. Khớp logic append của run_baselines.py.
summary_csv = (f"Light-HGGEP_memoryprofile_chunk{CNN_CHUNK}.csv" if SKIP_TRAIN
            else f"Light-HGGEP_results_{BATCH_SAMPLER}.csv")
if os.path.isfile(summary_csv):
    old = pd.read_csv(summary_csv)
    old = old[~old["fold"].isin(results["fold"])]
    results = pd.concat([old, results], ignore_index=True)
results.to_csv(summary_csv, index=False)
print("\n" + "="*72)
print(f"FINAL AGGREGATED RESULTS -- Light-HGGEP {DATASET} ({len(results)} folds)")
print("="*72)
# In bảng metric chính từng fold
show_cols = ['fold', 'test_section', 'pearson', 'spearman', 'rmse', 'mae', 'ari', 'nmi', 'best_val_loss', 'n_test_spots']
print(results[show_cols].to_string(index=False))
print("\n--- MEAN ± STD across folds ---")
for c in ['pearson', 'spearman', 'rmse', 'mae', 'ari', 'nmi', 'best_val_loss']:
    col = results[c].dropna()
    if len(col):
        print(f"  {c:16s}: mean={col.mean():.4f}  std={col.std():.4f}  (n={len(col)})")
print("="*72)
print("\nDa luu ket qua tong hop vao Light-HGGEP_results.csv")

# ============================================================================
# ---- Cell 33 (notebook gốc) ----
# ============================================================================
print("Checkpoints da luu trong:")
subprocess.run(f"ls -la {CKPT_DIR}", shell=True)  # [DỊCH TỪ IPYTHON] gốc: !ls -la {CKPT_DIR}
