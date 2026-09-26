import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WORKDIR = os.path.dirname(os.path.abspath(__file__))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from dataset import LightHGGEP_HER2ST, LightHGGEP_HER2ST_Top250
from models.LightHGGEP import LightHGGEP
from predict import lighthggep_predict
from sampler_utils import SectionBatchSampler, section_collate_fn


def main():
    p = argparse.ArgumentParser(
        description="Ve gene expression: prediction vs ground truth cho 1 lat cat cu the.")
    p.add_argument("--fold", type=int, required=True,
                  help="Fold da train (quyet dinh test patient/split khop luc train).")
    p.add_argument("--ckpt", type=str, required=True,
                  help="Duong dan checkpoint .ckpt cua fold nay.")
    p.add_argument("--section", type=str, required=True,
                  help="Ten lat cat can ve, vd 'B3'. Phai nam trong test_dataset.names cua fold nay.")
    p.add_argument("--gene", default="FASN", help="Ten gene (hoac index int). Mac dinh FASN.")
    p.add_argument("--dataset", choices=["her2st", "her2st_top250"], default="her2st")
    p.add_argument("--k", type=int, default=4, help="k_neighbors -- PHAI khop voi luc train.")
    p.add_argument("--batch-size", type=int, default=4,
                  help="cnn_chunk cua model -- chi anh huong toc do/VRAM, khong doi ket qua.")
    p.add_argument("--out-dir", type=str, default="figures/gene_viz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    DATASET_CLASS = LightHGGEP_HER2ST if args.dataset == "her2st" else LightHGGEP_HER2ST_Top250

    # ----- Load test dataset cho dung fold -----
    test_dataset = DATASET_CLASS(train=False, fold=args.fold, k_neighbors=args.k)
    if args.section not in test_dataset.names:
        raise SystemExit(
            f"Section '{args.section}' khong nam trong tap test cua fold {args.fold}. "
            f"Cac section hop le: {test_dataset.names}")

    n_genes = len(test_dataset.gene_set)
    print(f"  n_genes = {n_genes}, section = {args.section}")

    # ----- Load model -----
    model = LightHGGEP.load_from_checkpoint(
        args.ckpt, n_genes=n_genes, k_neighbors=args.k,
        learning_rate=1e-4, max_epochs=100, cnn_chunk=args.batch_size,
    ).to(device)
    model.eval()

    # Chi can set graph cho dung section can ve
    A_norm = test_dataset.A_norm_cache[args.section]
    model.set_graph(args.section, torch.from_numpy(A_norm).float())

    # ----- DataLoader: chi lay batch cua section nay, full_section=True de dam bao
    # A_norm_batch = A_norm_full (dung cong thuc SGC), giong test_batch_sampler mac
    # dinh cua run_pipeline.py -----
    sampler = SectionBatchSampler(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  include_sections=[args.section], full_section=True)
    loader = DataLoader(test_dataset, batch_sampler=sampler,
                        collate_fn=section_collate_fn, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    # ----- Predict (ca section = 1 batch) -----
    adata_pred, adata_gt = lighthggep_predict(model, loader, device=device)

    centers = adata_pred.obsm['spatial']
    n_spots = centers.shape[0]

    # ----- Chon gene -----
    gene_names = list(test_dataset.gene_set)
    if isinstance(args.gene, str) and args.gene in gene_names:
        gidx = gene_names.index(args.gene)
        gname = args.gene
    else:
        try:
            gidx = int(args.gene)
            gname = gene_names[gidx]
        except (ValueError, IndexError):
            gidx, gname = 0, gene_names[0]
            print(f"  [canh bao] Khong tim thay gene '{args.gene}', dung gene dau tien: {gname}")

    pred_vals = adata_pred.X[:, gidx].astype(float)
    gt_vals = adata_gt.X[:, gidx].astype(float)
    pcc = np.corrcoef(pred_vals, gt_vals)[0, 1]

    # ----- Anh H&E goc (lay truc tiep tu dataset, khong can tim thu muc rieng) -----
    orig_img = test_dataset.get_img(args.section)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pad = 200
    min_x, max_x = centers[:, 0].min() - pad, centers[:, 0].max() + pad
    min_y, max_y = centers[:, 1].min() - pad, centers[:, 1].max() + pad

    axes[0].imshow(orig_img)
    axes[0].set_title(f"(1) Input H&E - {args.section}\n{n_spots} spots")
    axes[0].axis("off")

    axes[1].imshow(orig_img)
    sc1 = axes[1].scatter(centers[:, 0], centers[:, 1], c=pred_vals, cmap="magma",
                          s=40, edgecolors="k", linewidths=0.3, alpha=0.8)
    axes[1].set_title(f"(2) Prediction: {gname}\nmean={pred_vals.mean():.3f}")
    axes[1].axis("off")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(orig_img)
    sc2 = axes[2].scatter(centers[:, 0], centers[:, 1], c=gt_vals, cmap="magma",
                          s=40, edgecolors="k", linewidths=0.3, alpha=0.8)
    axes[2].set_title(f"(3) Ground Truth: {gname}\nmean={gt_vals.mean():.3f}")
    axes[2].axis("off")
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)
        ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"fold{args.fold}_{args.section}_{gname}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  PCC ({gname}) pred vs gt tren section {args.section}: {pcc:.4f}")

    # ----- Luu them CSV so sanh chi tiet tung spot -----
    import pandas as pd
    df = pd.DataFrame({
        "x": centers[:, 0], "y": centers[:, 1],
        f"pred_{gname}": pred_vals, f"gt_{gname}": gt_vals,
    })
    csv_out = os.path.join(args.out_dir, f"fold{args.fold}_{args.section}_{gname}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
