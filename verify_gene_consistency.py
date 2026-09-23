"""verify_gene_consistency.py -- Doi chieu gene_list thuc te giua LightHGGEP va
baseline (ST-Net/HisToGene) SAU KHI ca hai script da chay xong, TRUOC KHI gop
bang so sanh 3 model. Neu lech, dung lai, khong dung so lieu.

Usage:
    python verify_gene_consistency.py --ckpt-dir model_ckpts --gene-panel fixed
"""
import argparse
import glob
import json
import os

_p = argparse.ArgumentParser()
_p.add_argument('--ckpt-dir', type=str, default='model_ckpts')
_p.add_argument('--gene-panel', type=str, default='fixed')
args = _p.parse_args()

gene_dir = os.path.join(args.ckpt_dir, "gene_lists")
files = sorted(glob.glob(os.path.join(gene_dir, f"*_{args.gene_panel}.json")))
if not files:
    raise SystemExit(f"Khong tim thay file gene_list nao trong {gene_dir} voi panel={args.gene_panel}")

by_fold = {}
for path in files:
    name = os.path.basename(path)[:-len(f"_{args.gene_panel}.json")]
    model, fold_part = name.rsplit("_fold", 1)      # vd "lighthggep_fold3" -> ("lighthggep", "3")
    fold = int(fold_part)
    with open(path) as f:
        genes = json.load(f)
    by_fold.setdefault(fold, {})[model] = genes

n_bad = 0
for fold, per_model in sorted(by_fold.items()):
    models = list(per_model.keys())
    ref_model = models[0]
    ref_genes = per_model[ref_model]
    for m in models[1:]:
        if per_model[m] != ref_genes:              # so sanh CA THU TU, khong chi noi dung
            n_bad += 1
            print(f"[LECH] fold={fold}: {m} ({len(per_model[m])} gene) != {ref_model} ({len(ref_genes)} gene)")
    print(f"fold={fold}: {len(models)} model ({models}), {len(ref_genes)} gene")

if n_bad == 0:
    print(f"\n✅ {len(by_fold)} fold co gene_list KHOP nhau giua cac model.")
else:
    raise SystemExit(f"\n❌ {n_bad} truong hop LECH gene_list — KHONG duoc gop bang so sanh.")