"""dataset_hest.py — Dataset HEST (.h5ad) cho HisToGene và ST-Net.

Đảm bảo fairness hoàn toàn với LightHGGEP_HEST_h5py trong dataset.py:
  • LOPO split  : sorted(all_samples)[fold] = test, KHÔNG dùng random.shuffle
  • Normalization: CP1M + log1p — COPY CHÍNH XÁC từ LightHGGEP_HEST_h5py._load_sample
  • Augmentation : ColorJitter(0.5,0.5,0.5) + RandomHFlip + RandomRotation(180),
                   áp lên PIL uint8 TRƯỚC khi normalize (giống LightHGGEP HEST)
  • ImageNet normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  • Gene thiếu  : fill 0, shape luôn (N, n_genes) — lọc cột toàn-0 khi eval (script)
  • Gene set    : truyền vào từ ngoài (785 gene từ her_hvg_cut_1000.npy)

KHÔNG sửa dataset.py / evaluation.py / run_hest_train.py / sampler_utils.py.

Classes:
  HisToGene_HEST_h5py — slide-level (1 __getitem__ = 1 slide = toàn bộ N spot)
  STNet_HEST_h5py     — spot-level  (1 __getitem__ = 1 spot, giống LightHGGEP_HEST_h5py)
"""

import glob
import json
import os

import cv2
import h5py
import numpy as np
import scipy.sparse as sp
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ─── Hằng số ImageNet (dùng chung cho cả 3 model trên HEST) ──────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Augmentation transform: giống hệt LightHGGEP_HEST_h5py ─────────────────
_TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.5, 0.5, 0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=180),
])


# ══════════════════════════════════════════════════════════════════════════════
#  HÀM DÙNG CHUNG — đọc 1 file .h5ad
#  (Copy chính xác từ LightHGGEP_HEST_h5py._load_sample để đảm bảo fairness)
# ══════════════════════════════════════════════════════════════════════════════

def _load_hest_sample(path: str, target_genes: list) -> dict:
    """Đọc 1 file .h5ad → dict {exp, loc, center, img, r_img}.

    Logic COPY NGUYÊN từ LightHGGEP_HEST_h5py._load_sample:
      - exp    : CP1M + log1p, gene thiếu = 0, shape (N, n_genes) float32
      - loc    : tọa độ thực fullres (N, 2) float64
      - center : pixel trên ảnh thu nhỏ = floor(loc * scale), (N, 2) int
      - img    : ảnh H&E thu nhỏ uint8 (H, W, 3), contiguous
      - r_img  : bán kính crop = round(spot_dia * scale * 0.75), int ≥ 1

    Sửa nhỏ so với bản gốc: xử lý đúng khi X là dense numpy array
    (gốc gọi .toarray() không điều kiện → lỗi nếu X dense).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, 'r') as f:

        # ── Tên gene trong file ──────────────────────────────────────────────
        var_group = f['var']
        index_col = var_group.attrs.get('_index', '_index')
        raw_genes  = var_group[index_col][:]
        var_names  = np.array([g.decode('utf-8') if isinstance(g, bytes) else g
                                for g in raw_genes])

        # ── Ma trận biểu hiện (CSR sparse hoặc dense) ───────────────────────
        X_node = f['X']
        shape  = tuple(X_node.attrs.get('shape')) if 'shape' in X_node.attrs else None
        if isinstance(X_node, h5py.Group) and 'data' in X_node:
            # Định dạng CSR được lưu theo nhóm h5py (phổ biến với anndata)
            X_mat = sp.csr_matrix(
                (X_node['data'][:], X_node['indices'][:], X_node['indptr'][:]),
                shape=shape,
            )
        else:
            # Dense dataset hoặc sparse đã flatten
            X_mat = X_node[:]
            if sp.issparse(X_mat):
                X_mat = X_mat.tocsr()
        N = X_mat.shape[0]

        # ── Biểu hiện gene: CP1M + log1p, fill 0 cho gene thiếu ─────────────
        gene2col  = {g: i for i, g in enumerate(var_names)}
        spot_sums = np.asarray(X_mat.sum(axis=1)).flatten()
        spot_sums[spot_sums == 0] = 1.0                          # tránh chia 0
        exp = np.zeros((N, len(target_genes)), dtype=np.float32)
        n_common = 0
        for j, g in enumerate(target_genes):
            col = gene2col.get(g)
            if col is None:
                continue                                          # gene vắng → cột 0
            # Lấy cột đúng cách cho cả sparse lẫn dense
            col_vec = (X_mat[:, col].toarray().flatten()
                       if sp.issparse(X_mat)
                       else np.asarray(X_mat[:, col]).flatten())
            exp[:, j] = np.log1p((col_vec / spot_sums) * 1e6)   # CP1M + log1p
            n_common += 1
        print(f"[HEST] {name}: {n_common}/{len(target_genes)} gene có mặt; N={N}")

        # ── Tọa độ spatial (pixel fullres) ───────────────────────────────────
        coords = np.asarray(f['obsm']['spatial'][:], dtype=np.float64)  # (N, 2)

        # ── Scale factor và đường kính spot ──────────────────────────────────
        spatial_grp = f['uns']['spatial']
        internal_id = list(spatial_grp.keys())[0]
        sf_grp = spatial_grp[internal_id]['scalefactors']
        if isinstance(sf_grp, h5py.Group):
            scale    = float(sf_grp['tissue_downscaled_fullres_scalef'][()])
            spot_dia = float(sf_grp['spot_diameter_fullres'][()])
        else:
            raw     = sf_grp[()]
            sf_json = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            scale    = float(sf_json['tissue_downscaled_fullres_scalef'])
            spot_dia = float(sf_json['spot_diameter_fullres'])

        # ── Ảnh H&E thu nhỏ ──────────────────────────────────────────────────
        imgs_grp = spatial_grp[internal_id]['images']
        img = (imgs_grp['downscaled_fullres'][:]
               if 'downscaled_fullres' in imgs_grp
               else imgs_grp['hires'][:])
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]                  # RGBA → RGB
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        img = np.ascontiguousarray(img)

        # ── Bán kính crop = 1.5 × bán kính spot trên ảnh thu nhỏ ────────────
        r_img = int(round(spot_dia * scale * 0.75))
        r_img = max(r_img, 1)

    return {
        'exp'   : exp,
        'loc'   : coords,
        'center': np.floor(coords * scale).astype(int),   # pixel trên ảnh thu nhỏ
        'img'   : img,
        'r_img' : r_img,
    }


def _crop_and_normalize(img: np.ndarray, center, r_img: int,
                         patch_size: int, aug_transforms=None) -> torch.Tensor:
    """Crop 1 spot từ ảnh H&E, resize về patch_size × patch_size, ImageNet normalize.

    Giống hệt LightHGGEP_HEST_h5py._crop_patch để đảm bảo fairness:
      1. Crop vùng [y-r : y+r, x-r : x+r] với padding BORDER_REPLICATE nếu ra ngoài ảnh
      2. Augment (nếu aug_transforms != None) trên PIL uint8 TRƯỚC khi normalize
      3. Resize về (patch_size, patch_size) bằng INTER_CUBIC
      4. Normalize ImageNet (chuẩn /255, trừ mean, chia std)

    Args:
        img           : numpy uint8 (H, W, 3) — ảnh H&E thu nhỏ
        center        : [x, y] pixel (int) trên ảnh thu nhỏ
        r_img         : bán kính crop (int)
        patch_size    : kích thước output (112 cho HisToGene, 224 cho STNet)
        aug_transforms: torchvision Compose áp lên PIL Image (None = không augment)

    Returns:
        tensor float32 (3, patch_size, patch_size) đã ImageNet normalize
    """
    x, y = int(center[0]), int(center[1])
    h, w = img.shape[:2]

    top, bottom = y - r_img, y + r_img
    left, right = x - r_img, x + r_img

    crop = img[max(top, 0):min(bottom, h), max(left, 0):min(right, w)]

    # Padding nếu crop ra ngoài biên ảnh
    pad_top    = max(-top, 0)
    pad_bottom = max(bottom - h, 0)
    pad_left   = max(-left, 0)
    pad_right  = max(right - w, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  cv2.BORDER_REPLICATE)
    crop = crop[:2 * r_img, :2 * r_img]   # đảm bảo kích thước 2r×2r

    # Augment trên uint8 PIL TRƯỚC khi normalize (giống LightHGGEP HEST)
    if aug_transforms is not None:
        crop = np.array(aug_transforms(Image.fromarray(crop)))

    # Resize → normalize ImageNet
    resized = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    t = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    t[0] = (t[0] - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
    t[1] = (t[1] - _IMAGENET_MEAN[1]) / _IMAGENET_STD[1]
    t[2] = (t[2] - _IMAGENET_MEAN[2]) / _IMAGENET_STD[2]
    return torch.from_numpy(t).float()


# ══════════════════════════════════════════════════════════════════════════════
#  HisToGene_HEST_h5py — Dataset slide-level cho HisToGene
# ══════════════════════════════════════════════════════════════════════════════

class HisToGene_HEST_h5py(Dataset):
    """Dataset HEST (.h5ad) slide-level cho HisToGene.

    HisToGene dùng self-attention toàn slide → 1 __getitem__ = 1 slide (toàn bộ
    N spot). Dùng với DataLoader(batch_size=1, shuffle=True/False).

    Tuple trả về:
      • train (self.train=True):
          patches_flat : (N, 3*112*112) float32   — patch đã flatten
          locs_long    : (N, 2) long               — tọa độ min-max scale [0,63] per-slide
          exps         : (N, n_genes) float32

      • test (self.train=False):
          patches_flat : (N, 3*112*112) float32
          locs_long    : (N, 2) long
          exps         : (N, n_genes) float32
          centers_pixel: (N, 2) float32 — pixel trên ảnh thu nhỏ (cho obsm['spatial'])
          section_name : str             — tên mẫu (cho section_ids)

    Tọa độ locs_long: min-max scale per-slide về [0.0, 63.0] → long().clamp(0,63).
    Lý do per-slide: tọa độ HEST là pixel lớn (hàng nghìn), đưa thẳng vào
    nn.Embedding(n_pos=64) sẽ bị clamp về 63 toàn bộ → mất thông tin vị trí.

    Attributes cần cho get_section_ids() (predict.py):
      lengths  : list[int]     — số spot thực tế trả về mỗi slide (≤ max_spots)
      id2name  : dict[int→str]

    Args:
        h5_dir    : thư mục chứa *.h5ad
        gene_list : danh sách tên gene (thứ tự cố định, ví dụ 785 gene)
        train     : True = split train, False = split test
        fold      : chỉ số fold LOPO (test = sorted(all_samples)[fold])
        patch_size: kích thước patch output (mặc định 112 cho HisToGene)
        max_spots : giới hạn số spot/slide (tránh OOM với slide lớn).
                    Khi N > max_spots, lấy max_spots spot ĐẦU TIÊN (không shuffle).
                    None = không giới hạn.
    """

    def __init__(self, h5_dir: str, gene_list: list, train: bool = True,
                 fold: int = 0, patch_size: int = 112, max_spots: int = 700):
        super().__init__()
        self.h5_dir       = h5_dir
        self.target_genes = list(gene_list)
        self.gene_set     = list(gene_list)   # alias cho compatibility
        self.patch_size   = patch_size
        self.max_spots    = max_spots
        self.train        = train

        # ── LOPO split: giống LightHGGEP_HEST_h5py ──────────────────────────
        paths = sorted(glob.glob(os.path.join(h5_dir, '*.h5ad')))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy *.h5ad trong {h5_dir}")
        all_samples = sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)

        test_sample = all_samples[fold % len(all_samples)]
        tr_names    = [s for s in all_samples if s != test_sample]
        te_names    = [s for s in all_samples if s == test_sample]
        self.names  = tr_names if train else te_names
        print(f"[HEST] HisToGene {'TRAIN' if train else 'TEST'} fold={fold} | "
              f"test={test_sample} | {'n_train' if train else 'n_test'}={len(self.names)}")

        # ── Load tất cả slide vào RAM ────────────────────────────────────────
        # Mỗi sample là dict {exp, loc, center, img, r_img}
        self.samples: dict = {}
        for name in self.names:
            p = os.path.join(h5_dir, name + '.h5ad')
            self.samples[name] = _load_hest_sample(p, self.target_genes)

        # ── Metadata cho get_section_ids (predict.py) ────────────────────────
        # lengths[i] = số spot THỰC TẾ mà __getitem__(i) trả về (≤ max_spots)
        # Phải khớp với số spot trong adata_pred/adata_gt để section_ids đúng thứ tự
        self.id2name = {i: n for i, n in enumerate(self.names)}
        self.lengths = []
        for n in self.names:
            n_raw = len(self.samples[n]['loc'])
            cap   = self.max_spots if self.max_spots is not None else n_raw
            self.lengths.append(min(n_raw, cap))
        self.cumlen = np.cumsum(self.lengths)

        # ── Augmentation (áp lên PIL uint8 TRƯỚC normalize) ──────────────────
        self.aug_transforms = _TRAIN_TRANSFORMS

    def __len__(self) -> int:
        """Số slide trong split hiện tại."""
        return len(self.names)

    def __getitem__(self, i: int):
        """Trả về toàn bộ N spot của slide i dưới dạng tensor slide-level.

        N bị giới hạn bởi max_spots (lấy N spot đầu, không shuffle).
        """
        name   = self.id2name[i]
        sample = self.samples[name]
        N_raw  = len(sample['loc'])

        # Giới hạn số spot (không shuffle — đảm bảo tái hiện được)
        if self.max_spots is not None and N_raw > self.max_spots:
            idx_sel = np.arange(self.max_spots)
        else:
            idx_sel = np.arange(N_raw)
        N = len(idx_sel)

        # Augment chỉ khi self.train = True (kiểm tra lúc gọi — tương thích với
        # pattern `ds_noaug.train = False` dùng để tắt augment cho val subset)
        aug = self.aug_transforms if self.train else None

        # ── Crop + augment + resize 112×112 + normalize từng spot ────────────
        patch_list = []
        for si in idx_sel:
            patch = _crop_and_normalize(
                sample['img'], sample['center'][si], sample['r_img'],
                self.patch_size, aug
            )
            patch_list.append(patch.view(-1))   # (3*112*112,) = (37632,)
        patches_flat = torch.stack(patch_list)   # (N, 37632)

        # ── Biểu hiện gene ───────────────────────────────────────────────────
        exps = torch.tensor(sample['exp'][idx_sel], dtype=torch.float32)  # (N, n_genes)

        # ── Tọa độ: min-max scale per-slide → [0, 63] → long ────────────────
        loc_raw = sample['loc'][idx_sel]          # (N, 2) float64
        loc_min = loc_raw.min(axis=0)
        loc_max = loc_raw.max(axis=0)
        denom   = (loc_max - loc_min) + 1e-8      # tránh chia 0 nếu slide 1 spot
        loc_scaled = (loc_raw - loc_min) / denom * 63.0
        locs_long  = torch.tensor(
            np.floor(loc_scaled).astype(np.int64), dtype=torch.long
        ).clamp(0, 63)                             # (N, 2)

        if self.train:
            # train/val tuple: 3 phần tử — khớp HisToGene.training_step / validation_step
            return patches_flat, locs_long, exps
        else:
            # test tuple: 5 phần tử — thêm center (pixel) và section_name cho predict
            centers_pixel = torch.tensor(
                sample['center'][idx_sel], dtype=torch.float32
            )
            return patches_flat, locs_long, exps, centers_pixel, name


# ══════════════════════════════════════════════════════════════════════════════
#  STNet_HEST_h5py — Dataset spot-level cho ST-Net
# ══════════════════════════════════════════════════════════════════════════════

class STNet_HEST_h5py(Dataset):
    """Dataset HEST (.h5ad) spot-level cho ST-Net.

    Giống cấu trúc LightHGGEP_HEST_h5py (iterate qua cumlen/id2name) nhưng:
      • Không build A_norm_cache (STNet không dùng Spatial SGC)
      • patch_size = 224 (giống STNet trên HER2ST)
      • Tuple __getitem__:
          train: (patch, loc, exp)                        — 3 phần tử
          test : (patch, loc, exp, center_t, section_name)— 5 phần tử

    Attributes cần cho get_section_ids() (predict.py):
      lengths  : list[int]
      id2name  : dict[int→str]

    Args:
        h5_dir    : thư mục chứa *.h5ad
        gene_list : danh sách tên gene (785 gene)
        train     : True = split train, False = split test
        fold      : chỉ số fold LOPO
        patch_size: kích thước patch output (mặc định 224)
    """

    def __init__(self, h5_dir: str, gene_list: list, train: bool = True,
                 fold: int = 0, patch_size: int = 224):
        super().__init__()
        self.h5_dir       = h5_dir
        self.target_genes = list(gene_list)
        self.gene_set     = list(gene_list)
        self.patch_size   = patch_size
        self.train        = train

        # ── LOPO split: giống LightHGGEP_HEST_h5py ──────────────────────────
        paths = sorted(glob.glob(os.path.join(h5_dir, '*.h5ad')))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy *.h5ad trong {h5_dir}")
        all_samples = sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)

        test_sample = all_samples[fold % len(all_samples)]
        tr_names    = [s for s in all_samples if s != test_sample]
        te_names    = [s for s in all_samples if s == test_sample]
        self.names  = tr_names if train else te_names
        print(f"[HEST] STNet {'TRAIN' if train else 'TEST'} fold={fold} | "
              f"test={test_sample} | {'n_train' if train else 'n_test'}={len(self.names)}")

        # ── Load metadata từng mẫu ───────────────────────────────────────────
        self.meta_dict: dict = {}
        for name in self.names:
            p = os.path.join(h5_dir, name + '.h5ad')
            self.meta_dict[name] = _load_hest_sample(p, self.target_genes)

        # Tách dict theo key để truy cập nhanh trong __getitem__
        self.exp_dict    = {n: m['exp']    for n, m in self.meta_dict.items()}
        self.center_dict = {n: m['center'] for n, m in self.meta_dict.items()}
        self.loc_dict    = {n: m['loc']    for n, m in self.meta_dict.items()}
        self.img_dict    = {n: m['img']    for n, m in self.meta_dict.items()}

        # ── Metadata cho get_section_ids (predict.py) ────────────────────────
        self.lengths = [len(m['loc']) for m in self.meta_dict.values()]
        self.cumlen  = np.cumsum(self.lengths)
        self.id2name = {i: n for i, n in enumerate(self.names)}

        # ── Augmentation ─────────────────────────────────────────────────────
        self.aug_transforms = _TRAIN_TRANSFORMS

    def __len__(self) -> int:
        """Tổng số spot trong tất cả sample của split."""
        return int(self.cumlen[-1])

    def __getitem__(self, index: int):
        """Trả về 1 spot (spot-level, giống HER2ST STNet)."""
        # Tìm section index và local spot index
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index if i == 0 else index - int(self.cumlen[i - 1])

        name     = self.id2name[i]
        meta     = self.meta_dict[name]
        center   = self.center_dict[name][idx]

        # Augment chỉ khi self.train = True (kiểm tra lúc gọi)
        aug   = self.aug_transforms if self.train else None
        patch = _crop_and_normalize(self.img_dict[name], center,
                                     meta['r_img'], self.patch_size, aug)

        exp      = torch.tensor(self.exp_dict[name][idx],  dtype=torch.float32)
        loc      = torch.tensor(self.loc_dict[name][idx],  dtype=torch.float32)
        center_t = torch.tensor(center,                     dtype=torch.float32)

        if self.train:
            # train/val tuple: 3 phần tử — khớp STModel.training_step / validation_step
            return patch, loc, exp
        else:
            # test tuple: 5 phần tử — thêm center và section_name cho predict + section_ids
            return patch, loc, exp, center_t, name
