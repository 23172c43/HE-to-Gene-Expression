"""Shared batching utilities for Light-HGGEP spatial SGC.

Giữ `SectionBatchSampler` (mỗi batch CHỈ chứa 1 section) và `section_collate_fn`
(giữ `section_name` là str thay vì list) ở module riêng KO side-effect, để cả
`run_pipeline.py` (HER2ST train/eval) lẫn `run_hest_pipeline.py` (external
validation HEST) import chung mà không kéo theo code clone-data/argparse của
run_pipeline.
"""

import math
import random

import torch
from torch.utils.data import Sampler


class SectionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True,
                 include_sections=None, exclude_sections=None,
                 rank=0, num_replicas=1, shard_sections=True,
                 full_section=False):
        # [MỚI - full_section] Chế độ "cả section = 1 batch", dùng riêng cho
        # test_sampler (xem test_sampler bên dưới). Lý do: A_norm_batch =
        # A_norm_full[local_indices][:, local_indices] (LightHGGEP.py forward())
        # chỉ có ý nghĩa "toàn đồ thị lát cắt" như Eq.(9)-(12) mô tả NẾU
        # local_indices bao phủ TOÀN BỘ section. Với batch_size=BATCH_SIZE=32
        # (giá trị cũ), một section 300-700 spot bị cắt thành nhiều batch 32 spot
        # LIÊN TIẾP theo index -- A_norm_batch khi đó chỉ là ma trận con 32x32,
        # bỏ mất các cạnh k-NN trỏ ra ngoài batch. full_section=True bỏ qua
        # batch_size khi cắt batch (mỗi section luôn là đúng 1 batch = toàn bộ
        # N spot), để local_indices = 0..N-1 đầy đủ -> A_norm_batch = A_norm_full
        # thật sự, đúng như lý thuyết. batch_size vẫn được lưu và truyền cho
        # model.cnn_chunk ở nơi khác (không đổi) -- CNN feature extractor vẫn
        # chạy theo nhóm nhỏ bên trong forward(), chỉ có SGC là thấy full graph.
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.full_section = full_section

        self.section_indices = {}
        start = 0
        for i, length in enumerate(dataset.lengths):
            name = dataset.id2name[i]
            self.section_indices[name] = list(range(start, start + length))
            start += length

        if include_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k in set(include_sections)}
        if exclude_sections is not None:
            self.section_indices = {k: v for k, v in self.section_indices.items()
                                     if k not in set(exclude_sections)}

        # A rank owns complete sections: spatial graphs are never split across
        # GPUs.  Greedy bin-packing balances *batch counts*, not merely section
        # counts, because HER2ST sections have very different spot counts.
        self.section_names = list(self.section_indices.keys())
        if full_section:
            # Mỗi section đúng 1 batch, bất kể batch_size.
            self.steps_per_epoch = len(self.section_indices)
        else:
            self.steps_per_epoch = sum(math.ceil(len(v) / self.batch_size)
                                       for v in self.section_indices.values())
        if shard_sections and num_replicas > 1:
            def _batch_count(name):
                return 1 if full_section else math.ceil(
                    len(self.section_indices[name]) / self.batch_size)
            bins = [([], 0) for _ in range(num_replicas)]
            names_by_size = sorted(
                self.section_names, key=_batch_count, reverse=True,
            )
            for name in names_by_size:
                target = min(range(num_replicas), key=lambda i: bins[i][1])
                batch_count = _batch_count(name)
                bins[target][0].append(name)
                bins[target] = (bins[target][0], bins[target][1] + batch_count)
            # DDP needs an identical number of optimizer steps on every rank.
            # The very small excess is dropped after shuffling, so it rotates
            # between sections across epochs instead of permanently omitting one.
            self.section_names = bins[rank][0]
            self.steps_per_epoch = min(total for _, total in bins)

    def __iter__(self):
        section_names = self.section_names.copy()
        if self.shuffle:
            random.shuffle(section_names)
        emitted = 0
        for name in section_names:
            idxs = list(self.section_indices[name])
            if self.shuffle:
                random.shuffle(idxs)
            if self.full_section:
                # Toàn bộ N spot của section trong đúng 1 batch.
                if emitted >= self.steps_per_epoch:
                    return
                yield idxs
                emitted += 1
                continue
            # Cắt thành các batch nhỏ theo BATCH_SIZE
            for s in range(0, len(idxs), self.batch_size):
                if emitted >= self.steps_per_epoch:
                    return
                yield idxs[s:s + self.batch_size]
                emitted += 1

    def __len__(self):
        return self.steps_per_epoch


def section_collate_fn(batch):
    """Thay the default_collate CHI cho truong section_name (str -> giu nguyen 1 chuoi
    thay vi bi goi thanh list). Moi truong khac (patch_3ch/loc/exp/center/local_idx) duoc
    torch.stack() giong het hanh vi mac dinh cua default_collate cho tensor cung shape --
    KHONG doi gia tri/kieu du lieu nao khac ngoai section_name."""
    is_train = (len(batch[0]) == 5)   # train: 5 phan tu; test: 6 phan tu (co them center)
    sec_pos = 3 if is_train else 4

    section_names = [b[sec_pos] for b in batch]
    assert len(set(section_names)) == 1, (
        f"SectionBatchSampler lỗi: 1 batch chứa nhiều section khác nhau {set(section_names)} "
        f"-- Spatial SGC yêu cầu mọi spot trong batch phải cùng 1 section."
    )
    section_name = section_names[0]

    patch_3ch = torch.stack([b[0] for b in batch])
    loc = torch.stack([b[1] for b in batch])
    exp = torch.stack([b[2] for b in batch])

    if is_train:
        local_idx = torch.tensor([b[4] for b in batch], dtype=torch.long)
        return patch_3ch, loc, exp, section_name, local_idx
    else:
        center = torch.stack([b[3] for b in batch])
        local_idx = torch.tensor([b[5] for b in batch], dtype=torch.long)
        return patch_3ch, loc, exp, center, section_name, local_idx