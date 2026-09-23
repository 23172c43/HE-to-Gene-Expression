import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import scipy.sparse as sp
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import albumentations as A
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()  
        self.names = [i[:2] for i in self.names]
        self.train = train
        # Leave-One-Patient-Out (LOPO) split
        samples = self.names # Giữ nguyên danh sách mẫu hợp lệ của bạn

        # Trích xuất danh sách các bệnh nhân duy nhất (ký tự đầu tiên của chuỗi, vd: 'A', 'B', 'C'...)
        patients = sorted(list(set([name[0] for name in samples])))

        # Lấy tên bệnh nhân cho tập Test dựa vào biến fold
        # Dùng phép chia lấy dư (%) để tránh lỗi index out of range nếu fold truyền vào lớn hơn số bệnh nhân
        test_patient = patients[fold % len(patients)]

        # Tách tập Test (tất cả các lát cắt của bệnh nhân test) và Train (các bệnh nhân còn lại)
        te_names = [name for name in samples if name[0] == test_patient]
        tr_names = [name for name in samples if name[0] != test_patient]

        print(f"LOPO Split - Bệnh nhân Test: {test_patient} | Số mẫu Test: {len(te_names)} | Số mẫu Train: {len(tr_names)}")

        if train:
            self.names = tr_names
        else:
            self.names = te_names
        print('Registering image paths...')
        # DDP launches one dataset instance per rank.  Keeping all decoded WSI
        # images alive in each process exhausts host RAM, so retain only a tiny
        # LRU cache and open a slide on demand.
        self.img_paths = {i: self.get_img_path(i) for i in self.names}
        self.img_dict = OrderedDict()
        self.img_cache_size = 1
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train:
            for name in self.names:
                if name in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl_full = self.get_lbl(name)
                    idx = self.meta_dict[name].index
                    lbl = lbl_full.loc[idx, :]['label'].values
                    self.label[name] = lbl
                # Lát cắt không có file annotation: self.label[name] giữ nguyên None
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        # [SỬA - đồng bộ chuẩn hóa với PixNet] Trước đây: library_size_normalize(rescale=10000)
        # rồi log10(x+1) -- tức CP10K. PixNet dùng np.log1p(raw_count): lấy thẳng raw count,
        # KHÔNG chia cho tổng count/spot (không chuẩn hóa library size), rồi ln(1+x). Đổi
        # đúng công thức này để khớp PixNet -- lưu ý: log1p dùng ln (base e), không phải log10
        # như bản cũ.
        self.exp_dict = {i: np.log1p(m[self.gene_set].values) for i, m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]
        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)
        x, y = center
        patch = self._get_img_cached(self.id2name[i]).crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)
        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)
    def __len__(self):
        return self.cumlen[-1]
    def get_test_labels(self):
        """Label cho toan bo spot cua tap test, theo dung thu tu section trong self.names.

        Voi LOPO, test_set co the gom NHIEU slide cua cung 1 benh nhan. Dataset test
        tra spot theo thu tu self.names (section 1) roi section 2... nen label phai
        gop theo dung thu tu do de khop do dai voi adata_pred (concat cua predict).
        Section khong co ground-truth duoc danh 'undetermined' de cluster_with_nmi
        tu dong bo qua.
        """
        parts = []
        for name in self.names:
            lab = self.label.get(name)
            if lab is None:
                lab = np.full(len(self.meta_dict[name]), 'undetermined')
            parts.append(np.asarray(lab))
        if not parts:
            return None
        return np.concatenate(parts)
    def get_img(self,name):
        return Image.open(self.get_img_path(name)).convert("RGB")

    def get_img_path(self, name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        return pre+'/'+fig_name

    def _get_img_cached(self, name):
        if name in self.img_dict:
            self.img_dict.move_to_end(name)
            return self.img_dict[name]
        image = Image.open(self.img_paths[name]).convert("RGB")
        self.img_dict[name] = image
        if len(self.img_dict) > self.img_cache_size:
            _, old_image = self.img_dict.popitem(last=False)
            old_image.close()
        return image
    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df
    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        df = pd.read_csv(path,sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        return df
    def get_lbl(self,name):
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

    def get_test_labels(self):
        """[MỚI - sửa lỗi ARI/NMI đa lát cắt] Nối nhãn ground-truth của TẤT CẢ lát cắt
        trong self.names theo đúng thứ tự, dùng 'undetermined' cho lát cắt không có
        annotation. Cùng thứ tự với adata_pred/adata_gt được nối trong stnet_predict /
        histogene_predict (torch.cat theo đúng thứ tự self.names, shuffle=False).
        """
        parts = []
        for name in self.names:
            lab = self.label.get(name)
            if lab is None:
                lab = np.full(len(self.meta_dict[name]), 'undetermined')
            parts.append(np.asarray(lab))
        if not parts:
            return None
        return np.concatenate(parts)

    
    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta
    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class LightHGGEP_HER2ST(torch.utils.data.Dataset):
    """
    Dataset cho Light-HGGEP voi input 4 kenh (RGB + Sobel Gradient)
    va xay dung ma tran ke cho Spatial SGC
    """
    def __init__(self, train=True, fold=0, k_neighbors=4):
        super(LightHGGEP_HER2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224 // 2  # patch size = 224
        self.k = k_neighbors
        
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        # Chon gene list qua hook. LightHGGEP_HER2ST_Top250 override de lay 250 gen co
        # muc bieu hien trung binh cao nhat; chay TRUOC LOOCV split de train/test instance
        # ra cung 1 bo gen (khong lech n_genes).
        self.gene_list = self._select_gene_list()
        gene_list = self.gene_list
        self.train = train
        
        # Leave-One-Patient-Out (LOPO) split
        samples = self.names # Giữ nguyên danh sách mẫu hợp lệ của bạn
        
        # Trích xuất danh sách các bệnh nhân duy nhất (ký tự đầu tiên của chuỗi, vd: 'A', 'B', 'C'...)
        patients = sorted(list(set([name[0] for name in samples])))
        
        # Lấy tên bệnh nhân cho tập Test dựa vào biến fold
        # Dùng phép chia lấy dư (%) để tránh lỗi index out of range nếu fold truyền vào lớn hơn số bệnh nhân
        test_patient = patients[fold % len(patients)]
        
        # Tách tập Test (tất cả các lát cắt của bệnh nhân test) và Train (các bệnh nhân còn lại)
        te_names = [name for name in samples if name[0] == test_patient]
        tr_names = [name for name in samples if name[0] != test_patient]
        
        print(f"LOPO Split - Bệnh nhân Test: {test_patient} | Số mẫu Test: {len(te_names)} | Số mẫu Train: {len(tr_names)}")
        
        if train:
            self.names = tr_names
        else:
            self.names = te_names
        
        print('Registering image paths for Light-HGGEP...')
        self.img_paths = {i: self.get_img_path(i) for i in self.names}
        self.img_dict = OrderedDict()
        self.img_cache_size = 1
        
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        
        # Labels for test set
        self.label = {i: None for i in self.names}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }
        
        if not train:
            # Fix cho LOPO: Duyệt qua từng slide trong tập test độc lập
            for i in self.names:
                idx = self.meta_dict[i].index
                # Chỉ đọc file label nếu slide thực sự có nhãn
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl_df = self.get_lbl(i)
                    self.label[i] = lbl_df.loc[idx, :]['label'].values
                else:
                    # Các slide không có ground-truth label (như A2, A3...) được gán 'undetermined'
                    self.label[i] = np.full(len(idx), 'undetermined')
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)
        
        self.gene_set = list(gene_list)
        # [SỬA - đồng bộ chuẩn hóa với PixNet] xem comment chi tiết ở class HER2ST phía trên
        # (cùng lý do, cùng công thức: bỏ library_size_normalize, chỉ dùng log1p trên raw count).
        self.exp_dict = {i: np.log1p(m[self.gene_set].values)
                         for i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
                            for i, m in self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        
        # Image transforms for PIL Image
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
        ])
        
        # Mean/Std cho 3 kenh
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Build K-NN graph cho tung section (cho Spatial SGC)
        self.A_norm_cache = {}
        self._build_graphs()
    
    def _select_gene_list(self):
        """Tra ve danh sach gene dung lam dau ra. Base: 785 gene tu file her_hvg_cut_1000."""
        return list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))

    def _build_graphs(self):
        """
        Xay dung ma tran ke chuan hoa A_norm cho tung section
        A_norm = D^(-1/2) * A_tilde * D^(-1/2)
        voi A_tilde = A + I (self-loops)
        """
        for section, meta in self.meta_dict.items():
            N = len(meta)
            if N < 2:
                self.A_norm_cache[section] = np.eye(1, dtype=np.float32)
                continue
            
            # Lay toa do spatial (x, y)
            coords = self.loc_dict[section]  # (N, 2)
            
            # K-NN graph
            D = pairwise_distances(coords, metric='euclidean')
            k_eff = min(self.k, N - 1) if N > 1 else 1
            A = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                order = np.argsort(D[i])
                order = order[order != i][:k_eff]
                A[i, order] = 1.0
            
            # A_tilde = A + I (self-loops)
            A_tilde = A + np.eye(N, dtype=np.float32)
            
            # D_hat^(-1/2) * A_tilde * D_hat^(-1/2)
            D_hat = np.diag(np.sum(A_tilde, axis=1) ** (-0.5))
            D_hat[np.isinf(D_hat)] = 0
            A_norm = D_hat @ A_tilde @ D_hat
            
            self.A_norm_cache[section] = A_norm.astype(np.float32)
    
    def compute_sobel_gradient_np(self, rgb_image):
        """Tinh Sobel gradient tu anh RGB (numpy)"""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
        gradient = gradient / (gradient.max() + 1e-8)
        return gradient[..., np.newaxis]
    
    def preprocess_3ch(self, patch_rgb):
        """Chuyen RGB patch thanh 3-channel (chi RGB)"""
        patch_tensor = torch.from_numpy(patch_rgb.transpose(2, 0, 1)).float() / 255.0
        for c in range(3):
            patch_tensor[c] = (patch_tensor[c] - self.mean[c]) / self.std[c]
        return patch_tensor
    
    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        name = self.id2name[i]
        exp = self.exp_dict[name][idx]
        center = self.center_dict[name][idx]
        loc = self.loc_dict[name][idx]
        
        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)
        
        x, y = center
        patch = self._get_img_cached(name).crop((x - self.r, y - self.r, x + self.r, y + self.r))
        
        # [SỬA] Áp dụng transforms khi patch còn là PIL Image
        if self.train:
            patch = self.transforms(patch)
        
        # [SỬA] Chuyển sang numpy array sau khi transforms
        patch = np.array(patch)
        patch_3ch = self.preprocess_3ch(patch)
        
        # Tra ve them section_name va local_idx cho SGC
        section_name = name
        local_idx = idx  # index trong section
        
        if self.train:
            return patch_3ch, loc, exp, section_name, local_idx
        else:
            return patch_3ch, loc, exp, torch.Tensor(center), section_name, local_idx
    
    def __len__(self):
        return self.cumlen[-1]
    
    def get_test_labels(self): 
        """Label cho toan bo spot cua tap test, theo dung thu tu section trong self.names.

        Voi LOPO, test_loader (shuffle=False, full_section=True) xuat batch theo dung
        thu tu self.names, moi section 1 batch, va lighthggep_predict concat lai theo
        thu tu do. Ham nay tra ve label cung thu tu -> khop do dai voi adata_pred, thay
        vi chi lay label cua names[0] nhu truoc (gay lech do dai khi benh nhan test co
        nhieu slide). Section khong co ground-truth duoc danh 'undetermined' de
        cluster_with_nmi tu dong bo qua.
        """
        parts = []
    
        for name in self.names:
            lab = self.label.get(name)
            if lab is None:
                lab = np.full(len(self.meta_dict[name]), 'undetermined')
            parts.append(np.asarray(lab))
        
        if not parts:
            return None
        return np.concatenate(parts)

    def get_img(self, name):
        return Image.open(self.get_img_path(name)).convert("RGB")

    def get_img_path(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        return pre + '/' + fig_name

    def _get_img_cached(self, name):
        if name in self.img_dict:
            self.img_dict.move_to_end(name)
            return self.img_dict[name]
        image = Image.open(self.img_paths[name]).convert("RGB")
        self.img_dict[name] = image
        if len(self.img_dict) > self.img_cache_size:
            _, old_image = self.img_dict.popitem(last=False)
            old_image.close()
        return image
    
    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.tsv'
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df
    
    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df
    
    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df
    
    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta


class LightHGGEP_HER2ST_Top250(LightHGGEP_HER2ST):
    """LightHGGEP_HER2ST voi dau ra 250 gen co muc bieu hien trung binh cao nhat.

    Toan bo logic (patch crop, augment, normalization, exp log-normalize, K-NN graph,
    section_name/local_idx) giong het class goc. Khac duy nhat: gene list duoc chon tu
    count matrix (TREN TOAN BO section, truoc LOOCV split) thay vi file her_hvg_cut_1000.npy
    -> train/test instance cung dung 1 bo 250 gen, khong lech n_genes.
    """
    def _select_gene_list(self):
        # 1) Lay intersection gene co mat trong TAT CA section
        common_genes = None
        for name in self.names:
            cnt = self.get_cnt(name)
            genes = set(cnt.columns)
            if common_genes is None:
                common_genes = genes
            else:
                common_genes &= genes

        # 2) Tinh mean bieu hien tung gen tren intersection -> 250 gen cao nhat
        gene_means = {}
        for name in self.names:
            cnt = self.get_cnt(name)
            for g in common_genes:
                gene_means[g] = gene_means.get(g, 0.0) + float(cnt[g].mean())
        top = sorted(gene_means.items(), key=lambda kv: kv[1], reverse=True)[:250]
        return [g for g, _ in top]


class LightHGGEP_HEST_h5py(torch.utils.data.Dataset):
    """Dataset HEST (.h5ad) cho Light-HGGEP, dạng GỘP nhiều mẫu theo kiểu
    LightHGGEP_HER2ST — dùng để TRAIN từ đầu trên HEST theo Leave-One-Patient-Out
    (LOPO): mỗi file .h5ad = 1 mẫu (patient), fold chọn 1 mẫu làm test, phần còn
    lại là train (validation = mẫu train alphabet đầu, do script train quyết định
    qua SectionBatchSampler include/exclude).

    Đọc .h5ad trực tiếp bằng h5py + scipy.sparse (KHÔNG scanpy). Gene target giữ
    đủ (gene_list, 785); gene vắng mặt trong mẫu được điền 0 để shape (N,785)
    khớp model. Khi tính điểm, script phải loại cột toàn-0 (xem run_hest_train.py).

    __getitem__: train trả tuple 5, test trả tuple 6 — khớp section_collate_fn:
        train: (patch_3ch, loc, exp, section_name, local_idx)
        test : (patch_3ch, loc, exp, center, section_name, local_idx)
    """
    def __init__(self, h5_dir, gene_list, train=True, fold=0, k_neighbors=4,
                 patch_size=224):
        super(LightHGGEP_HEST_h5py, self).__init__()
        self.h5_dir = h5_dir
        self.k = k_neighbors
        self.patch_size = patch_size
        self.target_genes = list(gene_list)
        self.gene_set = list(gene_list)
        self.train = train

        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # Danh sách mẫu (tên file không đuôi) — mỗi mẫu = 1 patient
        paths = sorted(glob.glob(os.path.join(h5_dir, '*.h5ad')))
        if not paths:
            raise FileNotFoundError(f"Không tìm thấy *.h5ad trong {h5_dir}")
        all_samples = sorted(os.path.basename(p).split('.')[0] for p in paths)

        # Leave-One-Patient-Out: fold -> mẫu test
        test_sample = all_samples[fold % len(all_samples)]
        te_names = [s for s in all_samples if s == test_sample]
        tr_names = [s for s in all_samples if s != test_sample]

        print(f"HEST LOPO Split - Test: {test_sample} | Train: {len(tr_names)} | "
              f"Test: {len(te_names)} | tổng {len(all_samples)} mẫu")
        self.names = te_names if not train else tr_names

        # Load metadata từng mẫu (giữ import h5py/sp ở module-level)
        self.meta_dict = {n: self._load_sample(n) for n in self.names}

        self.exp_dict = {n: m['exp'] for n, m in self.meta_dict.items()}
        self.center_dict = {n: m['center'] for n, m in self.meta_dict.items()}
        self.loc_dict = {n: m['loc'] for n, m in self.meta_dict.items()}
        self.img_dict = {n: m['img'] for n, m in self.meta_dict.items()}
        self.r_img = next(iter(self.meta_dict.values()))['r_img']

        self.lengths = [len(m['loc']) for m in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        # Augment khi train (áp lên patch đã crop, kiểu HER2ST)
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
        ])

        # Build K-NN graph riêng từng mẫu
        self.A_norm_cache = {}
        self._build_graphs()

    def _load_sample(self, name):
        """Đọc 1 file .h5ad -> dict: exp, loc, center, img, r_img."""
        path = os.path.join(self.h5_dir, name + '.h5ad')
        import json
        with h5py.File(path, 'r') as f:
            var_group = f['var']
            index_col = var_group.attrs.get('_index', '_index')
            raw_genes = var_group[index_col][:]
            var_names = np.array([g.decode('utf-8') if isinstance(g, bytes) else g
                                  for g in raw_genes])

            X_node = f['X']
            shape = tuple(X_node.attrs.get('shape')) if 'shape' in X_node.attrs else None
            if isinstance(X_node, h5py.Group) and 'data' in X_node:
                data = X_node['data'][:]
                indices = X_node['indices'][:]
                indptr = X_node['indptr'][:]
                X_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                X_matrix = X_node[:]
                if sp.issparse(X_matrix):
                    X_matrix = X_matrix.tocsr()
            N = X_matrix.shape[0]

            # exp: giữ đủ gene_list, gene thiếu -> cột 0
            gene2col = {g: i for i, g in enumerate(var_names)}
            spot_sums = np.asarray(X_matrix.sum(axis=1)).flatten()
            spot_sums[spot_sums == 0] = 1.0
            exp = np.zeros((N, len(self.target_genes)), dtype=np.float32)
            common = 0
            for j, g in enumerate(self.target_genes):
                col = gene2col.get(g)
                if col is None:
                    continue
                col_vec = X_matrix[:, col].toarray().flatten()
                exp[:, j] = np.log1p((col_vec / spot_sums) * 1e6)
                common += 1
            print(f"[HEST] {name}: {common}/{len(self.target_genes)} gene có mặt; N={N}")

            coords = np.asarray(f['obsm']['spatial'][:], dtype=np.float64)

            spatial_grp = f['uns']['spatial']
            internal_id = list(spatial_grp.keys())[0]
            sf_grp = spatial_grp[internal_id]['scalefactors']
            if isinstance(sf_grp, h5py.Group):
                scale = sf_grp['tissue_downscaled_fullres_scalef'][()]
                spot_dia = sf_grp['spot_diameter_fullres'][()]
            else:
                raw = sf_grp[()]
                sf_json = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                scale = sf_json['tissue_downscaled_fullres_scalef']
                spot_dia = sf_json['spot_diameter_fullres']
            scale = float(scale)

            imgs_grp = spatial_grp[internal_id]['images']
            img = imgs_grp['downscaled_fullres'][:] if 'downscaled_fullres' in imgs_grp \
                else imgs_grp['hires'][:]
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            if img.max() <= 1.0:
                img = (img * 255.0).astype(np.uint8)
            img = np.ascontiguousarray(img)

            r_img = int(round(float(spot_dia) * scale * 0.75))
            if r_img < 1:
                r_img = 1

        return {
            'exp': exp,
            'loc': coords,
            'center': np.floor(coords * scale).astype(int),
            'img': img,
            'r_img': r_img,
        }

    def _build_graphs(self):
        for name, meta in self.meta_dict.items():
            coords = meta['loc']
            N = len(coords)
            if N < 2:
                self.A_norm_cache[name] = np.eye(1, dtype=np.float32)
                continue
            D = pairwise_distances(coords, metric='euclidean')
            k_eff = min(self.k, N - 1)
            A = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                order = np.argsort(D[i])
                order = order[order != i][:k_eff]
                A[i, order] = 1.0
            A_tilde = A + np.eye(N, dtype=np.float32)
            D_hat = np.diag(np.sum(A_tilde, axis=1) ** (-0.5))
            D_hat[np.isinf(D_hat)] = 0
            self.A_norm_cache[name] = (D_hat @ A_tilde @ D_hat).astype(np.float32)

    def __len__(self):
        return self.cumlen[-1]

    def _crop_patch(self, name, idx):
        meta = self.meta_dict[name]
        r = meta['r_img']
        x, y = meta['center'][idx]
        h, w, _ = meta['img'].shape
        top, bottom = y - r, y + r
        left, right = x - r, x + r
        crop = meta['img'][max(top, 0):min(bottom, h), max(left, 0):min(right, w)]
        pad_top = max(-top, 0); pad_bottom = max(bottom - h, 0)
        pad_left = max(-left, 0); pad_right = max(right - w, 0)
        if pad_top or pad_bottom or pad_left or pad_right:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_REPLICATE)
        crop = crop[:2 * r, :2 * r]
        if self.train:
            # augment trên ảnh uint8 trước khi chuẩn hóa (dùng torchvision trên tensor)
            from PIL import Image
            crop = np.array(self.transforms(Image.fromarray(crop)))
        resized = cv2.resize(crop, (self.patch_size, self.patch_size),
                             interpolation=cv2.INTER_CUBIC)
        t = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        for c in range(3):
            t[c] = (t[c] - self.mean[c][0][0]) / self.std[c][0][0]
        return torch.from_numpy(t).float()

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        name = self.id2name[i]
        pat = self._crop_patch(name, idx)
        exp = self.exp_dict[name][idx]
        loc = self.loc_dict[name][idx]
        center = self.center_dict[name][idx]

        exp_t = torch.tensor(exp, dtype=torch.float32)
        loc_t = torch.tensor(loc, dtype=torch.float32)
        center_t = torch.tensor(center, dtype=torch.float32)
        if self.train:
            return pat, loc_t, exp_t, name, idx
        else:
            return pat, loc_t, exp_t, center_t, name, idx
