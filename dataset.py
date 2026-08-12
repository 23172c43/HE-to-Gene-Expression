import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
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
        samples = self.names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))
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
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            self.label[self.names[0]]=lbl
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
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
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
        
        # LOOCV split (giống ViT_HER2ST)
        samples = self.names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))
        
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
        
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
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
        self.exp_dict = {i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
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


class LightHGGEP_BRAINST(LightHGGEP_HER2ST):
    """
    LightHGGEP dataset tren V1_Adult_Mouse_Brain (scanpy visium_sge).

    Giong het LightHGGEP_HER2ST ve pipeline (patch crop 224, ImageNet norm,
    exp log-normalize, K-NN graph, section_name/local_idx). Khac:
      - Anh H&E lay tu adata.uns['spatial'][sid]['images']['hires'] (ndarray,
        full-res pixel) thay vi doc tu file .tif WSI.
      - Toa do spot la adata.obsm['spatial'] (full-res pixel) thay vi
        ['pixel_x','pixel_y'] cua HER2ST.
      - Chi CO 1 section (mouse brain) nen LOOCV vo nghia -> khong split fold,
        train/val cat theo slide duy nhat (VAL_SECTION = chinh section do).
      - Gene list (Top250) tinh tren count matrix giong LightHGGEP_HER2ST_Top250.
    """
    def __init__(self, train=True, fold=0, k_neighbors=4, sample_id='V1_Adult_Mouse_Brain'):
        # KHONG goi super().__init__ vi no doc file HER2ST; setup tu tay nhung
        # tai dung cac thuoc tinh chung (r, k, transforms, mean/std, gene_list...).
        self.sample_id = sample_id
        self.r = 224 // 2
        self.k = k_neighbors
        self.train = train
        self.gene_list = self._select_gene_list()
        gene_list = self.gene_list

        print('Loading V1_Adult_Mouse_Brain via scanpy ...')
        self._brain_adata = None
        meta = self.get_meta(sample_id)   # tai 1 lan, cache
        self.names = [sample_id]
        self.id2name = {0: sample_id}
        self.meta_dict = {sample_id: meta}
        self.lengths = [len(meta)]
        self.cumlen = np.cumsum(self.lengths)

        self.label = {sample_id: None}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1,
        }
        # BRAINST khong co label -> tat ca -1
        self.label[sample_id] = torch.full((len(meta),), -1)

        cnt = self.get_cnt(sample_id)
        self.gene_set = list(gene_list)
        exp_mat = scp.transform.log(
            scp.normalize.library_size_normalize(cnt[self.gene_set].values))
        self.exp_dict = {sample_id: exp_mat}

        # center: full-res pixel tu obsm['spatial']; loc: chinh center (dung lam input)
        coords = np.asarray(meta.obsm['spatial'], dtype=np.int64)
        self.center_dict = {sample_id: coords}
        self.loc_dict = {sample_id: coords.astype(float)}

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
        ])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.A_norm_cache = {}
        self._build_graphs()

    def _select_gene_list(self):
        # Top 250 gen co mean bieu hien cao nhat tren count matrix (1 section)
        cnt = self.get_cnt(self.sample_id)
        gene_means = cnt.mean(axis=0)
        top = gene_means.sort_values(ascending=False).index[:250].tolist()
        # dam bao dung 250, khong trung lap
        seen = []
        for g in top:
            if g not in seen:
                seen.append(g)
            if len(seen) == 250:
                break
        return seen

    def get_img_path(self, name):
        # BRAINST khong doc file, anh da nam trong self.hires_img
        return None

    def _load_hires(self):
        if not hasattr(self, 'hires_img') or self.hires_img is None:
            adata = self.get_meta(self.sample_id)
            sid = self.sample_id
            self.hires_img = np.asarray(
                adata.uns['spatial'][sid]['images']['hires'], dtype=np.float32)
            self.hires_scale = adata.uns['spatial'][sid]['scalefactors']['tissue_hires_scalef']

    def _get_img_array(self, name):
        self._load_hires()
        return self.hires_img

    def get_cnt(self, name):
        # name bi voi sample_id; tra ve count matrix tu adata
        adata = self.get_meta(self.sample_id)
        cols = list(adata.var_names)
        # 10x Visium co the co ten gene trung lap -> drop de cnt[gene_set] khong
        # mo rong them cot (lam lech n_genes). Giu lan xuat hien dau tien.
        seen = set(); keep = []
        for j, c in enumerate(cols):
            if c not in seen:
                seen.add(c); keep.append(j)
        mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        mat = mat[:, keep]
        uniq = [cols[j] for j in keep]
        return pd.DataFrame(mat, index=adata.obs.index, columns=uniq)

    def get_pos(self, name):
        # tra ve DataFrame voi cot x,y la full-res pixel coordinate
        adata = self.get_meta(self.sample_id)
        coords = np.asarray(adata.obsm['spatial'], dtype=np.int64)
        df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]})
        df['id'] = [str(coords[i, 0]) + 'x' + str(coords[i, 1]) for i in range(len(coords))]
        return df

    def get_lbl(self, name):
        # BRAINST khong co label; tra ve DataFrame rong de ke thua khong crash
        df = self.get_pos(name).copy()
        df['label'] = 'undetermined'
        df.set_index('id', inplace=True)
        return df

    def get_meta(self, name, gene_list=None):
        # Cache 1 lan
        if not hasattr(self, '_brain_adata') or self._brain_adata is None:
            adata = sc.datasets.visium_sge(sample_id=self.sample_id,
                                           include_hires_tiff=False)
            self._brain_adata = adata
        return self._brain_adata

    def _get_img_cached(self, name):
        # BRAINST luu anh numpy thay vi PIL; crop truc tiep
        self._load_hires()
        return self.hires_img

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i - 1]

        name = self.id2name[i]
        exp = self.exp_dict[name][idx]
        # BRAINST: center la full-res pixel tu obsm['spatial']
        center = self.center_dict[name][idx]
        loc = self.loc_dict[name][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        # Anh numpy (H, W, 3) float32; crop 224 quanh (x, y)
        img = self._get_img_cached(name)
        H, W, _ = img.shape
        r = self.r
        x0, y0 = int(round(x * self.hires_scale)), int(round(y * self.hires_scale))
        x0 = max(0, min(x0, W)) - 0  # clamp ben duoi
        # crop an toan
        x0c = max(0, x0 - r); x1c = min(W, x0 + r)
        y0c = max(0, y0 - r); y1c = min(H, y0 + r)
        patch = img[y0c:y1c, x0c:x1c, :]
        # pad neu bien
        ph = 2 * r - (y1c - y0c); pw = 2 * r - (x1c - x0c)
        if ph > 0 or pw > 0:
            patch = np.pad(patch, ((max(0, r - y0c), max(0, (y0c + (y1c - y0c) + r) - H)),
                                   (max(0, r - x0c), max(0, (x0c + (x1c - x0c) + r) - W)),
                                   (0, 0)), mode='constant')
        patch = np.asarray(patch, dtype=np.float32) / 255.0
        patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        for c in range(3):
            patch[c] = (patch[c] - self.mean[c]) / self.std[c]

        if self.train:
            patch = self.transforms(patch)

        section_name = name
        local_idx = idx

        if self.train:
            return patch, loc, exp, section_name, local_idx
        else:
            return patch, loc, exp, torch.Tensor(center), section_name, local_idx


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


# ============================================================================
# BRAIN-ST baseline adapters (cho STNet / HisToGene trong run_baselines.py)
# ============================================================================
class BRAINSTSpotDataset(Dataset):
    """Spot-level adapter cho STNet. Tra ve (patch, center, exp).

    Lay patch 224x224 tu LightHGGEP_BRAINST.__getitem__ (bo phan tu
    section_name/local_idx), tra ve (patch, center, exp) giong HER2ST.__getitem__.
    """
    def __init__(self, train=True, k_neighbors=4, seed=42):
        self.brain = LightHGGEP_BRAINST(train=train, fold=0, k_neighbors=k_neighbors)
        self.train = train
        self.rng = random.Random(seed)
        self.indices = list(range(len(self.brain)))
        if self.train:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        bi = self.indices[index]
        item = self.brain[bi]
        # LightHGGEP_BRAINST.__getitem__ tra ve 5 (train) / 6 (test) items:
        # (patch, loc, exp, [center,] section_name, local_idx)
        patch = item[0]
        exp = item[2]
        center = item[3] if not self.train else item[1]  # test co center o idx3; train center=loc
        if self.train:
            center = item[1]
        else:
            center = item[3]
        return patch, center, exp


class BRAINSTSlideDataset(Dataset):
    """Slide-level adapter cho HisToGene. Tra ve (patches_flat, centers_clamped, exp).

    Giong HisToGeneSlideDataset: gom toan bo spot cua 1 section thanh 1 item,
    crop 224->112 o trung tam, flatten patch, clamp center ve [0,63].
    """
    def __init__(self, spot_dataset, indices=None):
        self.spot_dataset = spot_dataset
        # Section chi co 1, lay toan bo indices
        self.indices = indices if indices is not None else list(range(len(spot_dataset)))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        patches, locations, expressions = [], [], []
        for si in self.indices:
            item = self.spot_dataset[si]
            patch = item[0]
            loc = item[1]
            exp = item[2]
            h, w = patch.shape[-2:]
            top, left = (h - 112) // 2, (w - 112) // 2
            patch = patch[:, top:top + 112, left:left + 112]
            patches.append(patch.flatten())
            locations.append(loc.long().clamp(0, 63))
            expressions.append(exp)
        return (torch.stack(patches), torch.stack(locations), torch.stack(expressions))
