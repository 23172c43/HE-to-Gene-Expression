import torch
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import anndata as ann

MODEL_PATH = ''

def lighthggep_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho Light-HGGEP voi Spatial SGC
    Dataset tra ve: patch_3ch, positions, exp, centers, section_name, local_indices
    """
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Dataset tra ve 6 items cho test
            patch_3ch, positions, exp, centers, section_name, local_indices = batch
            patch_3ch, positions = patch_3ch.to(device), positions.to(device)
            local_indices = local_indices.to(device)
            
            pred = model(patch_3ch, positions, section_name, local_indices)
            
            if preds is None:
                preds = pred
                ct = centers
                gt = exp
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, centers), dim=0)
                gt = torch.cat((gt, exp), dim=0)
    
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct
    
    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    
    return adata, adata_gt

def _per_section_average(v1, v2, section_ids, stat_func):
    """[MỚI - đồng bộ với per-section Moran's I] Tính stat_func(v1_sub, v2_sub) RIÊNG
    cho từng lát cắt trong section_ids, rồi lấy TRUNG BÌNH các lát hợp lệ.

    Lý do cần hàm này: khi 1 fold test gồm nhiều lát cắt của cùng 1 bệnh nhân (LOPO),
    gộp toàn bộ spot rồi tính 1 lần (pooled) cho ra kết quả LỆCH so với tính riêng từng
    lát rồi trung bình -- vì 2 nguyên nhân:
      (1) Với MSE/MAE: pooled = trung bình CÓ TRỌNG SỐ theo số spot mỗi lát (lát nhiều
          spot lấn át lát ít spot), còn per-section-average coi mỗi lát nặng NGANG NHAU.
      (2) Với Pearson/Spearman: pooled bị ảnh hưởng bởi CHÊNH LỆCH BASELINE biểu hiện
          gen giữa các lát cắt (khác biệt nhuộm màu, độ dày mô...) -- kiểu Nghịch lý
          Simpson: tương quan gộp có thể CAO hoặc THẤP hơn hẳn tương quan thật sự bên
          trong từng lát, dù model không hề dự đoán tốt/tệ hơn thực chất.

    Tham số:
        v1, v2      : mảng 1-D cùng độ dài N (giá trị của 1 gene, N spot)
        section_ids : mảng N phần tử, section_name của từng spot (đúng thứ tự với v1, v2)
        stat_func   : hàm nhận (v1_sub, v2_sub) -> trả về 1 số float (KHÔNG phải tuple).
                      Với pearsonr/spearmanr (trả tuple (r, p)), phải bọc lại kiểu
                      `lambda a, b: pearsonr(a, b)[0]` trước khi truyền vào đây.
    Trả về: (gia_tri_trung_binh, so_lat_hop_le)
    """
    vals = []
    for name in np.unique(section_ids):
        mask = section_ids == name
        if mask.sum() < 2:          # can it nhat 2 spot moi tinh duoc tuong quan/loi
            continue
        v1_sub, v2_sub = v1[mask], v2[mask]
        try:
            val = stat_func(v1_sub, v2_sub)
        except Exception:
            val = float('nan')
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            vals.append(val)
    if len(vals) == 0:
        return float('nan'), 0
    return float(np.mean(vals)), len(vals)


def get_R(data1,data2,dim=1,func=pearsonr,section_ids=None):
    """
    section_ids: [MỚI] None (mặc định) -> hành vi CŨ, gộp toàn bộ N spot rồi tính 1 lần
        cho mỗi gene (pooled). Có giá trị -> tính riêng từng lát cắt rồi trung bình
        (per-section average), tránh bị thổi phồng/lệch bởi chênh lệch baseline giữa
        các lát hoặc số spot không đều nhau khi 1 fold gồm nhiều lát cắt (LOPO).
        p-value vẫn luôn tính theo kiểu pooled (chỉ mang tính tham khảo).
    """
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            col1, col2 = adata1[:,g], adata2[:,g]
        elif dim==0:
            col1, col2 = adata1[g,:], adata2[g,:]
        _, pv = func(col1, col2)   # p-value: giữ pooled, chỉ để tham khảo
        if section_ids is None:
            r, _ = func(col1, col2)
        else:
            r, _n_valid = _per_section_average(col1, col2, section_ids,
                                                lambda a, b: func(a, b)[0])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

def _full_pca_tsne(tmp):
    """Thay sc.pp.pca + sc.tl.tsne bằng sklearn (ko phụ thuộc scanpy).
    Gán adata.obsm['X_pca'] = PCA, rồi t-SNE. Tái dùng cho cluster() và
    cluster_with_nmi()."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = np.asarray(tmp.X, dtype=np.float64)
    pca = PCA(n_components=min(50, X.shape[0], X.shape[1]), random_state=0).fit_transform(X)
    tsne = TSNE(n_components=2, random_state=0).fit_transform(pca)
    tmp.obsm['X_pca'] = tsne.astype(np.float32)


def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    _full_pca_tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)

def get_MSE(data1, data2, dim=1, section_ids=None):
    """
    section_ids: [MỚI] None (mặc định) -> hành vi CŨ, gộp toàn bộ N spot rồi tính 1 lần
        (= trung bình CÓ TRỌNG SỐ theo số spot mỗi lát). Có giá trị -> tính MSE riêng
        từng lát cắt rồi trung bình KHÔNG trọng số (mỗi lát nặng ngang nhau, tránh lát
        đông spot lấn át lát ít spot khi 1 fold gồm nhiều lát cắt -- LOPO).
    """
    adata1 = data1.X
    adata2 = data2.X
    mse_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            col1, col2 = adata1[:, g], adata2[:, g]
        elif dim == 0:
            col1, col2 = adata1[g, :], adata2[g, :]
        if section_ids is None:
            mse = mean_squared_error(col1, col2)
        else:
            mse, _n_valid = _per_section_average(col1, col2, section_ids,
                                                  mean_squared_error)
        mse_list.append(mse)
    return np.array(mse_list)

def get_MAE(data1, data2, dim=1, section_ids=None):
    """section_ids: [MỚI] xem docstring get_MSE() -- cùng lý do, cùng cơ chế."""
    adata1 = data1.X
    adata2 = data2.X
    mae_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            col1, col2 = adata1[:, g], adata2[:, g]
        elif dim == 0:
            col1, col2 = adata1[g, :], adata2[g, :]
        if section_ids is None:
            mae = mean_absolute_error(col1, col2)
        else:
            mae, _n_valid = _per_section_average(col1, col2, section_ids,
                                                  mean_absolute_error)
        mae_list.append(mae)
    return np.array(mae_list)


def get_Spearman(data1, data2, dim=1, section_ids=None):
    """
    Tính gene-wise Spearman Correlation Coefficient.
    Cùng convention với get_R: dim=1 → lặp theo gene (cột),
    trả về (rho_array, pvalue_array) shape (n_genes,).
    Spearman bổ sung cho PCC vì không giả định phân phối tuyến tính --
    bắt được cả monotonic relationship giữa pred và gt.

    section_ids: [MỚI] xem docstring get_R() -- cùng lý do, cùng cơ chế.
    """
    adata1 = data1.X
    adata2 = data2.X
    rho_list, p_list = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            col1, col2 = adata1[:, g], adata2[:, g]
        elif dim == 0:
            col1, col2 = adata1[g, :], adata2[g, :]
        _, pv = spearmanr(col1, col2)   # p-value: giữ pooled, chỉ để tham khảo
        if section_ids is None:
            rho, _ = spearmanr(col1, col2)
        else:
            rho, _n_valid = _per_section_average(col1, col2, section_ids,
                                                  lambda a, b: spearmanr(a, b)[0])
        rho_list.append(rho)
        p_list.append(pv)
    return np.array(rho_list), np.array(p_list)


def get_section_ids(dataset):
    """[MỚI - sửa lỗi Moran's I đa lát cắt]
    Trả về mảng section_name (str) cho từng spot, ĐÚNG THỨ TỰ global index
    0..N-1 mà dataset.__getitem__ trả về (tức đúng thứ tự mà stnet_predict /
    histogene_predict / lighthggep_predict nối các batch lại bằng torch.cat),
    miễn là DataLoader/SectionBatchSampler chạy với shuffle=False (luôn đúng
    cho test_loader trong run_pipeline.py và run_baselines.py).

    Dùng dataset.lengths và dataset.id2name (đã có sẵn trong cả HER2ST và
    LightHGGEP_HER2ST, xem dataset.py) -- không cần sửa gì bên trong 3 hàm
    predict, chỉ cần build mảng này 1 lần ở nơi gọi evaluate_her2st_predictions
    rồi truyền vào.

    Ví dụ: section A có 300 spot, section B có 250 spot
        -> trả về ['A']*300 + ['B']*250  (list[str] length 550)
    """
    ids = []
    for name, length in zip(dataset.id2name.values(), dataset.lengths):
        ids.extend([name] * length)
    return np.array(ids)


def get_MoransI(adata, gene_idx, spatial_key='spatial', section_ids=None):
    """
    Tính Moran's I cho 1 gene trên toàn bộ spot, dùng tọa độ spatial làm
    weight matrix (inverse distance, cắt tại K=6 láng giềng gần nhất).

    Moran's I đo mức độ auto-correlation không gian của biểu hiện gene:
      I ≈ +1 → biểu hiện phân bố thành cụm không gian (spatially clustered)
      I ≈  0 → ngẫu nhiên
      I ≈ -1 → phân tán đều

    Trả về scalar I ∈ [-1, 1].

    Tham số:
        adata      : AnnData với adata.X shape (N, G) và adata.obsm[spatial_key] shape (N, 2)
        gene_idx   : chỉ số gene (int) hoặc tên gene (str) trong adata.var_names
        spatial_key: key trong obsm chứa tọa độ (x, y)
        section_ids: [MỚI] None hoặc array/list[str] độ dài N, section_name của từng spot
                     (lấy từ get_section_ids(dataset), CÙNG THỨ TỰ với adata).
                     - None (mặc định): giữ nguyên hành vi CŨ -- build 1 đồ thị K-NN trên
                       TOÀN BỘ N spot, ĐÚNG khi adata chỉ chứa 1 lát cắt (fold kiểu
                       Leave-One-Slide-Out cũ). SAI khi adata gộp nhiều lát cắt (LOPO
                       hiện tại), vì sẽ nối "láng giềng" giữa các spot ở 2 lát mô KHÁC
                       NHAU chỉ vì toạ độ pixel của chúng tình cờ gần nhau -- các lát HER2ST
                       đều có toạ độ lưới bắt đầu từ ~0 nên rất dễ trùng khoảng giá trị.
                     - Có giá trị: build đồ thị K-NN RIÊNG cho từng section (chỉ nối láng
                       giềng trong cùng 1 lát cắt), tính Moran's I riêng từng section, rồi
                       lấy TRUNG BÌNH các section hợp lệ (bỏ qua section có N<2 spot).
                       Đây là cách bắt buộc phải dùng khi test set của 1 fold có nhiều
                       lát cắt (patient-level split / LOPO).
    """
    from sklearn.metrics.pairwise import pairwise_distances

    def _morans_i_single_block(coords, x):
        """Moran's I trên 1 khối toạ độ liền mạch (1 lát cắt, hoặc toàn bộ nếu không
        chia section). Giữ nguyên công thức gốc, chỉ tách ra để tái dùng cho cả 2 nhánh."""
        N = len(x)
        if N < 3:          # cần it nhat vai spot moi co y nghia thong ke
            return float('nan')
        x_dev = x - x.mean()

        D = pairwise_distances(coords, metric='euclidean')
        K = min(6, N - 1)
        W = np.zeros((N, N), dtype=float)
        for i in range(N):
            order = np.argsort(D[i])
            neighbors = order[order != i][:K]
            for j in neighbors:
                W[i, j] = 1.0 / (D[i, j] + 1e-8)

        W_sum = W.sum()
        if W_sum == 0:
            return float('nan')
        denominator = W_sum * np.sum(x_dev ** 2)
        if denominator == 0:
            return float('nan')
        numerator = N * np.sum(W * np.outer(x_dev, x_dev))
        return numerator / denominator

    coords_all = adata.obsm[spatial_key].astype(float)   # (N, 2)
    if isinstance(gene_idx, str):
        gene_idx = list(adata.var_names).index(gene_idx)
    x_all = adata.X[:, gene_idx].astype(float)           # (N,)

    if section_ids is None:
        # Hành vi CŨ -- CHỈ dùng khi chắc chắn adata là 1 lát cắt duy nhất.
        return _morans_i_single_block(coords_all, x_all)

    section_ids = np.asarray(section_ids)
    if len(section_ids) != len(x_all):
        raise ValueError(
            f"get_MoransI: len(section_ids)={len(section_ids)} != N spot trong adata="
            f"{len(x_all)}. Kiem tra lai get_section_ids(dataset) co dung dataset/thu tu "
            f"voi adata dang truyen vao khong.")

    # [MỚI] Tính riêng từng section, không cho láng giềng xuyên lát cắt.
    per_section_values = []
    for name in np.unique(section_ids):
        mask = section_ids == name
        val = _morans_i_single_block(coords_all[mask], x_all[mask])
        if not np.isnan(val):
            per_section_values.append(val)

    if len(per_section_values) == 0:
        return float('nan')
    return float(np.mean(per_section_values))


def get_MoransI_all(data_pred, data_gt, top_k=50, spatial_key='spatial', section_ids=None):
    """
    Tính Moran's I cho cả pred lẫn gt trên top_k gene có variance cao nhất
    (tính trên gt để chọn gene thú vị về mặt sinh học).

    section_ids: [MỚI] xem docstring get_MoransI(). Bắt buộc truyền khi data_pred/data_gt
        gộp nhiều lát cắt (LOPO); None nếu chỉ 1 lát cắt (hành vi cũ).

    Trả về dict:
        {
          'pred': np.array shape (top_k,),  -- Moran's I của từng gene trên pred
          'gt':   np.array shape (top_k,),  -- Moran's I của từng gene trên gt
          'gene_indices': np.array shape (top_k,)
        }
    """
    gt_X = data_gt.X
    var_per_gene = np.var(gt_X, axis=0)                    # variance theo từng gene
    top_indices  = np.argsort(var_per_gene)[::-1][:top_k]  # top_k gene variance cao nhất

    mi_pred, mi_gt = [], []
    for idx in top_indices:
        mi_pred.append(get_MoransI(data_pred, idx, spatial_key, section_ids=section_ids))
        mi_gt.append(get_MoransI(data_gt,   idx, spatial_key, section_ids=section_ids))

    return {
        'pred':         np.array(mi_pred),
        'gt':           np.array(mi_gt),
        'gene_indices': top_indices,
    }


def cluster_with_nmi(adata, label):
    """
    Mở rộng cluster(): tính thêm NMI bên cạnh ARI.
    Trả về (cluster_labels, ARI, NMI).

    NMI bổ sung cho ARI vì:
      - ARI hiệu chỉnh theo chance, nhạy với số cluster và size imbalance.
      - NMI đo mức độ chia sẻ thông tin giữa 2 phân hoạch, ít bị ảnh hưởng
        bởi số cluster hơn.
    """
    label = np.asarray(label)
    # HER2ST uses integer IDs and marks undetermined spots with -1.  Retain
    # support for the original string labels used by older datasets.
    unknown = -1 if np.issubdtype(label.dtype, np.number) else 'undetermined'
    idx = label != unknown
    tmp = adata[idx].copy()
    l   = label[idx]
    if len(l) < 2 or len(np.unique(l)) < 2:
        return np.array([], dtype=str), float('nan'), float('nan')
    _full_pca_tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p = kmeans.labels_.astype(str)

    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    adata.obs['kmeans'] = lbl

    ari = round(ari_score(p, l), 4)
    nmi = round(nmi_score(p, l, average_method='arithmetic'), 4)
    return p, ari, nmi


def stnet_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho STModel dùng HER2ST dataset.
    HER2ST test trả về: (patch, loc, exp, center)
      - patch  : (B, 3, 224, 224)
      - loc    : (B, 2)  -- tọa độ grid (x, y)
      - exp    : (B, n_genes)
      - center : (B, 2)  -- tọa độ pixel
    STModel.forward(patch, center) → pred (B, n_genes)
    """
    model.eval()
    model = model.to(device)
    preds, gts, centers = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="STNet Predicting"):
            patch, loc, exp, center = batch
            patch  = patch.to(device)
            center = center.to(device)
            pred   = model(patch, center)
            preds.append(pred.cpu())
            gts.append(exp)
            centers.append(center.cpu())

    preds   = torch.cat(preds,   dim=0).numpy()
    gts     = torch.cat(gts,     dim=0).numpy()
    centers = torch.cat(centers, dim=0).numpy()

    adata_pred = ann.AnnData(preds)
    adata_pred.obsm['spatial'] = centers

    adata_gt = ann.AnnData(gts)
    adata_gt.obsm['spatial'] = centers

    return adata_pred, adata_gt


def histogene_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho HisToGene dùng HER2ST dataset.

    HisToGene.forward(patches, centers) kỳ vọng input slide-level:
      patches : (1, N_spots, patch_dim)   -- flatten từng patch
      centers : (1, N_spots, 2)           -- tọa độ grid đã discretize

    HER2ST test trả về patch-level: (patch, loc, exp, center)
      patch  : (B, 3, H, W)
      loc    : (B, 2)   -- tọa độ grid float
      exp    : (B, n_genes)
      center : (B, 2)   -- tọa độ pixel

    Chiến lược: gom toàn bộ spots của test section vào 1 batch slide,
    flatten patch và discretize loc sang index để dùng Embedding.
    Vì test chỉ có 1 section (LOOCV), load hết rồi forward 1 lần.
    """
    model.eval()
    model = model.to(device)

    all_patches, all_locs, all_exps, all_centers = [], [], [], []
    for batch in test_loader:
        patch, loc, exp, center = batch
        all_patches.append(patch)
        all_locs.append(loc)
        all_exps.append(exp)
        all_centers.append(center)

    # Gom thành 1 tensor
    patches = torch.cat(all_patches, dim=0)   # (N, 3, H, W)
    locs    = torch.cat(all_locs,    dim=0)   # (N, 2)
    exps    = torch.cat(all_exps,    dim=0)   # (N, n_genes)
    centers = torch.cat(all_centers, dim=0)   # (N, 2)

    # The model was trained with centred 112 px crops (patch_dim=3*112*112).
    # Keep inference identical even though HER2ST stores 224 px patches.
    if patches.shape[-2:] != (112, 112):
        h, w = patches.shape[-2:]
        top, left = (h - 112) // 2, (w - 112) // 2
        patches = patches[:, :, top:top + 112, left:left + 112]

    # Flatten patch: (N, 3*112*112) → thêm batch dim → (1, N, patch_dim)
    N = patches.shape[0]
    # Centre-cropping can create a non-contiguous tensor; reshape preserves
    # values while safely flattening it for the patch embedding.
    patch_flat = patches.reshape(N, -1).unsqueeze(0).to(device)  # (1, N, patch_dim)

    # Discretize tọa độ grid sang long index cho Embedding
    # HisToGene dùng n_pos=64 → clamp về [0, 63]
    locs_long = locs.long().clamp(0, 63).unsqueeze(0).to(device)  # (1, N, 2)

    with torch.no_grad():
        pred = model(patch_flat, locs_long)   # (1, N, n_genes)
    pred = pred.squeeze(0).cpu().numpy()      # (N, n_genes)

    centers_np = centers.numpy()
    exps_np    = exps.numpy()

    adata_pred = ann.AnnData(pred)
    adata_pred.obsm['spatial'] = centers_np

    adata_gt = ann.AnnData(exps_np)
    adata_gt.obsm['spatial'] = centers_np

    return adata_pred, adata_gt
