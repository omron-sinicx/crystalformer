import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def onehot(x, n_classes):
    return torch.eye(n_classes)[x]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
                
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


# for analysis embeddings
def search_kNN(embedding_query, embedding_target, use_gpu=True, k=10):
    import faiss
    """
    embeddingはtensorなどarray-like、kは探す近傍点の個数（1以上のint）
    return:
        D: クエリベクトルからk近傍までの距離
        I: クエリベクトルに対するk近傍のインデックス
    """
    vec_dim = embedding_target.shape[1]                   # ベクトルの次元(dimension)
    n_data = embedding_target.shape[0]                    # データベースのサイズ(database size)
    x_target_vec = embedding_target.numpy().astype('float32')
    x_query_vec = embedding_query.numpy().astype('float32')
    faiss_index_cpu = faiss.IndexFlatL2(vec_dim)
    if use_gpu:
        faiss_index_gpu = faiss.index_cpu_to_all_gpus(faiss_index_cpu)
        faiss_index_gpu.add(x_target_vec)
        D, I = faiss_index_gpu.search(x_query_vec, k)
    else:
        faiss_index_cpu.add(x_target_vec)
        D, I = faiss_index_cpu.search(x_query_vec, k)
    return D, I

def calc_grobal_top_k_acc(embedding_query, embedding_target, k=10):
    # top-k accを計算
    top_k_correct_samplenums = []
    top_k_acc = []
    n_data = embedding_target.shape[0]                    # データベースのサイズ(database size)
    for i in range(k):
        # k近傍のインデックスのarray内にクエリベクトル自身が入っていればok
        correct_samplenum = get_bool_of_corrected_predictions(embedding_query=embedding_query, 
                                                              embedding_target=embedding_target, 
                                                              k=i+1).sum() 
        correct_samplenum = correct_samplenum.astype(np.float)
        top_k_correct_samplenums.append(correct_samplenum)
        top_k_acc.append(correct_samplenum/n_data)
    return top_k_acc

def get_bool_of_corrected_predictions(embedding_query, embedding_target, k=10):
    D, I = search_kNN(embedding_query=embedding_query, embedding_target=embedding_target, k=k)
    series_idx = np.arange(embedding_target.shape[0]) # クエリベクトル自身のインデックス
    is_predict_correctly = np.isin(series_idx,  I[:, :k])
    return is_predict_correctly

def retrieve_materials_properties(metadata, torch_dataset):
    """
    metadata: 材料ごとのメタデータのdictをlistに入れたもの
    torch_dataset: metadataと突き合わせるためのPyTorchのdataset object
    metadataとtorch_datasetの中身の順番は一致していないといけない
    
    metadata dictの例：
    {'material_id': 'mp-1025051',
     'pretty_formula': 'YbB2Rh3',
     'energy_per_atom': -6.870412148333333,
     'energy': -41.22247289,
     'density': 10.61234910722115,
     'final_structure': Structure Summary
     （中略）
     'spacegroup.number': 191,
     'band_gap': 0.0,
     'formation_energy_per_atom': -0.7159613472222226,
     'total_magnetization': 0.0005206,
     'xrd_hist': array([0., 0., 0., ..., 0., 0., 0.])}
    """
    pretty_formula = []
    energy_per_atom = []
    energy = []
    density = []
    formation_energy_per_atom = []
    total_magnetization_uB = []
    total_magnetization_T = []
    band_gap = []
    sgr = []
    num_sites = []
    cell_volume = []
    weight = []
    material_id = []
    valid_material_ids = set(torch_dataset.data.material_id)
    const_µB_to_T = 9.27401007833 * 4*np.pi /10
    if 'e_above_hull' in metadata[0]:
        e_above_hull = []
    else:
        e_above_hull = None

    for data in metadata:
        if data['material_id'] in valid_material_ids:
            material_id.append(data['material_id'])
            pretty_formula.append(data['pretty_formula'])
            energy_per_atom.append(data['energy_per_atom'])
            energy.append(data['energy'])
            density.append(data['density'])
            formation_energy_per_atom.append(data['formation_energy_per_atom'])
            # missing value workaround
            if data["material_id"] == "mp-1245128":
                data['total_magnetization'] = 0.0
            total_magnetization_uB.append(data['total_magnetization'])
            total_magnetization_T.append(data['total_magnetization']/data['final_structure'].volume*const_µB_to_T)
            band_gap.append(data['band_gap'])
            sgr.append(data['spacegroup.number'])
            num_sites.append(data['final_structure'].num_sites)
            cell_volume.append(data['final_structure'].volume)
            weight.append(data['final_structure'].composition.weight)
        if e_above_hull is not None:
            e_above_hull.append(data['e_above_hull'])
        
    properties = {'energy_per_atom':energy_per_atom , 'energy':energy, 'density':density, 
                  'formation_energy_per_atom':formation_energy_per_atom, 
                  'total_magnetization_uB':total_magnetization_uB, 'total_magnetization_T':total_magnetization_T, 'band_gap':band_gap, 'sgr':sgr ,
                  'num_sites':num_sites, 'cell_volume':cell_volume, 'weight':weight}
    if e_above_hull is not None:
        properties['e_above_hull'] = e_above_hull
    return properties, material_id, pretty_formula

def plot_embedding(tsne_embedding_xrd, tsne_embedding_crystal, value_for_color, color_key=None):
    fig, (ax1, ax2) = plt.subplots(figsize=(11, 4), ncols=2, dpi=100)
    # ax1 = plt.subplot(121)
    pos = ax1.scatter(tsne_embedding_xrd[:, 0], tsne_embedding_xrd[:, 1],
                      c=value_for_color, s=8, linewidths=0.01, alpha=0.2)
    ax1.set_title('t-SNE visualization of xrd embedding')
    fig.colorbar(pos, ax=ax1)

    # ax2 = plt.subplot(122)
    pos = ax2.scatter(tsne_embedding_crystal[:, 0], tsne_embedding_crystal[:, 1],
                      c=value_for_color, s=8, linewidths=0.01, alpha=0.2)
    ax2.set_title('t-SNE visualization of crystal embedding')
    fig.colorbar(pos, ax=ax2)
    if color_key is not None:
        fig.suptitle(color_key, fontsize='large')

def retrieve_neighbour_materials(query_mp_id, embedding, embedding_metadata, n_neighbours=1000, use_gpu=True):
    """
    ある物質のembeddingについて近傍の物質を検索して可視化する

    Parameters
    ----------
    query_mp_id: str
        クエリする物質のmp_id (e.g. mp-764)
        
    embedding: array-like
        検索対象のembeddingのtensorやarray

    Returns
    -------
    retrieved_neighbours : pd.DataFrame
        クエリ近傍の物質のメタデータのdataframe
        
    disp: ipywidgets.widgets.widget_box.HBox
        クエリ近傍の物質の結晶構造を可視化するNGL Viewerのipython widget
        
    """
    idx = embedding_metadata.query('mp_id == @query_mp_id').index[0]
    D, I = search_kNN(embedding_query=embedding[idx].unsqueeze(0), embedding_target=embedding, k=n_neighbours, use_gpu=use_gpu)
    retrieved_neighbours = embedding_metadata.iloc[I.squeeze()]
    return retrieved_neighbours
