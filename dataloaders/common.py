import os
import warnings
import pandas as pd
import torch
import numpy
import pymatgen
#from tqdm import tqdm
from functools import partial
from enum import Enum

import contextlib
from typing import Optional
import joblib
from tqdm.auto import tqdm

@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs):
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)
    
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, separate
from torch_geometric.data.separate import separate
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


def generate_site_species_vector(structure: pymatgen.core.structure.Structure, ATOM_NUM_UPPER):

    if hasattr(structure, 'species'):
        atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
        x_species_vector = torch.eye(ATOM_NUM_UPPER)[atom_num - 1].squeeze()

    else:
        x_species_vector = []
        for site in structure.species_and_occu:
            site_species_and_occupancy = []
            # サイトの各元素について、one-hot encodingした上で占有率をかけて、元素ごとの占有率ベクトルを計算
            for elem in site.elements:
                if type(elem) == pymatgen.core.Element:
                    occupancy = site.element_composition[elem]
                elif type(elem) == pymatgen.core.periodic_table.Specie or type(elem) == pymatgen.core.periodic_table.Species:
                    occupancy = site.element_composition[elem.element]
                elif type(elem) == pymatgen.core.composition.Composition:
                    occupancy = site.element_composition[elem.element]
                    # print(elem, occupancy)
                elif type(elem) == pymatgen.core.periodic_table.DummySpecie or type(elem) == pymatgen.core.periodic_table.DummySpecies:
                    raise ValueError(f'Unsupported specie: {site}! Skipped')
                else:
                    print(site, type(site))
                    raise AttributeError
                atom_num = torch.tensor(elem.Z, dtype=torch.long)
                elem_onehot = torch.eye(ATOM_NUM_UPPER)[atom_num - 1]
                site_species_and_occupancy.append(elem_onehot*occupancy)
            # サイトの各元素についてのone-hot vectorのsumとって、サイトごとの占有率に変換
            site_species_and_occupancy_sum = torch.stack(site_species_and_occupancy).sum(0)
            x_species_vector.append(site_species_and_occupancy_sum)
        x_species_vector = torch.stack(x_species_vector, 0)
        
    if x_species_vector.dim() == 1:
        x_species_vector.unsqueeze_(0)
    return x_species_vector


def filter_by_atom_num(data, min_val=0, max_val=0):
    # Set the default n > 1. This is to ensure that
    # when data has neither pos nor x (eg, xrd data)
    # the code returns True (ie, not exclude).
    n = 2
    if hasattr(data, 'pos') and data.pos is not None:
        n = data.pos.shape[0]
    elif hasattr(data, 'x') and data.x is not None:
        n = data.x.shape[0]

    if min_val <= n  and (n <= max_val or max_val <= 0):
        return True

    return False

def exclude_one_atom_crystal(data):
    return partial(filter_by_atom_num, min_val=2)(data)


def try_to_get_xrd(material):
    if 'xrd_hist' in material:
        return material['xrd_hist']

    c = XRDCalculator()
    structure = material['final_structure']
    two_theta_range = (10, 110)
    diff_peaks = c.get_pattern(structure, two_theta_range=two_theta_range, scaled=False)
    xrd_hist = numpy.histogram(diff_peaks.x, bins=5000, range=two_theta_range,
                            weights=diff_peaks.y)

    return xrd_hist[0]


class MultimodalDatasetMP(InMemoryDataset):
    def __init__(self, target_split, target_set=None, post_filter=None):
        self.ATOM_NUM_UPPER = 98
        if target_set is None:
            target_set = "less_than_quinary20200608_with_xrd_10_110"
            urls = {
                'train':'https://multimodal.blob.core.windows.net/train/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_10k':'https://multimodal.blob.core.windows.net/train-10k/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_6400':'https://multimodal.blob.core.windows.net/train-6400/less_than_quinary20200608_with_xrd_10_110.zip',
                'val':'https://multimodal.blob.core.windows.net/val/less_than_quinary20200608_with_xrd_10_110.zip',
                'test':'https://multimodal.blob.core.windows.net/test/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_on_all':'https://multimodal.blob.core.windows.net/all/less_than_quinary20200608_with_xrd_10_110.zip'
            }
            self.url = urls[target_split]
        
        root = f"data/{target_set}/{target_split}"

        super(MultimodalDatasetMP, self).__init__(root, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        if post_filter is not None:
            print("Applying post filtering...")
            n = len(self)
            data_list = []
            for idx in tqdm(range(n)): 
                data = separate(
                    cls=self.data.__class__,
                    batch=self.data,
                    idx=idx,
                    slice_dict=self.slices,
                    decrement=False,
                )
                if post_filter(data):
                    data_list.append(data)
            self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return 'raw_data.pkl'

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def download(self):
        """
        Azure Storageからファイルをダウンロードしてraw以下に展開
        """
        print(self.url)
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)

        # rename the unzipped file to "raw_data.pkl"
        if not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names)):
            fname = os.path.splitext(os.path.basename(path))[0]
            path_src = os.path.join(self.raw_dir, fname + ".pkl")
            os.rename(path_src, path)

        os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        crystals = pd.read_pickle(self.raw_paths[0])
        print('loaded data: ', self.raw_paths[0])

        data_list = []
        for material in crystals:
            data = self.process_input(material)
            data.xrd = try_to_get_xrd(material)
            data.xrd = torch.tensor(data.xrd, dtype=torch.float)[None, None]
            if data is None:
                continue
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CellFormat(Enum):
    RAW = 1
    PRIMITIVE = 2
    CONVENTIONAL = 3

class RegressionDatasetMP(InMemoryDataset):
    """
    PyTorch Geometric (PyG)のDatasetクラス
    MaterialsProjectから書き出しておいたpymatgenのstructureオブジェクトから、
    PyGで取り扱える点群データセットを生成する
    """

    def __init__(self, target_split, target_set=None, cell_format:CellFormat=CellFormat.PRIMITIVE, post_filter=None):
        self.ATOM_NUM_UPPER = 98
        self.model_name = ""
        self.cell_format = cell_format

        if target_set is None:
            target_set = "less_than_quinary20200608_with_xrd_10_110"
            urls = {
                'train':'https://multimodal.blob.core.windows.net/train/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_10k':'https://multimodal.blob.core.windows.net/train-10k/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_6400':'https://multimodal.blob.core.windows.net/train-6400/less_than_quinary20200608_with_xrd_10_110.zip',
                'val':'https://multimodal.blob.core.windows.net/val/less_than_quinary20200608_with_xrd_10_110.zip',
                'test':'https://multimodal.blob.core.windows.net/test/less_than_quinary20200608_with_xrd_10_110.zip',
                'train_on_all':'https://multimodal.blob.core.windows.net/all/less_than_quinary20200608_with_xrd_10_110.zip'
            }
            self.url = urls[target_split]
        
        root = f"data/{target_set}/{target_split}"
        
        # ローカルのデータセットを読む時用
        #self.load_filename = 'less_than_quinary_asof2021_06_10_with_xrd.pkl'

        super().__init__(root, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if post_filter is not None:
            print("Applying post filtering...")
            n = len(self)
            data_list = []
            for idx in tqdm(range(n)): 
                data = separate(
                    cls=self.data.__class__,
                    batch=self.data,
                    idx=idx,
                    slice_dict=self.slices,
                    decrement=False,
                )
                if post_filter(data):
                    data_list.append(data)
            self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return 'raw_data.pkl'

    @property
    def processed_file_names(self):
        if self.cell_format == CellFormat.RAW:
            return f'processed_data_{self.model_name}.pt'
        if self.cell_format == CellFormat.PRIMITIVE:
            return f'processed_data_primitive_{self.model_name}.pt'
        if self.cell_format == CellFormat.CONVENTIONAL:
            return f'processed_data_convcell_{self.model_name}.pt'

        raise NotImplementedError()

    def download(self):
        """
        Azure Storageからファイルをダウンロードしてraw以下に展開
        """
        path = download_url(self.url, self.root)
        extract_zip(path, self.raw_dir)

        # rename the unzipped file to "raw_data.pkl"
        if not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names)):
            fname = os.path.splitext(os.path.basename(path))[0]
            path_src = os.path.join(self.raw_dir, fname + ".pkl")
            os.rename(path_src, os.path.join(self.raw_dir, self.raw_file_names))

        os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        print(f'loaded data: {self.raw_paths[0]}')
        crystals = pd.read_pickle(self.raw_paths[0])

        # Get the items of the first and last entries to guess
        # the common items in the dataset.
        keys0 = [key for key in crystals[0] if crystals[0][key] is not None]
        keys1 = [key for key in crystals[-1] if crystals[-1][key] is not None]
        attrs = { key: key for key in keys0 + keys1 }
        if "band_gap" in attrs:
            attrs["band_gap"] = "bandgap"
        if "spacegroup.number" in attrs:
            attrs["spacegroup.number"] = "sgr_class"
        print(attrs.keys())

        @joblib.delayed
        def func(material):
            # Check if all the items exist in this material and skip if any unavailable.
            ok = True
            for key in attrs:
                v = material.get(key, None)
                if v is None:
                    warnings.warn(f'Warning! Value "{key}" is None in {material["material_id"]}. Skipped.')
                    ok = False
            if not ok:
                return None 

            data = self.process_input(material)
            if data is None:
                return None

            for key in attrs:
                setattr(data, attrs[key], material[key])

            return data
        
        # data_list = [func(material) for material in tqdm(crystals)]
        with tqdm_joblib(len(crystals)):
            data_list = joblib.Parallel(n_jobs=-1)(
            func(material) for material in crystals
        )
        data_list = [data for data in data_list if data is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
class MultimodalDatasetCOD(InMemoryDataset):
    def __init__(self, target_split):
        self.ATOM_NUM_UPPER = 98
        # target_data: "cif_sc", "cif_NOTsc", "cif_thermoelectric", "cif_NOTthermoelectric"
        root = "data/cod/" + target_split

        urls = {'sc':'https://multimodal.blob.core.windows.net/cod/cod_sc_20210628_primitive.pkl',
                'NOTsc':'https://multimodal.blob.core.windows.net/cod/cod_NOTsc_20210628_primitive.pkl',
                'thermoelectric':'https://multimodal.blob.core.windows.net/cod/cod_thermoelectric_20210628_primitive.pkl',
                'NOTthermoelectric':'https://multimodal.blob.core.windows.net/cod/cod_NOTthermoelectric_20210628_primitive.pkl',
                'ferroelectric':'https://multimodal.blob.core.windows.net/cod/cod_ferroelectric_20210714_primitive.pkl',
                'NOTferroelectric':'https://multimodal.blob.core.windows.net/cod/cod_NOTferroelectric_20210714_primitive.pkl',
                'ferromagnetic':'https://multimodal.blob.core.windows.net/cod/cod_ferromagnetic_20210715_primitive.pkl',
                'NOTferromagnetic':'https://multimodal.blob.core.windows.net/cod/cod_NOTferromagnetic_20210715_primitive.pkl',
                'antiferromagnetic':'https://multimodal.blob.core.windows.net/cod/cod_antiferromagnetic_20210715_primitive.pkl',
                'NOTantiferromagnetic':'https://multimodal.blob.core.windows.net/cod/cod_NOTantiferromagnetic_20210715_primitive.pkl'}
        self.url = urls[target_split]
        self.target_data = target_split
        super(MultimodalDatasetCOD, self).__init__(root, pre_filter=exclude_one_atom_crystal)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'{self.url.split("/")[-1]}'

    @property
    def processed_file_names(self):
        raise NotImplementedError()

    def download(self):
        """
        Azure Storageからファイルをダウンロードしてraw以下に展開
        """
        path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.unlink(path)
        
    def process_input(self, material):
        raise NotImplementedError()

    def process(self):
        crystals = pd.read_pickle(self.raw_paths[0])
        print('loaded data: ', self.raw_paths[0])

        data_list = []
        for material in tqdm(crystals):
            try:
                assert material['file_id'] is not None
                assert material['formula'] is not None
                assert material['title'] is not None
                assert material['journal'] is not None
                assert material['year'] is not None
                assert material['svnrevision'] is not None
                
                data = self.process_input(material)
                if data is None:
                    continue
                # data（←PyGのdata型）自体は、データセットに合わせてmake_dataメソッドを定義して、このクラスを継承したクラス内で処理する
                # structure = material['structure']
                # atom_pos, atom_species = structure2tensor(structure=structure)
                # data = Data(x=atom_species, y=None, pos=atom_pos)
                data.material_id = material['file_id']
                data.pretty_formula = material['formula']
                data.title = material['title']
                data.journal = material['journal']
                data.title = material['title']
                data.year = material['year']
                data.svnrevision = material['svnrevision']
                # set a dummy XRD pattern for compatibility
                data.xrd = torch.zeros(5000)
                data_list.append(data)
            except AssertionError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except AttributeError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except IndexError as e:
                print(e)
                print(f"material id: {material['file_id']}")
            except ValueError as e:
                print(e)                
                print(f"material id: {material['file_id']}")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])