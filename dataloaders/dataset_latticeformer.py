import torch

from torch_geometric.data import Data
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from dataloaders.common import MultimodalDatasetMP, RegressionDatasetMP
from dataloaders.common import generate_site_species_vector
from .common import CellFormat


def make_data(material, ATOM_NUM_UPPER, cell_format:CellFormat):
    if "final_structure" in material:
        structure = material['final_structure']
    elif "structure" in material:
        structure = material['structure']
    else:
        raise AttributeError("Material has no structure!")
    if cell_format == CellFormat.CONVENTIONAL:
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    elif cell_format == CellFormat.PRIMITIVE:
        structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        # # assert len(structure.cart_coords) == len(primitive.cart_coords), f"{len(structure.cart_coords)}, {len(primitive.cart_coords)}"


    if "material_id" in material:
        id = material['material_id']
    elif "file_id" in material:
        id = material['file_id']
    else:
        id = material['id']

    atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    atom_fea = generate_site_species_vector(structure, ATOM_NUM_UPPER)
    data = Data(x=atom_fea, y=None, pos=atom_pos)
    data.trans_vec = torch.tensor(structure.lattice.matrix, dtype=torch.float)[None]
    data.material_id = id
    data.sizes = torch.tensor([atom_pos.shape[0]], dtype=torch.long)
    return data

class MultimodalDatasetMP_Latticeformer(MultimodalDatasetMP):
    def __init__(self, params, target_split, target_set=None, post_filter=None):
        self.use_primitive = params.use_primitive if hasattr(params, 'use_primitive') else True

        super(MultimodalDatasetMP_Latticeformer, self).__init__(target_split, target_set, post_filter)
    
    @property
    def processed_file_names(self):
        if self.use_primitive:
            return 'processed_data_latticeformer.pt'
        else:
            return 'processed_data_convcell_latticeformer.pt'

    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.use_primitive)

    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()


class RegressionDatasetMP_Latticeformer(RegressionDatasetMP):
    def __init__(self, target_split, target_set=None, cell_format:CellFormat=CellFormat.PRIMITIVE, post_filter=None):
        self.model_name = "latticeformer"
        super(RegressionDatasetMP_Latticeformer, self).__init__(target_split, target_set, cell_format, post_filter)
    
    def process_input(self, material):
        return make_data(material, self.ATOM_NUM_UPPER, self.cell_format)
    
    # In torch_geometric.data.dataset.Dataset, these functions are checked
    # if exist in self.__class__.__dict__.keys(). But __dict__ does not capture
    # the inherited functions. So, here explicitly claim the process and download functions
    def process(self):
        super().process()
    def download(self):
        super().download()
