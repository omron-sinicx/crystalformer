import os
import requests
import pickle
from tqdm import tqdm
from jarvis.core.atoms import Atoms

# Prepare bulk and shear megnet dataset from the Matformer paper (NeurIPS 22):
# https://github.com/YKQ98/Matformer/tree/569a7e9331b2acacc184fab38f5f6085e46a9881 
urls = [
    'https://figshare.com/ndownloader/files/40258705',
    'https://figshare.com/ndownloader/files/40258675',
    'https://figshare.com/ndownloader/files/40258666',
    'https://figshare.com/ndownloader/files/40258681',
    'https://figshare.com/ndownloader/files/40258684',
    'https://figshare.com/ndownloader/files/40258678'
]

elems = {
    'bulk': {
    'id': ('material_id', str),
    'bulk modulus': ('bulk_modulus', float),
    'structure': ('structure', object)
    },
    'shear': {
    'id': ('material_id', str),
    'shear modulus': ('shear_modulus', float),
    'structure': ('structure', object)
    },
}

count = 0
for target in ["bulk", "shear"]:
    outdir = f'jarvis__megnet-{target}'
    os.makedirs(outdir, exist_ok=True)

    all = []
    for split in ["train", "val", "test"]:
        url = urls[count]
        filename = f'{outdir}/{target}_megnet_{split}.pkl'

        # under proxy, use verify=False to avoid an SSL error.
        urlData = requests.get(url, verify=False).content
        with open(filename ,mode='wb') as f:
            f.write(urlData)

        with open(filename, mode="rb") as fp:
            data = pickle.load(fp)
        
        new_data = []
        for x in tqdm(data):
            atoms = Atoms(
                lattice_mat=x['atoms']['lattice_mat'],
                coords=x['atoms']['coords'],
                elements=x['atoms']['elements'],
                cartesian=x['atoms']['cartesian'],
            )
            x['structure'] = atoms.pymatgen_converter()

            new_x = {}
            for key, (newkey, vtype) in elems[target].items():
                val = x[key]

                if vtype == float and type(val) != float:
                    val = float(val)
                elif vtype == int and type(val) != int:
                    val = int(val)
                elif vtype == str and type(val) != str:
                    val = str(val)

                new_x[newkey] = val
            new_data.append(new_x)

        os.makedirs(f'{outdir}/{split}/raw', exist_ok=True)
        with open(f'{outdir}/{split}/raw/raw_data.pkl', mode="wb") as fp:
            pickle.dump(new_data, fp)
        print(new_data[0])
        
        count += 1