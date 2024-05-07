import jarvis
import os
import pathlib
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms

from tqdm import tqdm
import random
import numpy
import pickle
import math

def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(numpy.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


if __name__ == '__main__':
    # If SSL certification error occurs Under proxy, the following workaround may work.
    # os.environ['REQUESTS_CA_BUNDLE'] = ''
    # os.environ['CURL_CA_BUNDLE'] = ''

    # print(jarvis.__path__)
    # cached_files = pathlib.Path(jarvis.__path__[0]).glob("db/*.zip")
    # for file in cached_files:
    #     print(f"Removing {file.absolute()}")
    #     os.remove(file.absolute())

    datasets = [
        "megnet",
        "dft_3d_2021",
        "dft_3d_2021",
        "dft_3d_2021",
    ]
    save_names = [
        "jarvis__megnet",
        "jarvis__dft_3d_2021",
        "jarvis__dft_3d_2021-mbj_bandgap",
        "jarvis__dft_3d_2021-ehull",
    ]
    used_vals = [
        {
            'id': ('material_id', str),
            'gap pbe': ('bandgap', float),
            'e_form': ('e_form', float),
            'structure': ('structure', object)
        },
        {
            'structure': ('structure', object),
            'jid': ('material_id', str),
            'formation_energy_peratom': ('formation_energy', float),
            'optb88vdw_total_energy': ('total_energy', float),
            'optb88vdw_bandgap': ('opt_bandgap', float),
        },
        {
            'structure': ('structure', object),
            'jid': ('material_id', str),
            'mbj_bandgap': ('mbj_bandgap', float),
        },
        {
            'structure': ('structure', object),
            'jid': ('material_id', str),
            'ehull': ('ehull', float),
        }
    ]

    for i, t in enumerate(datasets):
        try:
            print(f"Processing dataset: {t}")
            data = jdata(t)
            new_data = []
            print(data[0])
            for x in tqdm(data):
                atoms = Atoms(
                    lattice_mat=x['atoms']['lattice_mat'],
                    coords=x['atoms']['coords'],
                    elements=x['atoms']['elements'],
                    cartesian=x['atoms']['cartesian'],
                )
                x['structure'] = atoms.pymatgen_converter()

                new_x = {}
                ok = True
                for key in used_vals[i]:
                    newkey, vtype = used_vals[i][key]
                    val = x[key]
                    new_x[newkey] = val
                    if vtype == int and type(val) != int:
                        ok = False
                        break

                    elif vtype == float:
                        if type(val) == int:
                            x[newkey] = float(val)
                        elif type(val) == float and not math.isnan(val) and not math.isinf(val):
                            pass
                        else:
                            ok = False
                            break
                    
                    elif vtype == str and val is None:
                        x[newkey] = ""
                    elif vtype == str and type(val) != str:
                        x[newkey] = str(val)

                if ok:
                    new_data.append(new_x)
            
            print(f"filtered: {len(data)} -> {len(new_data)} ({len(new_data) - len(data)})")
            data = new_data
            
            print("Printing the first item...")
            for k in data[0]:
                print(f"{k}\t: {data[0][k]}")

            if t == "megnet":
                id_train, id_val, id_test = get_id_train_val_test(
                    len(data),
                    n_train=60000,
                    n_val=5000,
                    n_test=4239
                )
            else:
                id_train, id_val, id_test = get_id_train_val_test(
                    len(data),
                )

            splits = {}
            splits['train'] = [data[i] for i in id_train]
            splits['val'] = [data[i] for i in id_val]
            splits['test'] = [data[i] for i in id_test]
            splits['all'] = data

            print("Saving split data...")
            for key in splits:
                save_dir = f"{save_names[i]}/{key}/raw"
                os.makedirs(save_dir, exist_ok=True)

                print(f"{key}\t:{len(splits[key])}")
                with open(f"{save_dir}/raw_data.pkl", "wb") as fp:
                    pickle.dump(splits[key], fp)
            
        except Exception as e:
            raise e
