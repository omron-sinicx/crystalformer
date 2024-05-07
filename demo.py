from argparse import ArgumentParser
import os
import sys
import torch
from torch_geometric.loader import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from utils import Params
from distutils.util import strtobool

def get_option():
    argparser = ArgumentParser(description='Training the network')
    argparser.add_argument('-p', '--param_file', type=str, default='default.json', help='filename of the parameter JSON')
    args, unknown = argparser.parse_known_args()
    return args

def demo():
    args = get_option()
    print('parsed args :')
    print(args)
    try:
        params = Params(f'{args.param_file}')
    except:
        params = Params(f'./params/{args.param_file}')

    parser = ArgumentParser(description='Training the network')
    parser.add_argument('-p', '--param_file', type=str, default=args.param_file, help='Config json file for default params')
    # load the json config and use it as default values.
    boolder = lambda x:bool(strtobool(x))
    typefinder = lambda v: str if v is None else boolder if type(v)==bool else type(v)
    for key in params.dict:
        v = params.dict[key]
        if isinstance(v, (list, tuple)):
            parser.add_argument(f"--{key}", type=typefinder(v[0]), default=v, nargs='+')
        else:
            parser.add_argument(f"--{key}", type=typefinder(v), default=v)
    params.__dict__ = parser.parse_args().__dict__
    print(params.dict)

    import models.global_config as config
    config.REPRODUCIBLITY_STATE = getattr(params, 'reproduciblity_state', 0)
    print(f"reproduciblity_state = {config.REPRODUCIBLITY_STATE}")

    # Reproducibility
    seed = getattr(params, 'seed', 123)
    deterministic = params.encoder_name in ["latticeformer"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.preferred_linalg_library("cusolver") # since torch 1.11, needed to avoid an error by torch.det(), but now det_3x3 is implemented manually.

    from dataloaders.dataset_latticeformer import RegressionDatasetMP_Latticeformer as Dataset
    from models.latticeformer import Latticeformer

    model = Latticeformer(params)
    param_num = sum([p.nelement() for p in model.parameters()])
    print(f"Whole: {param_num}, {param_num*4/1024**2} MB")
    param_num = sum([p.nelement() for p in model.encoder.layers[0].parameters()])
    print(f"Block: {param_num}, {param_num*4/1024**1} KB")
    
    if params.pretrained_model is not None:
        with open(params.pretrained_model, "rb") as f:
            ckeckpoint = torch.load(f)
            state_dict = ckeckpoint['state_dict']
            target_std = ckeckpoint['state_dict']['target_std']
            target_mean = ckeckpoint['state_dict']['target_mean']
            
            model_name = "swa_model.module."
            # if params.pretrained_model.endswith("best.ckpt"):
            #     model_name = "model."
            # else:
            #     model_name = "swa_model.module."

            print("model name:", model_name)
            model_dict = { key.replace(model_name, ""):state_dict[key] for key in state_dict if key.startswith(model_name) }
            model.load_state_dict(model_dict)
            # correct the last linear layer weights
            target_std = target_std.to(model.mlp[-1].weight.device)
            target_mean = target_mean.to(model.mlp[-1].weight.device)
            model.mlp[-1].load_state_dict({
                'weight': model.mlp[-1].weight * target_std[:,None],
                'bias': model.mlp[-1].bias * target_std + target_mean,
            })

    else:
        print("Specify --pretrained_model for demonstration.")
        exit()
    
    # Setup datasets
    target_set = getattr(params, "target_set", None)
    test_dataset = Dataset(target_split='test', target_set=target_set)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = model.cuda()
    model.eval()
    targets = params.targets if isinstance(params.targets, list) else [params.targets]

    with torch.no_grad():
        mae_err = {t: [] for t in targets}
        for batch in tqdm(test_loader):
            batch = batch.cuda()
            output = model(batch)
            for i, t in enumerate(targets):
                labels = batch[t]
                mae_err[t].append(abs(output[:, i] - labels).detach().cpu())

        for t in targets:
            print(f"{t}: {torch.cat(mae_err[t]).mean().item()}")


if __name__ == '__main__':
    demo()
