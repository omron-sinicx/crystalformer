
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_lightning.core import LightningModule
import torch.optim as optim
from losses.regression_loss import regression_loss

from models.latticeformer import Latticeformer
from torch.optim import lr_scheduler
import numpy
from torch.optim import swa_utils
from typing import Callable

class AvgFn:
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        return averaged_model_parameter + \
            (model_parameter - averaged_model_parameter) / (num_averaged + 1)
    
class RegressionModel(LightningModule):
    def __init__(self, params, train_loader, val_loader):
        super(RegressionModel, self).__init__()

        if params.encoder_name == 'latticeformer':
            self.model = Latticeformer(params)
        else:
            raise Exception(f"Invalid params.encoder_name: {params.encoder_name}")
        
        self.swa_epochs = getattr(params, "swa_epochs", 0)
        if self.swa_epochs > 0:
            # In DDP, classes can't have function pointers as members, so define avg_fn as a class.
            self.swa_model = swa_utils.AveragedModel(self.model)
        else:
            self.swa_model = None

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.save_hyperparameters(params.__dict__)
        self.automatic_optimization = False
        self.clip_norm = getattr(params, "clip_norm", 0.0)
        self.clip_grad = getattr(params, "clip_grad", 0.0)

        self.targets = self.params.targets
        if isinstance(self.targets, str):
            self.targets = [self.targets]
        self.val_history = []

        target_std = torch.ones((len(self.targets)), dtype=torch.float32)
        target_mean = torch.zeros((len(self.targets)), dtype=torch.float32)
        self.register_buffer('target_std', target_std)
        self.register_buffer('target_mean', target_mean)
        self.normalize_targets = getattr(params, "normalize_targets", "no")
        if self.normalize_targets in ("scale_bias", "bias", "scale"):
            self.update_target_normalizers()

        self.use_average_model:bool = False
        self.logging_key:str = None
        self.validation_step_outputs = []

    def enable_average_model(self, logging_key:str=None) -> bool:
        if self.swa_model is not None:
            self.logging_key = logging_key
            self.use_average_model = True
            return True
        return False
        
    def disable_average_model(self):
        self.logging_key = None
        self.use_average_model = False
    
    def update_target_normalizers(self):
        target_vals = [getattr(self.train_loader.dataset.data, t) for t in self.targets]
        target_vals = torch.stack(target_vals, dim=0)

        if "bias" in self.normalize_targets:
            target_mean = torch.mean(target_vals, dim=1)
        else:
            target_mean = torch.zeros((len(self.targets)), dtype=torch.float32)
        if "scale" in self.normalize_targets:
            target_std = ((target_vals-target_mean[:, None])**2).mean(dim=1)**0.5
        else:
            target_std = torch.ones((len(self.targets)), dtype=torch.float32)

        print("Computing normalizing scales and biases for target values ---")
        for i, t in enumerate(self.targets):
            print(f"{t}\t: scale={target_std[i].item()}\t bias={target_mean[i].item()}")
        print("-------------------------")

        self.target_std[:] = target_std.to(self.target_std.device)
        self.target_mean[:] = target_mean.to(self.target_mean.device)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Override for backward compatibility
        new_dict = {}
        for key in state_dict:
            if key.startswith("model.xrd_"):
                # replace 'model' with 'model_xrd'
                new_dict['model_xrd' + key[5:]] = state_dict[key]
            else:
                new_dict[key] = state_dict[key]

        return super().load_state_dict(new_dict, strict)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        freeze_bn_epochs = getattr(self.params, 'freeze_bn_epochs', 0)
        if self.current_epoch + freeze_bn_epochs >= self.params.n_epochs:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                    m.eval()

        output = self.forward(batch)
        loss = regression_loss(output, batch, self.targets, self.target_std, self.target_mean, self.params.loss_func)
        loss, bsz = loss.mean(), loss.shape[0]

        self.manual_backward(loss)
        if self.clip_norm > 0:
            total_norm = nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.log('train/total_norm', total_norm, on_step=False, on_epoch=True, \
                prog_bar=False, logger=True, batch_size=bsz)
        if self.clip_grad > 0:
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), self.clip_grad)
        opt.step()

        swa_enabled = self.swa_epochs+self.current_epoch >= self.params.n_epochs
        if swa_enabled:
            self.swa_model.update_parameters(self.model)

        sch = self.lr_schedulers()
        if sch is not None and not swa_enabled:
            sch.step()
        
        output = {'loss': loss}
        # note: make sure disabling on_step logging, which may frequently
        # cause unexpected pauses due to the logging IO when the disc is on a NAS.
        self.log('train/loss', loss, on_step=False, on_epoch=True, \
            prog_bar=False, logger=True, batch_size=bsz)
        return output

    def on_train_end(self):
        print("Updating BNs for Stochastic Weight Averaging")
        device = self.swa_model.parameters().__next__().device
        torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device)

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = regression_loss(output, batch, self.targets, self.target_std, self.target_mean, self.params.loss_func)

        out = {
            'val/loss': loss, 
            'output': output.detach().cpu(),
        }

        for i, t in enumerate(self.targets):
            labels = batch[t]
            out[t] = abs(output[:, i]*self.target_std[i]+self.target_mean[i] - labels).detach().cpu()

        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.cat([x['val/loss'] for x in outputs]).mean()
        
        print(f'\r\rval loss: {avg_loss:.3f} ', end='')
        logging_key = self.logging_key if self.logging_key is not None else 'val'
        self.log(f'{logging_key}/loss', avg_loss)
        for t in self.targets:
            v = torch.cat([x[t] for x in outputs], dim=0).mean()
            self.log(f'{logging_key}/' + t, v)
            print(f'{t}: {v.item():.3f} ', end='')
        print('   ')

        if self.logging_key is None:
            self.val_history.append(avg_loss.item())
            v = numpy.array(self.val_history, dtype=numpy.float64)
            K = 50  # mean filter width
            T = 10  # mean of top-T scores
            k = numpy.ones(K, dtype=numpy.float64)
            m = numpy.convolve(v, k)[:-(K-1)] / numpy.convolve(numpy.ones_like(v), k)[:-(K-1)]
            r = numpy.sort(v)[:min(len(v),T)]
            self.log("hp/avr50", m[-1])
            self.log("hp/min_avr50", m[~numpy.isnan(m)].min())
            self.log("hp/min", v[~numpy.isnan(v)].min())
            self.log("hp/mean_min10", r.mean())
            self.log("hp/val", avg_loss)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        
    def on_test_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.cat([x['val/loss'] for x in outputs]).mean()
        
        logging_key = self.logging_key if self.logging_key is not None else 'test'
        print(f'\r\rtest loss: {avg_loss:.3f} ', end='')
        self.log(f'{logging_key}/loss', avg_loss)
        for t in self.targets:
            v = torch.cat([x[t] for x in outputs], dim=0).mean()
            key = f'{logging_key}/' + t
            self.log(key, v)
            print(f'{t}: {v.item():.3f} ', end='')
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # opt may be re-initialized if the initial lr is different from lr,
        # ie, the lr lambda f does not satisfy f(0) = lr.
        lr = self.params.lr
        opt = getattr(self.params, 'optimizer', 'adam')
        weight_decay = getattr(self.params, 'weight_decay', 0.0)

        if weight_decay <= 0:
            opt_params = [{
                'params': self.model.parameters(),
            }]
        else:
            nodecay = []
            decay = []
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.SyncBatchNorm)):
                    nodecay.extend(m.parameters(False))
                else:
                    for name, param in m.named_parameters(recurse=False):
                        if "bias" in name:
                            nodecay.append(param)
                        else:
                            decay.append(param)
            opt_params = [
                {'params': nodecay},
                {'params': decay, 'weight_decay': weight_decay},
            ]
            num_nodecay = sum([p.numel() for p in nodecay])
            num_decay = sum([p.numel() for p in decay])
            num_total = sum([p.numel() for p in self.model.parameters()])
            print(f"# nodecay params = {num_nodecay}")
            print(f"# decay params = {num_decay}")
            print(f"# params = {num_total}")
            assert num_decay + num_nodecay == num_total

        opt_args = {
            'lr': lr
        }
        if opt == 'adam':
            opt_args['betas'] = self.params.adam_betas
            opt = optim.Adam
        elif opt == 'adamw':
            opt_args['betas'] = self.params.adam_betas
            opt = optim.AdamW
        elif opt == 'sgd':
            opt_args['momentum'] = self.params.momentum
            opt = optim.SGD
        else:
            return NotImplementedError(f'Unknown optimizer: {self.params.optimizer}')
        sch = None
        
        if self.params.lr_sch == "const":
            return opt(opt_params, **opt_args)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup":
            # used in the t-fixup paper
            decay = self.params.sch_params[0]
            f = lambda t: (decay / (decay + t))**0.5
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, f)
        elif self.params.lr_sch == "inverse_sqrt_nowarmup_dmodel":
            decay = self.params.sch_params[0]
            f = lambda t: self.params.model_dim**-0.5*(decay / (decay + t))**0.5
            opt_args['lr'] = lr*f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup":
            # used in the original transformer paper
            warmup_steps = self.params.sch_params[0]
            f = lambda t: self.params.model_dim**-0.5*min((t+1)**-0.5, (t+1)*warmup_steps**-1.5)
            opt_args['lr'] = f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "inverse_sqrt_warmup_lrmax":
            warmup_steps = self.params.sch_params[0]
            f = lambda t: warmup_steps**0.5*min((t+1)**-0.5, (t+1)*warmup_steps**-1.5)
            opt_args['lr'] = lr*f(0)
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.LambdaLR(opt, lambda t: f(t)/f(0))
        elif self.params.lr_sch == "multistep":
            opt = opt(opt_params, **opt_args)
            sch = lr_scheduler.MultiStepLR(opt, milestones=self.params.sch_params, gamma=0.1)
        else:
            return NotImplementedError(f'Unknown lr_sch: {self.params.lr_sch}')
        
        return [[opt], [sch]]

    def forward(self, x):
        if self.use_average_model and self.swa_model is not None:
            return self.swa_model(x)
        return self.model(x)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
