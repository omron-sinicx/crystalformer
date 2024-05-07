from typing import TypeVar, Type, List
import copy
Entity = TypeVar('Entity', bound='LatticeformerParams')

class LatticeformerParams:
    def __init__(self, 
                 domain:str="real",
                 lattice_range:int=4,
                 minimum_range:bool=True,
                 adaptive_cutoff_sigma:float=-3.5,
                 gauss_lb_real:float=0.5,
                 gauss_lb_reci:float=0.5,
                 scale_real:List[float]=[1.4],
                 scale_reci:List[float]=[2.2],
                 normalize_gauss:bool=True,
                 value_pe_dist_real:int=64,
                 value_pe_wave_real:int=0,
                 value_pe_dist_reci:int=0,
                 value_pe_wave_reci:int=0,
                 value_pe_headed:bool=True,
                 value_pe_condproj:str="no",
                 positive_func:str='elu=0.1',
                 exclude_self:bool=False,
                 layer_index:int=-1,
                 norm_func_mode:int=0,
                 value_pe_dist_max:float=-10.0,
                 value_pe_width_scale:float=1.0,
                 gauss_state:str="q",
                 use_low_memory:bool=False,
                 ) -> None:

        self.layer_index = layer_index
        self.domain = domain
        self.lattice_range = lattice_range
        self.minimum_range = minimum_range
        self.adaptive_cutoff_sigma = adaptive_cutoff_sigma
        self.gauss_lb_real = gauss_lb_real
        self.gauss_lb_reci = gauss_lb_reci
        self.scale_real = scale_real
        self.scale_reci = scale_reci
        self.normalize_gauss = normalize_gauss
        self.value_pe_dist_real = value_pe_dist_real
        self.value_pe_wave_real = value_pe_wave_real
        self.value_pe_dist_reci = value_pe_dist_reci
        self.value_pe_wave_reci = value_pe_wave_reci
        self.value_pe_headed = value_pe_headed
        self.value_pe_condproj = value_pe_condproj
        self.positive_func = positive_func
        self.exclude_self = exclude_self
        self.norm_func_mode = norm_func_mode
        self.value_pe_dist_max = value_pe_dist_max
        self.value_pe_width_scale = value_pe_width_scale
        self.gauss_state = gauss_state
        self.use_low_memory = use_low_memory

    def parseFromArgs(self, args):
        for key in self.__dict__:
            self.__dict__[key] = getattr(args, key, self.__dict__[key])
        print("Parsed LatticeformerParams:")
        print(self.__dict__)

    def getLayerParameters(self, layer_index) -> Entity:
        if self.domain in ("real", "reci", "multihead"):
            domain = self.domain
        else:
            domains = self.domain.split('-')
            domain = domains[layer_index % len(domains)]

        scale_real = self.scale_real
        scale_reci = self.scale_reci
        if isinstance(scale_real, (list,tuple)):
            scale_real = scale_real[layer_index % len(scale_real)]
        if isinstance(scale_reci, (list,tuple)):
            scale_reci = scale_reci[layer_index % len(scale_reci)]

        params = copy.deepcopy(self)
        params.domain = domain
        params.scale_real = scale_real
        params.scale_reci = scale_reci
        params.layer_index = layer_index
        return params
    