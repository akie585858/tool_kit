from torch import Module

def load_params(net:Module, param_name:str='all'):
    if param_name == 'all':
        pass
