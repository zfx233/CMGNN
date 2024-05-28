import os
import json
import yaml
from texttable import Texttable
current_dir = os.path.dirname(os.path.abspath(__file__))


def setup_cfg(args, args_dict):
    """load experiment's changed model hyparamenter from yaml file
    
    Parameters
    ----------
    args : ArgumentParser
        ArgumentParser
    args_dict : dict
        dict of arguments from hyparamenter yaml file
    """
    with open(args_dict['config'], 'r', encoding='utf-8') as f:
        cfg_file = f.read()
        cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)[args_dict['dataset']]
        for k, v in cfg_dict.items():
            args.__setattr__(k, v)
    return args

def tab_printer(args) -> None:
    """Print args in table shape

    Parameters
    ----------
    args : ArgumentParser
        ArgumentParser
    """
    keys = sorted(args.keys())
    txt = Texttable()
    txt.set_precision(5)
    params = [["Parameter", "Value"]]
    params.extend([[
        k.replace("_", " "),
        f"{args[k]}" if isinstance(args[k], bool) else args[k]
    ] for k in keys])
    txt.add_rows(params)
    print(txt.draw())
