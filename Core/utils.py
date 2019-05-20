# -*- coding: UTF-8 -*
from torch.utils.data import DataLoader

def get_SVR_loaders(SVRDataset, regular_config, data_set_config):
    loader_dict = {}
    for key in ['train', 'val', 'test']:
        if key not in data_set_config:
            continue
        value = data_set_config[key]
        loader_set = SVRDataset(value)
        loader_dict[key] = DataLoader(loader_set, batch_size=regular_config['batch_size'],
                                      shuffle=value['shuffle'],
                                      num_workers=regular_config['num_workers'])
    return loader_dict
