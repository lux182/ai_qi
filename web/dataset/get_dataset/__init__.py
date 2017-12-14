from ..base import DATA_SET_PATH
import os
import pickle as pk

def get_all_dataset_filename():
    '''
    获取全部的数据集的文件名称
    :return:
    '''
    if os.path.exists(DATA_SET_PATH):
        return os.listdir(DATA_SET_PATH)

def get_all_dataset_info_list():
    '''
    获取所有数据集的info
    :return:
    '''
    infolist = []
    for name in get_all_dataset_filename():
        dataset = get_dataset(name)
        infolist.append(dataset.info)
    return infolist

def get_dataset(name):
    '''
    根据数据集的文件名称获取数据集对象
    :param name:
    :return:
    '''
    filepath = DATA_SET_PATH + os.path.sep + name
    with open(filepath,'rb') as f:
        return pk.load(f)

def remove_dataset(name):
    '''
    移除指定的数据集
    :param name:
    :return:
    '''
    path = DATA_SET_PATH + os.path.sep + name
    os.remove(path)

def remove_all():
    '''
    删除所有数据集
    :return:
    '''
    for name in get_all_dataset_filename():
        remove_dataset(name)