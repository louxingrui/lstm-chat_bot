'''
完成数据集的准备
'''

from torch.utils.data import Dataset, DataLoader
import config
import torch
class ChatbotDataset(Dataset):
    def __init__(self):
        self.input_path = config.chatbot_input_path
        self.target_path = config.chatbot_target_path
        self.input_lines = open(self.input_path, encoding='utf-8').readlines()
        self.target_lines = open(self.target_path, encoding='utf-8').readlines()

        assert len(self.input_lines) == len(self.target_lines)


    def __getitem__(self, index):
        input = self.input_lines[index].strip().split()
        target = self.target_lines[index].strip().split()
        input_length = len(input) if len(input) < config.input_max_len else config.input_max_len
        target_length = len(target) if len(target) < config.target_max_len+1 else config.target_max_len+1
        return input, target, input_length, target_length

    def __len__(self):
        return len(self.input_lines)

def collate_fn(batch):
    '''

    :param batch:[(input, target, input_length, target_length),...]
    :return:
    '''

    #1.排序
    batch = sorted(batch, key=lambda x:x[2], reverse=True)
    input, target, input_length, target_length = zip(*batch)
    #transform
    input = [config.chatbot_ws_input.transform(i, max_len = config.input_max_len) for i in input]
    target = [config.chatbot_ws_target.transform(i, max_len = config.target_max_len, add_eos = True) for i in target]
    input = torch.LongTensor(input)
    target  = torch.LongTensor(target)
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)
    return input, target, input_length, target_length




train_data_loader = DataLoader(ChatbotDataset(), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)