"""
模型评估
"""
from seq2seq import Seq2Seq
import config
from tqdm import tqdm
import torch
import numpy as np
from lib import cut
#训练
#1.准备测试数据

#2.实例化模型，加载model
def eval(by_word = True):
    seq2seq = Seq2Seq().to(config.device)
    seq2seq.load_state_dict(torch.load(config.model_save_path))


    while True:
        input_sentence = input("请输入：")
        input_sentence = cut(input_sentence, by_word=True)
        input_sentence = torch.LongTensor([config.chatbot_ws_input.transform(input_sentence, max_len = config.input_max_len)]).to(config.device)
        input_length = torch.LongTensor([len(input_sentence) if len(input_sentence) < config.input_max_len else config.input_max_len])
        indices = np.array(seq2seq.evaluate(input_sentence, input_length)).flatten()
        outputs = config.chatbot_ws_target.inverse_transform(indices=indices)
#3.获取预测值
        print("回复：", outputs)
#4. 反序列化，观察结果

