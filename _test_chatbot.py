'''
测试chatbot相关的api
'''
from wor_sequence import Word_seq
from config import *
import pickle
from dataset import train_data_loader
from train import train
from eval import eval

def save_ws():
    ws = Word_seq()
    for line in open(chatbot_input_path, 'r', encoding='utf-8').readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(chatbot_ws_input_path, 'wb'))

    ws = Word_seq()
    for line in open(chatbot_target_path, 'r', encoding='utf-8').readlines():
        ws.fit(line.strip().split())
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open(chatbot_ws_target_path, 'wb'))


def test_data_loader():
    for idx, (input, target, input_length, target_length) in enumerate(train_data_loader):
        print("idx:", idx)
        print("input:", input)
        print("target", target)
        print("input_length", input_length)
        print("target_length", target_length)
        break

def train_seq2seq():
    for i in range(20):
        train(i)

if __name__ == "__main__":
    # save_ws()

    # test_data_loader()

    train_seq2seq()
    # eval()