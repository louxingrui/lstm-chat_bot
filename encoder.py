'''
编码器
'''
from config import *
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(chatbot_ws_input),
                                      embedding_dim=chatbot_embedding_dim,
                                      padding_idx=chatbot_ws_input.PAD)
        self.gru = nn.GRU(input_size=chatbot_embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

    def forward(self, input, input_length):
        # print("input, input_lengthL", input.size(), input_length)
        embed = self.embedding(input)
        # print("embedded:", embed.size())
        embed = pack_padded_sequence(embed, input_length, batch_first=True)
        out, hidden = self.gru(embed)
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=chatbot_ws_input.PAD)

        #out_size = [batch_size, sen_len, hidden_size]
        #hidden = [1, batch_size, hidden_size]
        return out, hidden, out_length

if __name__ == "__main__":
    from dataset import  train_data_loader
    encoder = Encoder()
    print(encoder)
    for input, target, input_length, target_length in train_data_loader:
        out, hidden, out_length = encoder(input, input_length)
        print(out.size())
        print(hidden.shape)
        print(out_length)
        break