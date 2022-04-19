"""
实现attention
"""

import torch
import torch.nn as nn
import torch.functional as F
import config


class Attention(nn.Module):
    def __init__(self, method="general"):
        super(Attention,self).__init__()
        self.method = method
        assert method in ["dot","general","concat"], "method error"

        if self.method == "general":
            self.Wa = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)

        elif self.method == "concat":
            self.Wa = nn.Linear(config.hidden_size+config.decoder_hidden_size, config.decoder_hidden_size, bias=False)
            self.Va = nn.Linear(config.decoder_hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        '''
        :param hidden_state:[num_layer, batch_size, decoder_hidden_size]
        :param encoder_outputs:[batch_size, seq_len, encoder_hidden_size]
        :return:
        '''
        if self.method == "dot":
            hidden_state = hidden_state[-1,:,:].unsqueeze(-1) #[ batch_size, decoder_hidden_size, 1]
            attention_weights = encoder_outputs.bmm(hidden_state).squeeze(-1) #[batch_size, seq_len, 1]
            attention_weights = torch.softmax(attention_weights, dim=-1) #[batch_size, seq_len]

        elif self.method == "general":
            encoder_outputs = self.Wa(encoder_outputs)  #[batch_size, seq_len, decoder_hidden_size]
            # print("hidden_state:", hidden_state.shape)

            hidden_state = hidden_state[-1, :, :].unsqueeze(-1)  # [ batch_size, decoder_hidden_size, 1]
            attention_weights = encoder_outputs.bmm(hidden_state).squeeze(-1) #[batch_size, seq_len]
            attention_weights = torch.softmax(attention_weights, dim=-1)#[batch_size, seq_len]

        elif self.method == "concat":
            hidden_state = hidden_state[-1,:,:].squeeze(0) #[ batch_size, decoder_hidden_size]
            hidden_state = hidden_state.repeat(1, encoder_outputs.size(1),1) #[ batch_size, seq_len, decoder_hidden_size]
            concated = torch.cat([hidden_state, encoder_outputs], dim = -1) #[batch_size, seq_len, decoder_hidden_size+encoder_hidden_size]
            batch_size = encoder_outputs.size(0)
            encoder_seq_len = encoder_outputs.size(1)
            attention_weights = self.Va(F.tanh(self.Wa(concated.view((batch_size*encoder_seq_len,-1))))).squeeze(-1) #[batch_size*seq_len,1]
            attention_weights = torch.softmax(attention_weights.view(batch_size, encoder_seq_len), dim=-1) #[batch_size, seq_len]

        return attention_weights

