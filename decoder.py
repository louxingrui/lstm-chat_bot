'''
解码器
'''
from config import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from attention import Attention
import heapq

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(chatbot_ws_target),
                                      embedding_dim=chatbot_embedding_dim,
                                      padding_idx=chatbot_ws_target.PAD)

        self.gru = nn.GRU(input_size=chatbot_embedding_dim,
                          hidden_size=decoder_hidden_size,
                          num_layers=decoder_num_layers,
                          batch_first=True)

        self.fc = nn.Linear(decoder_hidden_size, len(chatbot_ws_target))
        self.attn = Attention()
        self.Wa = nn.Linear(hidden_size+decoder_hidden_size, decoder_hidden_size, bias=False)



    def forward(self, target, encoder_hidden, encoder_outputs):
        #1. 获取encoder的输出，作为decoder的第一次的hidden_state
        decoder_hidden = encoder_hidden
        batch_size = target.size(0)
        #2. 准备decoder第一个时间步的输入，[batch_size, 1] SOS 作为输入
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * chatbot_ws_target.SOS).to(device)

        #3. 在第一个时间步上进行计算，得到第一个时间步的输出，hidden_state
        #4. 把前一个时间步的输出进行计算，得到第一个最后的输出的结果
        #5. 把前一次的hidden_state作为当前是时间步的hidden_state的输入，把前一次的输出，作为当前时间步的输入
        #6. 循环4-5步骤

        #保存预测的结果
        decoder_outputs = torch.zeros([batch_size, target_max_len+1, len(chatbot_ws_target)]).to(device)

        if random.random() > teach_forcing:
            for t in range(target_max_len+1):
                decode_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                #保存decoder_output_t到decoder_ouputs中
                decoder_outputs[:, t, :] = decode_output_t
                decoder_input = target[:, t].unsqueeze(-1)
                # print("decoder_input:", decoder_input.size())
        else:
            for t in range(target_max_len + 1):
                decode_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
                # 保存decoder_output_t到decoder_ouputs中
                decoder_outputs[:, t, :] = decode_output_t
                value, index = torch.topk(decode_output_t, 1)
                decoder_input = index

        return decoder_outputs, decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        '''
        计算每个时间步上的结果
        :param decoder_input:[batch_size, 1]
        :param decoder_hidden:[1, batch_size, hidden_size]
        :return:
        '''

        decoder_input_embeded = self.embedding(decoder_input) #[batch_size, 1, embedding_dim]

        #out: [batch_size, 1, hidden_size]
        #decoder_hidden: [1, batch_size, hidden_size]
        out,decoder_hidden = self.gru(decoder_input_embeded)

        ###############   添加attention ###############
        #[batch_size, input_seq_len] * [batch_size,input_seq_len,encoder_hidden_size]
        attention_weights = self.attn(decoder_hidden, encoder_outputs).unsqueeze(1)
        context_vector = attention_weights.bmm(encoder_outputs) #[batch_size, 1, encoder_hidden_size]
        concated = torch.cat([out, context_vector], dim=-1).squeeze(1) #[batch_size, 1, encoder_hidden_size + decoder_hidden_size]
        out = torch.tanh(self.Wa(concated)) #[batch_size, hidden_size]
        ##############################################


        # out = out.squeeze(1) #[batch_size, hidden_size]
        out_put = F.log_softmax(self.fc(out), dim=-1)   #[batch_size, vocab_size]
        # print("output:", out_put.size()) #Size([128, 14])

        return out_put, decoder_hidden


    def evaluate(self, encoder_hidden, encoder_outputs):
        '''评估'''
        decoder_hidden = encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * chatbot_ws_input.SOS).to(device)
        indices = []

        for i in range(target_max_len+5):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs=encoder_outputs)
            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index
            # if index == num_sequence.EOS:
            #     break
            indices.append(index.squeeze(-1).cpu().detach().numpy())
        return indices

    #beam search的评估
    def eval_beam_search_heapq(self, encoder_outputs, encoder_hidden):
        '''
        使用堆来完成beam search,堆是一种优先级队列, 按照优先级顺序存取数据
        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        '''
        batch_size = encoder_hidden.size(1)
        #1. 构造第一次需要的输入数据,保存在堆中
        decoder_input = torch.LongTensor([[chatbot_ws_input.SOS]*batch_size]).to(device)
        decoder_hidden = encoder_hidden

        prev_beam = Beam()
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)
        while True:
            cur_beam = Beam()
            #2.取出堆中数据,进行forward_step操作,获得当前时间步的output,hidden
            #这里使用下划线区分
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden in prev_beam:
                #判断前一次的_complete是否为True,如果是,则不需要forward
                #有可能为True,但是概率并不是最大
                if _complete == True:
                    cur_beam.add(_probility, _complete, _seq, _decoder_input, _decoder_hidden)
                else:
                    decoder_input_t, decoder_hidden = self.forward_step(_decoder_input, _decoder_hidden, encoder_outputs)

                    value, index = torch.topk(decoder_input_t, beam_width)
                    #3.从output中选择topk个输出,作为下一次的input
                    for m, n in zip(value[0], index[0]):
                        decoder_input = torch.LongTensor([[n]]).to(device)
                        seq = _seq + [n]
                        probility = _probility + m
                        if n.item() == chatbot_ws_target.EOS:
                            complete = True
                        else:
                            complete = False

                            #4.把下一个实践步骤需要的输入数据保存在一个新的堆中
                        cur_beam.add(probility, complete, seq, decoder_input, decoder_hidden)
            #5.获取新的堆中的优先级最高的数据,判断数据是否是EOS结尾或者达到最大长度
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)

            if best_complete == True or len(best_seq) - 1 == target_max_len +1 :
                return self._prepare_seq(best_seq)
            else:
                #6.重新遍历新的堆中的数据
                prev_beam = cur_beam

    def _prepare_seq(self, seq):
        if seq[0].item() == chatbot_ws_target.SOS:
            seq = seq[1:]
        if seq[-1].item() == chatbot_ws_target.EOS:
            seq = seq[:-1]
        seq = [i.item() for i in seq]
        return seq





class Beam:
    def __init__(self):
        self.heap = list() #保存数据位置
        self.beam_width = beam_width

    def add(self, probility, complete, seq, decoder_input, decoder_hidden):
        '''
        添加数据,同时判断总的数据个数,多则删除
        :param probility:
        :param complete:
        :param seq:
        :param decoder_input:
        :param decoder_hidden:
        :return:
        '''
        heapq.heappush(self.heap, [probility, complete, seq, decoder_input, decoder_hidden])
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        return iter(self.heap)
