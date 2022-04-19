from dataset import train_data_loader
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
import config
from tqdm import tqdm
import torch
#训练
#1.实例化model,optimizer,loss
#2.遍历dataloader
#3.调用模型output
#4.计算损失
#5.模型保存和加载

seq2seq = Seq2Seq()
seq2seq = seq2seq.to(config.device)
optimizer = Adam(seq2seq.parameters(), lr=0.001)
def train(epoch):
    bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), ascii=True, desc="train")
    for idx,(input, target, input_length, target_length) in bar:
        input = torch.LongTensor(input).to(device=config.device)
        target = torch.LongTensor(target).to(device=config.device)
        input_length = torch.LongTensor(input_length)
        target_length = torch.LongTensor(target_length)


        optimizer.zero_grad()
        decoder_outputs, _ = seq2seq(input, target, input_length, target_length)
        #decoder_outputs 和target在二维和三维无法进行损失计算
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1)
        target = target.view(-1)
        loss = F.nll_loss(decoder_outputs, target, ignore_index=config.chatbot_ws_input.PAD)
        loss.backward()
        optimizer.step()
        bar.set_description("epoch:{}\tidx:{}\tloss:{:.3f}".format(epoch,idx,loss.item()))

        torch.save({
            "epoch": epoch,
            "model_state_dict": seq2seq.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, config.model_save_path)



if __name__ == "__main__":
    for i in range(10):
        train(i)