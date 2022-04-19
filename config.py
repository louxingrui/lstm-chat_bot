input_path = r'G:\Python_projection\chat_bot\corpus\input.txt'
target_path = r'G:\Python_projection\chat_bot\corpus\target.txt'
import torch
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#################chatbot#############
chatbot_by_word = True
if chatbot_by_word:
    chatbot_input_path = r'G:\Python_projection\chat_bot\corpus\input_byword.txt'
    chatbot_target_path = r'G:\Python_projection\chat_bot\corpus\target_byword.txt'
else:
    chatbot_input_path = r'G:\Python_projection\chat_bot\corpus\input.txt'
    chatbot_target_path = r'G:\Python_projection\chat_bot\corpus\target.txt'

#ws
if chatbot_by_word:
    chatbot_ws_input_path = 'model/chatbot/ws_byword_input_path.pkl'
    chatbot_ws_target_path = 'model/chatbot/ws_byword_target_path.pkl'
else:
    chatbot_ws_input_path = 'model/chatbot/ws_input_path.pkl'
    chatbot_ws_target_path = 'model/chatbot/ws_target_path.pkl'


chatbot_ws_input = pickle.load(open(chatbot_ws_input_path, "rb"))
chatbot_ws_target = pickle.load(open(chatbot_ws_target_path, "rb"))

batch_size = 128
chatbot_embedding_dim = 256
num_layers = 1
hidden_size = 128

decoder_num_layers = 1
decoder_hidden_size = 128

teach_forcing = 0.7
beam_width = 5
model_save_path = "model/chatbot/seq2seq_2.model" if chatbot_by_word else "model/chatbot/seq2seq_byword_2.model"
optimizer_save_path = "model/chatbot/optimizer_2.model" if chatbot_by_word else "model/chatbot/optimizer_byword_2.model"

if chatbot_by_word:
    input_max_len = 20
    target_max_len = 20
else:
    input_max_len = 12
    target_max_len = 12