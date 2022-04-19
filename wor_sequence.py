

class Word_seq:
    PAD_TAG = "PAD"
    UNK_TAG = "UNK"
    SOS_TAG = "SOS"
    EOS_TAG = "EOS"

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS}

        self.count = {}

    def fit(self, sentence):
        '''
        传入句子，词频统计
        :param sentence:
        :return:
        '''
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1



    def build_vocab(self, min_count = 5, max_count= None, max_feature = None):
        '''
        构造词典
        :param min_count:
        :param max_count:
        :param max_feature:
        :return:
        '''

        temp = self.count.copy()    #copy()是真正的复制
        for key in temp:
            cur_count = self.count.get(key, 0)
            if min_count is not None:
                if cur_count < min_count:
                    del self.count[key]
            if max_count is not None:
                if cur_count > max_count:
                    del self.count[key]
        if max_feature is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x:x[1], reverse=True)[:max_feature])

        for key in self.count:
            self.dict[key] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))




    def transform(self, sentence, max_len=None, add_eos=False):
        '''
        把sentence转化为数字序列
        :param sentence: list
        :param max_len:
        :return: list
        '''
        '''
        eg1:
        sentence :11
        max_len :10
        eg2:
        sentence：8
        max_len: 10
        add_eos:true 输出句子的长度为max_len+1
        add_eos:false 输出句子的长度为
        '''

        if len(sentence) > max_len:
            sentence = sentence[:max_len]

        sentence_len = len(sentence)    #提前计算句子长度，句子长度通通变11

        if add_eos:
            sentence = sentence + [self.EOS_TAG]

        if sentence_len < max_len:
            sentence = sentence + [self.PAD_TAG] * (max_len-sentence_len)


        result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        '''
        把数字序列转化为字符串
        :param indices:
        :return:
        '''
        result = []
        for i in indices:
            if i == self.EOS:
                break
            result.append(self.inverse_dict.get(i, self.UNK_TAG))

        return result

    def __len__(self):
        return len(self.dict)
if __name__ == "__main__":
    num_seq = Num_seq()
    print(num_seq.dict)