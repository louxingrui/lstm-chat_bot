'''
准备语料库
'''
import string
from lib import cut
from tqdm import tqdm



def prepare_STC(by_word = False):
    path = r'G:\Python_projection\chat_bot\corpus\STC.json'
    if by_word:
        input_path = r'G:\Python_projection\chat_bot\corpus\STC_train_byword.txt'
        target_path = r'G:\Python_projection\chat_bot\corpus\STC_test_byword.txt'
    else:
        input_path = r'G:\Python_projection\chat_bot\corpus\STC_train.txt'
        target_path = r'G:\Python_projection\chat_bot\corpus\STC_test.txt'
    f_input = open(input_path, "a", encoding='utf-8')
    f_target = open(target_path, "a", encoding='utf-8')
    # flag = 0

    one_qa_pair = [] #保存一个问答对
    num = 0
    for line in tqdm(open(path, 'r', encoding='utf-8').readlines(), ascii=True, desc="处理STC语料"):
        if line[0] == ("{"):
            continue
        elif line[0] == ("["):
            continue
        elif line[0] == ("]"):
            continue
        else:
            line = line.replace(" ", "")
            line = line[1:].strip().lower()
            # if filter(line):
            #     flag = 2
            #     continue
            # if flag == 2:
            #     flag = 0
            #     continue
            line_cuted = cut(line, by_word=by_word)
            line_cuted = " ".join(line_cuted)
            if len(one_qa_pair) == 0:
                line_cuted = line_cuted[:-3]
                one_qa_pair.append(line_cuted)
            elif len(one_qa_pair) == 1:
                line_cuted = line_cuted[:-1]
                one_qa_pair.append(line_cuted)
            if len(one_qa_pair) == 2:
                f_input.write(one_qa_pair[0]+"\n")
                f_target.write(one_qa_pair[1]+"\n")
                one_qa_pair = []
                num += 1

            # if flag == 0:   #问
            #     f_input.write(line)
            #     flag = 1
            # else:   #答
            #     f_target.write(line)
            #     flag = 0
    f_input.close()
    f_target.close()
    return num

if __name__ == "__main__":
    num = prepare_STC(by_word=True)
