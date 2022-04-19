'''
准备语料库
'''
import string
from lib import cut
from tqdm import tqdm

def filter(pair):
    if pair[0][1].strip() in list(string.ascii_lowercase):  # input只有一个字母时候舍弃
        return True
    elif pair[1][1].count("=") >= 2 and len(pair[1][0].split()) < 4:
        return True
    elif "黄鸡" in pair[0][1] or "黄鸡" in pair[1][1] or "小通" in pair[0][1] or "小通" in pair[1][1]:
        return True
    elif len(pair[0][0].strip()) == 0 or len(pair[1][0].strip()) == 0:
        return True

def prepare_xiaohuangji(by_word = False):
    path = r'G:\Python_projection\chat_bot\corpus\xiaohuangji50w_nofenci.conv'
    if by_word:
        input_path = r'G:\Python_projection\chat_bot\corpus\input_byword.txt'
        target_path = r'G:\Python_projection\chat_bot\corpus\target_byword.txt'
    else:
        input_path = r'G:\Python_projection\chat_bot\corpus\input.txt'
        target_path = r'G:\Python_projection\chat_bot\corpus\target.txt'
    f_input = open(input_path, "a", encoding='utf-8')
    f_target = open(target_path, "a", encoding='utf-8')
    # flag = 0

    one_qa_pair = [] #保存一个问答对
    num = 0
    for line in tqdm(open(path, 'r', encoding='utf-8').readlines(), ascii=True, desc="处理小黄鸡语料"):
        if line.startswith("E"):
            continue
        else:
            line = line[1:].strip().lower()
            # if filter(line):
            #     flag = 2
            #     continue
            # if flag == 2:
            #     flag = 0
            #     continue
            line_cuted = cut(line, by_word=by_word)
            line_cuted = " ".join(line_cuted)
            if len(one_qa_pair) < 2:
                one_qa_pair.append([line_cuted, line])
            if len(one_qa_pair) == 2:   #写入
                # assert len(one_qa_pair) == 2, "error"
                #判断句子是否需要
                if filter(one_qa_pair):
                    one_qa_pair = []
                    continue

                f_input.write(one_qa_pair[0][0]+"\n")
                f_target.write(one_qa_pair[1][0]+"\n")
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
    num = prepare_xiaohuangji()
    print(num)