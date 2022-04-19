import string
import jieba
import jieba.posseg as psg
import logging

# 关闭jieba日制
jieba.setLogLevel(logging.INFO)

jieba.load_userdict(r"G:\Python_projection\chat_bot\corpus\keywords.txt")

stopwords_path = r"G:\Python_projection\chat_bot\corpus\stop_words.txt"
stopwords = [i.strip() for i in open(stopwords_path, encoding="utf-8").readlines()]

continue_words = string.ascii_lowercase + string.digits + '+'


def _cut_sentence_by_word(sentence):
    '''
    按照单个字进行分词
    :param sentence:
    :return:
    '''
    temp = ""
    result = []
    for word in sentence:
        if word.lower() in continue_words:
            temp += word
        else:
            if len(temp) > 0:
                result.append(temp.lower())
                temp = ""
            result.append(word.strip())
    if len(temp) > 0:
        result.append(temp.lower())
    return result


def _cut_sentence(sentence, use_stopwords, use_seg):
    '''
    按照词语进行分词
    :param sentence:"python和c++哪个难？" ——>[python, 和， c++， 哪， 个， 难， ？]
    :return:
    '''
    if not use_seg:
        result = jieba.lcut(sentence)
    else:
        result = [(i.word, i.flag) for i in psg.lcut(sentence)]
    if use_stopwords:
        if not use_seg:
            result = [i for i in result if i not in stopwords]
        else:
            result = [i for i in result if i[0] not in stopwords]

    return result


def cut(sentence, by_word=False, use_stopwords=False, use_seg=False):
    '''
    封装上述方法
    :param sentence:
    :param by_word:是否按照单个字分词
    :param use_stopwords:是否使用停用词
    :param use_seg:是否返回词性
    :return:
    '''
    if by_word:
        return _cut_sentence_by_word(sentence)
    else:
        return _cut_sentence(sentence, use_stopwords, use_seg)


if __name__ == '__main__':
    print(_cut_sentence_by_word("python和c++哪个难"))
    print(cut("python和c++哪个难", use_seg=False, use_stopwords=False))