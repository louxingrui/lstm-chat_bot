def joint_sentence(path1, path2):
    f1 = open(path1, 'r', encoding='utf-8')
    f2 = open(path2, 'w', encoding='utf-8')
    for line in f1.readlines():
        newline = line.replace(" ", "")
        f2.write(newline)

path1 = '../corpus/STC_test_byword.txt'
path2 = '../corpus/STC_test.txt'

if __name__ == "__main__":
    joint_sentence(path1, path2)