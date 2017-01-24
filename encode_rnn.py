import numpy as np

def encodeHouseNumber(data, maxLen):
    chars = list(map(str, range(10))) + list(map(chr, range(65, 91))) + ['-', ' ']
    ret = np.zeros((len(data), maxLen, len(chars)))

    for i, text in enumerate(data):
        diffLength = maxLen - len(text)
        for o in range(diffLength):
            ret[i][o][len(chars) - 1] = 1

        for o, char in enumerate(text):
            ret[i][diffLength+o][chars.index(char)] = 1
    return ret


def decodeHouseNumber(matrix, maxLen):
    chars = list(map(str, range(10))) + list(map(chr, range(65, 91))) + ['-', ' ']
    ret = []
    for i in matrix:
        ret.append(chars[np.argmax(i)])

    return ''.join(ret)
