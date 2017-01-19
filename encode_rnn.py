import numpy as np

def encodeHouseNumber(data, maxLen):
    chars = list(map(str, range(10))) + list(map(chr, range(65, 91))) + ['-', ' ']
    ret = np.zeros((len(data), maxLen, len(chars)))

    for i, text in enumerate(data):
        diffLength = maxLen - len(text)
        for o, char in enumerate(text):
            ret[i][diffLength+o][chars.index(char)] = 1
    return ret
