import numpy as np

def encodeHouseNumber(data, encodeTable, maxLen):
    chars = map(str, range(10)) + list(map(chr, range(65, 91))) + ['-', ' ']
    dater = np.array(encodeTable)[data]
    ret = np.zeros((len(dater), maxLen, len(chars)))

    for i, text in enumerate(dater):
        diffLength = maxLen - len(text)
        for o, char in enumerate(text):
            ret[i][diffLength+o][chars.index(char)] = 1
    return ret

# print encodeHouseNumber(np.array([3,1,2,0, 0]), ['12-d', '14-a', '143-0', '9999-4', '1-0'], 7)
