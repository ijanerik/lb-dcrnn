from PIL import Image, ImageFont, ImageDraw
import glob, random, math, time
from collections import OrderedDict

differendAdresses = 100000
imagesPerAdress = 1

## Write to roman
def write_roman(num):
    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"
    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num > 0:
                roman_num(num)
            else:
                break

    return "".join([a for a in roman_num(num)])


def generateAdresNumber():
    if(random.random() < 0.02):
        number = random.randint(0, 1200)
    elif(random.random() < 0.80):
        number = abs(math.floor(random.gauss(10, 100)))
    else:
        number = abs(math.floor(random.gauss(0, 40)))
    return int(number) + 1

def generateSuffixNumber():
    if(random.random() < 0.5):
        return 0
    else:
        return random.choice([1, 2, 3, 4, 'A', 'B', 'C', 'D'])

def giveSuffixTypes(suffix):
    if(suffix == 0):
        if(random.random() < 0.4):
            return ''
        else:
            return random.choice(['hs', 'bg','BG', 'HS', 'h'])
    elif(isinstance(suffix, str)):
        return random.choice([suffix.lower(), suffix.upper()])
    else:
        return random.choice([suffix, 'i'*suffix, 'I'*suffix, '|'*suffix, write_roman(suffix), write_roman(suffix).lower(), write_roman(suffix).replace('I', '|')])

def generateBetweenNumber(category):
    if(category == 0):
        return ''
    else:
        return random.choice(['-', ' ', '  ', '/', ' / '])

def makeTextImage(name, text, font, loc, size, color):
    im = Image.new("L", size, "white")
    draw = ImageDraw.Draw(im)
    draw.text(loc, text, font=font, fill=color)
    im.save('images/'+name+'.gif')

def loadFonts(dir):
    fonts = []
    for filename in glob.glob(dir+'/*.ttf'):
        fonts.append(ImageFont.truetype(filename, 16))
    return fonts

fonts = loadFonts('fonts')

def generateAdres():
    number1 = generateAdresNumber()
    number2 = generateSuffixNumber()
    return (number1, number2)

def generateAdresText((number1, number2)):
    return str(number1) + generateBetweenNumber(number2) + str(giveSuffixTypes(number2))

def generateTextImage(id, text, loc, size):
    makeTextImage(str(id), str(text), random.choice(fonts), loc, size, "black")

# GENERATE!!!! :O
start = time.time()
for i in range(differendAdresses):
    (number1, number2) = generateAdres()
    print(number1, number2)

    for j in range(imagesPerAdress):
        text = generateAdresText((number1, number2))
        loc = (random.randint(0, 5), random.randint(0,5))
        generateTextImage(str(i)+'-'+str(j)+'_'+str(number1)+'-'+str(number2), text, loc, (60,30))

end = time.time()
print(end - start)
