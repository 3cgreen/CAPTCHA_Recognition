import cv2
import numpy as np
from urllib import request
from matplotlib import pyplot as plt
import Binarization

'''
def grey(pic):
    return pic.convert('L')
'''


def grey(pic):
    return cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)


def binarizationOrigin(pic):
    picgray = np.array(pic)
    height, width = picgray.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            greyvalue = picgray[i, j]
            if greyvalue <= 170:
                picgray[i, j] = 0
            else:
                picgray[i, j] = 255
    for i in range(height):
        picgray[i, 0] = 255
        picgray[i, width-1] = 255
    for j in range(width):
        picgray[0, j] = 255
        picgray[height-1, j] = 255
    return picgray


def binarization(pic):
    # im = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    # plt.subplot(131), plt.imshow(im, "gray")
    # plt.title("source image"), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.hist(im.ravel(), 256)
    # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
    threshold, th1 = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    plt.subplot(133), plt.imshow(th1, "gray")
    plt.title("OTSU,threshold is " + str(threshold)), plt.xticks([]), plt.yticks([])
    # plt.show()
    return th1, threshold


def pic_download(picname):
    url = 'http://my.cnki.net/Register/CheckCode.aspx'
    res = request.urlopen(url)
    img = res.read()
    with open(r'E:\Pictures\CAPTCHA\Cnki\CAPTCHA_Cnki%(no)04d.jpg' % {'no': picname}, 'wb') as f:
        f.write(img)


def pic_download_N(picname):
    url = 'http://my.cnki.net/Register/CheckCode.aspx'
    res = request.urlopen(url)
    img = res.read()
    with open(r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d.jpg' % {'no': picname}, 'wb') as f:
        f.write(img)


def noise_eliminate(picarray):
    height, width = picarray.shape
    neighbor = 0
    for i in range(height):
        for j in range(width):
            if i-1 >= 0 and picarray[i-1, j] == 0:
                neighbor += 1
            if i+1 < height and picarray[i+1, j] == 0:
                neighbor += 1
            if j-1 >= 0 and picarray[i, j-1] == 0:
                neighbor += 1
            if j+1 < width and picarray[i, j+1] == 0:
                neighbor += 1
            if neighbor == 0:
                picarray[i, j] = 255
                continue
            if i-1 >= 0 and j-1 >= 0 and picarray[i-1, j-1] == 0:
                neighbor += 1
            if i-1 >= 0 and j+1 < width and picarray[i-1, j+1] == 0:
                neighbor += 1
            if i+1 < height and j+1 < width and picarray[i+1, j+1] == 0:
                neighbor += 1
            if i+1 < height and j-1 >= 0 and picarray[i+1, j-1] == 0:
                neighbor += 1
            if neighbor < 2:
                picarray[i, j] = 255
            neighbor = 0
    return picarray


def slice(pic):
    plt.subplot(121), plt.imshow(pic, "gray"), plt.axis("off")
    (h, w) = pic.shape  # 返回高和宽
    # print(h,w)#s输出高和宽
    a = [0 for z in range(0, w)]
    print(a)  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if pic[i, j] == 0:  # 如果改点为黑点
                a[j] += 1  # 该列的计数器加一计数
                pic[i, j] = 255  # 记录完后将其变为白色

    #
    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
            pic[i, j] = 0  # 涂黑

    plt.subplot(122), plt.imshow(pic, "gray"), plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # for i in range(50):
    #     pic_no = i + 1
    #     pic_download_N(pic_no)
    for i in range(6,50):
        uri = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d.jpg' % {'no': i + 1}
        uril = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_G.jpg' % {'no': i + 1}
        urib = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_S.jpg' % {'no': i + 1}
        urin = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_X.jpg' % {'no': i + 1}
        image = cv2.imread(uri)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.subplot(221), plt.imshow(image, "gray"), plt.axis("off")
        plt.title("Origin Gray Graph")
        otsuimage, threshold = binarization(image)
        plt.subplot(222), plt.imshow(otsuimage, "gray"), plt.axis("off")
        plt.title("OTSU,threshold is " + str(threshold))
        image = binarizationOrigin(image)
        plt.subplot(223), plt.imshow(image, "gray"), plt.axis("off")
        plt.title("Binarized Graph")
        image = noise_eliminate(image)
        plt.subplot(224), plt.imshow(image, "gray"), plt.axis("off")
        plt.title("Eliminated Noise Graph")
        plt.show()
        slice(otsuimage)
        exit(0)
        # image = binarization(image)
        # cv2.imwrite(urin, image)
        # binarization(image)
        # with Image.open(uri, 'r') as image:
        #     binarization(image)
        #     grey(image).save(uril, 'gif')
        #     Image.fromarray(np.uint8(binarization(grey(image)))).save(urib, 'gif')
        #     Image.fromarray(np.uint8(noise_eliminate(binarization(grey(image))))).save(urin, 'gif')
