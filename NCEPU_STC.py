import cv2
import imageio
import numpy as np
from PIL import Image
from urllib import request
from matplotlib import pyplot as plt


def grey(pic):
    return pic.convert('L')


def binarization(pic):
    im = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    plt.subplot(221), plt.imshow(im, "gray")
    plt.title("source image"), plt.axis("off")
    blur = cv2.GaussianBlur(im, (3, 3), 0)
    plt.subplot(223), plt.imshow(blur, "gray"), plt.axis("off")
    plt.title("Blured image")
    # plt.subplot(221), plt.hist(pic.ravel(), 256)
    # plt.title("Histogram")
    threshold2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    plt.subplot(222), plt.imshow(th2, "gray")
    plt.title("OTSU blur,threshold is " + str(threshold2)), plt.axis("off")
    threshold, th1 = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    plt.subplot(224), plt.imshow(th1, "gray")
    plt.title("OTSU,threshold is " + str(threshold)), plt.axis("off")
    plt.show()
    return th1, threshold


def binarizationOrigin(pic):
    picgray = np.array(pic)
    height, width = picgray.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            greyvalue = picgray[i, j]
            if greyvalue <= 220:
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


'''
from svmutil import *
dictionarypath = r'D:\libsvm-3.23\heart_scale'
y, x = svm_read_problem(dictionarypath)
m = svm_train(y[:200], x[:200], '-c 500')
p_label, p_acc, p_val = svm_predict(y[:200], x[:200], m)
'''


def pic_download(picname):
    url = 'http://219.226.132.42/CheckCode.aspx'
    res = request.urlopen(url)
    img = res.read()
    with open(r'E:\Pictures\CAPTCHA\NCEPU-STC\CAPTCHA%(no)04d.jpg' % {'no': picname}, 'wb') as f:
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
            if neighbor < 3:
                picarray[i, j] = 255
            neighbor = 0
    return picarray


def sliceCast(pic):
    plt.subplot(121), plt.imshow(pic, cmap="Greys_r")
    (h, w) = pic.shape  # 返回高和宽
    array = np.zeros(w, dtype=int)
    for i in range(h):
        for j in range(w):
            if pic[i, j] == 0:
                array[j] += 1
    plt.subplot(122), plt.plot(np.arange(w), array)
    print(array)
    plt.show()


if __name__ == '__main__':
    # for i in range(50):
    #     pic_no = i + 1
    #     pic_download(pic_no)
    for i in range(50):
        uri = r'E:\Pictures\CAPTCHA\NCEPU-STC\CAPTCHA%(no)04d.jpg' % {'no': i + 1}
        # uri = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d.jpg' % {'no': i + 1}
        uril = r'E:\Pictures\CAPTCHA\NCEPU-STC\CAPTCHA%(no)04d_G.jpg' % {'no': i + 1}
        urib = r'E:\Pictures\CAPTCHA\NCEPU-STC\CAPTCHA%(no)04d_S.jpg' % {'no': i + 1}
        urin = r'E:\Pictures\CAPTCHA\NCEPU-STC\CAPTCHA%(no)04d_X.jpg' % {'no': i + 1}
        # image = cv2.imread(uri)
        image = Image.open(uri, "r")
        # image = imageio.imread(uri)
        if image is None:
            print("opencv failed")
            # image = Image.open(uri, "r")
            image = imageio.imread(uri)
            if image is None:
                print("none")
                exit(0)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2RGBA)
        # binarization(image)
        picture = Image.open(urin, "r")
        sliceCast(np.array(picture))
        # with Image.open(uri, 'r') as image:
        #     grey(image).save(uril, 'gif')
        #     Image.fromarray(np.uint8(binarizationOrigin(grey(image)))).save(urib, 'gif')
        #     Image.fromarray(np.uint8(noise_eliminate(binarizationOrigin(grey(image))))).save(urin, 'gif')
