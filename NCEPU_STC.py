import cv2
import queue
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
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            greyvalue = picgray[i, j]
            if greyvalue <= 220:
                picgray[i, j] = 0
            else:
                picgray[i, j] = 255
    for i in range(height):
        picgray[i, 0] = 255
        picgray[i, width - 1] = 255
    for j in range(width):
        picgray[0, j] = 255
        picgray[height - 1, j] = 255
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
    with open(r'E:\Pictures\CAPTCHA\NCEPU-STC_1\CAPTCHA%(no)04d.jpg' % {'no': picname}, 'wb') as f:
        f.write(img)


def noise_eliminate(picarray):
    height, width = picarray.shape
    neighbor = 0
    for i in range(height):
        for j in range(width):
            if i - 1 >= 0 and picarray[i - 1, j] == 0:
                neighbor += 1
            if i + 1 < height and picarray[i + 1, j] == 0:
                neighbor += 1
            if j - 1 >= 0 and picarray[i, j - 1] == 0:
                neighbor += 1
            if j + 1 < width and picarray[i, j + 1] == 0:
                neighbor += 1
            if neighbor == 0:
                picarray[i, j] = 255
                continue
            if i - 1 >= 0 and j - 1 >= 0 and picarray[i - 1, j - 1] == 0:
                neighbor += 1
            if i - 1 >= 0 and j + 1 < width and picarray[i - 1, j + 1] == 0:
                neighbor += 1
            if i + 1 < height and j + 1 < width and picarray[i + 1, j + 1] == 0:
                neighbor += 1
            if i + 1 < height and j - 1 >= 0 and picarray[i + 1, j - 1] == 0:
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
    # print(array)
    plt.subplot(122), plt.plot(np.arange(w), array)
    plt.show()
    suffixoffset = 0
    originarray = array
    while array[0] == 0:
        array = np.delete(array, 0)
        suffixoffset += 1
    while array[len(array)-1] == 0:
        array = np.delete(array, len(array)-1)
    # print(array)
    counts = len(array)
    cutlocations = [suffixoffset-1]
    for x in range(counts):
        if array[x] == 0 and array[x-1] != 0:
            cutlocations.append(x + suffixoffset)
    cutlocations.append(len(array) + suffixoffset)
    print(cutlocations)
    x = 0
    while len(cutlocations) < 5:
        print(str(x+1) + " " + str(x) + " " + str(len(cutlocations)))
        if cutlocations[x + 1] - cutlocations[x] > 15:
            middlenum = int((cutlocations[x+1] - cutlocations[x]) / 2 + cutlocations[x])
            tempregion = originarray[middlenum - 4:middlenum + 3    :1]
            cutlocations.insert(x+1, np.argmin(tempregion) + middlenum - 2)
            # cutlocations.insert(x+1, middlenum)
            print(cutlocations)
        else:
            x += 1
    print(cutlocations)
    cutimage = Image.fromarray(pic)
    for x in range(4):
        box = (cutlocations[x], 0, cutlocations[x + 1], h)
        plt.subplot(141 + x), plt.imshow(cutimage.crop(box), "gray")
    plt.show()




def Opening(pic):
    pic_temp = pic.copy()
    height, width = pic_temp.shape
    for x in range(height):
        for y in range(width):
            pic_temp[x, y] = 255 - pic_temp[x, y]
    # 创建矩形结构单元
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 形态学处理,开运算
    img_open = cv2.morphologyEx(pic_temp, cv2.MORPH_OPEN, g)
    # plt.subplot(121), plt.imshow(pic_temp, "gray"), plt.title("Source Pic")
    # plt.subplot(122), plt.imshow(img_open, "gray"), plt.title("Opened Pic")
    # plt.show()
    height, width = img_open.shape
    for x in range(height):
        for y in range(width):
            img_open[x, y] = 255 - img_open[x, y]
    return img_open


def cfs(img):
    """传入二值化后的图片进行连通域分割"""
    w, h = img.shape
    visited = set()
    q = queue.Queue()
    offset = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    cuts = []
    for x in range(w):
        for y in range(h):
            x_axis = []
            # y_axis = []
            # y_axis = []
            if img[x, y] == 0 and (x, y) not in visited:
                q.put((x, y))
                visited.add((x, y))
            while not q.empty():
                x_p, y_p = q.get()
                for x_offset, y_offset in offset:
                    x_c, y_c = x_p + x_offset, y_p + y_offset
                    if (x_c, y_c) in visited:
                        continue
                    visited.add((x_c, y_c))
                    try:
                        if img[x_c, y_c] == 0:
                            q.put((x_c, y_c))
                            x_axis.append(x_c)
                            # y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x, max_x = min(x_axis), max(x_axis)
                if max_x - min_x > 3:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts.append((min_x, max_x))
    return cuts


if __name__ == '__main__':
    # for i in range(50):
    #     pic_no = i + 1
    #     pic_download(pic_no)
    for i in range(50):
        uri = r'E:\Pictures\CAPTCHA\NCEPU-STC_1\CAPTCHA%(no)04d.jpg' % {'no': i + 1}
        # uri = r'F:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d.jpg' % {'no': i + 1}
        uril = r'E:\Pictures\CAPTCHA\NCEPU-STC_1\CAPTCHA%(no)04d_G.jpg' % {'no': i + 1}
        urib = r'E:\Pictures\CAPTCHA\NCEPU-STC_1\CAPTCHA%(no)04d_S.jpg' % {'no': i + 1}
        urin = r'E:\Pictures\CAPTCHA\NCEPU-STC_1\CAPTCHA%(no)04d_X.jpg' % {'no': i + 1}
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
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        binarizedpic = binarizationOrigin(grey(image))
        noiseEliminatedpic = noise_eliminate(binarizedpic)
        # binarization(image)
        # picture = Image.open(uri, "r")
        # sliceCast(noiseEliminatedpic)
        cutarray = cfs(Opening(noiseEliminatedpic))
        sliceCast(Opening(noiseEliminatedpic))
        print(cutarray)
        # exit(0)
        # with Image.open(uri, 'r') as image:
        #     grey(image).save(uril, 'gif')
        #     Image.fromarray(np.uint8(binarizationOrigin(grey(image)))).save(urib, 'gif')
        #     Image.fromarray(np.uint8(noise_eliminate(binarizationOrigin(grey(image))))).save(urin, 'gif')
