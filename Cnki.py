import cv2
import queue
import numpy as np
from PIL import Image
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
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            greyvalue = picgray[i, j]
            if greyvalue <= 170:
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


def binarization(pic):
    # im = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    # plt.subplot(131), plt.imshow(pic, "gray")
    # plt.title("source image"), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.hist(pic.ravel(), 256)
    # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
    threshold, th1 = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
    # plt.subplot(133), plt.imshow(th1, "gray")
    # plt.title("OTSU,threshold is " + str(threshold)), plt.xticks([]), plt.yticks([])
    # plt.show()
    return th1, threshold


def pic_download(picname):
    url = 'http://my.cnki.net/Register/CheckCode.aspx'
    res = request.urlopen(url)
    img = res.read()
    with open(r'F:\Pictures\CAPTCHA\Cnki\CAPTCHA_Cnki%(no)04d.jpg' % {'no': picname}, 'wb') as f:
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
            if neighbor < 2:
                picarray[i, j] = 255
            neighbor = 0
    return picarray


def sliceCast(pic, times):
    if times > 5:
        return "Error"
    sample = pic.astype(np.uint8)
    # plt.subplot(121), plt.imshow(pic, cmap="Greys_r")
    (height, width) = pic.shape  # 返回高和宽
    array = np.zeros(width, dtype=int)
    for i in range(height):
        for j in range(width):
            if pic[i, j] == 0:
                array[j] += 1
    # plt.subplot(122), plt.plot(np.arange(w), array)
    # plt.show()
    # img1 = cv2.imread(r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki0001.jpg')
    img1 = sample.copy()
    # plt.imshow(img1, "gray"), plt.axis("off"), plt.title("Sources")
    # plt.show()
    background1 = np.zeros((height, width), dtype=int)
    background2 = np.zeros((height, width), dtype=int)
    background3 = np.zeros((height, width), dtype=int)
    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 如果图像是二值图，这一行就可以删除
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    level = hierarchy[0][0][2]
    nextcon = level
    contourslist = []
    while nextcon != -1:
        contourslist.append(contours[nextcon])
        nextcon = hierarchy[0][nextcon][0]
    if len(contourslist) < 4:
        return sliceCast(Opening(pic), times+1)
    if len(contourslist) > 4:
        return sliceCast(noise_eliminate(pic), times+1)
    rectangle = []
    for i in range(len(contourslist)):
        x, y, w, h = cv2.boundingRect(contourslist[i])
        rectangle.append((x, y, w, h))
        cv2.rectangle(background3, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # y + 2: y + h - 2, x + 2: x + w - 2
        # y - 1: y + h + 1, x - 1: x + w + 1
    # for i in range(len(contourslist)):
    #     rect = cv2.minAreaRect(contourslist[i])
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(background1, [box], 0, (255, 255, 255), 1)
    # # 画出轮廓，-1,表示所有轮廓，画笔颜色为(255, 255, 255)，即Green，粗细为1
    # cv2.drawContours(background2, contourslist, -1, (255, 255, 255), 1)
    # # 显示图片
    # plt.subplot(131), plt.imshow(background1, "gray_r"), plt.axis("off"), plt.title("MinAreaRectangle")
    # plt.subplot(132), plt.imshow(background2, "gray_r"), plt.axis("off"), plt.title("Contours")
    # plt.subplot(133), plt.imshow(background3, "gray_r"), plt.axis("off"), plt.title("Rectangle")
    # plt.show()
    pics = []
    for x in range(4):
        scale = rectangle[x]
        background2 = np.zeros((height, width), dtype=int)
        cv2.drawContours(background2, contourslist, x, (255, 255, 255), -1)
        piccopy = pic.copy()
        for i in range(height):
            for j in range(width):
                if background2[i, j] == 0:
                    piccopy[i, j] = 255
        picsegment = piccopy[scale[1]:scale[1] + scale[3], scale[0]:scale[0] + scale[2]]
        pics.append(picsegment)
    #     plt.subplot(2, 2, x+1), plt.imshow(picsegment, "gray")
    # plt.show()
    times += 1
    return pics



def Opening(pic):
    pic_temp = pic.copy()
    # plt.subplot(121), plt.imshow(pic_temp, "gray"), plt.title("Source Pic")
    height, width = pic_temp.shape
    for x in range(height):
        for y in range(width):
            pic_temp[x, y] = 255 - pic_temp[x, y]
    # 创建矩形结构单元
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 形态学处理,开运算
    img_open = cv2.morphologyEx(pic_temp, cv2.MORPH_OPEN, g)
    height, width = img_open.shape
    for x in range(height):
        for y in range(width):
            img_open[x, y] = 255 - img_open[x, y]
    # plt.subplot(122), plt.imshow(img_open, "gray"), plt.title("Opened Pic")
    # plt.show()
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
    for i in range(100, 200):
        pic_no = i + 1
        pic_download_N(pic_no)
        # exit(0)
    for i in range(100, 200):
        uri = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d.jpg' % {'no': i + 1}
        uril = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_G.jpg' % {'no': i + 1}
        urib = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_S.jpg' % {'no': i + 1}
        urin = r'E:\Pictures\CAPTCHA\Cnki_1\CAPTCHA_Cnki%(no)04d_X.jpg' % {'no': i + 1}
        image = cv2.imread(uri)
        imageg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # plt.subplot(221), plt.imshow(image, "gray"), plt.axis("off")
        # plt.title("Origin Gray Graph")
        otsuimage, threshold = binarization(imageg)
        # plt.subplot(222), plt.imshow(otsuimage, "gray"), plt.axis("off")
        # plt.title("OTSU,threshold is " + str(threshold))
        imagem = binarizationOrigin(imageg)
        # plt.subplot(223), plt.imshow(image, "gray"), plt.axis("off")
        # plt.title("Binarized Graph")
        imagem = noise_eliminate(imageg)
        # plt.subplot(224), plt.imshow(image, "gray"), plt.axis("off")
        # plt.title("Eliminated Noise Graph")
        # plt.show()
        # cutarray = cfs(otsuimage)
        # print(cutarray)
        # samplepic = Opening(otsuimage)
        # sliceCast(samplepic)
        picassembly = sliceCast(otsuimage, 0)
        if picassembly != "Error":
            for p in range(4):
                uriS = r'E:\Pictures\CAPTCHA\Cnki_Samples\Sample{0:0>4d}_{1}.jpg'.format(i+1, p)
                cv2.imwrite(uriS, picassembly[p])
        # image = binarization(image)
        # cv2.imwrite(urin, otsuimage)
        # exit(0)
        # binarization(image)
        # with Image.open(uri, 'r') as image:
        #     binarization(image)
        #     grey(image).save(uril, 'gif')
        #     Image.fromarray(np.uint8(binarization(grey(image)))).save(urib, 'gif')
        #     Image.fromarray(np.uint8(noise_eliminate(binarization(grey(image))))).save(urin, 'gif')
