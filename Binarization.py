import cv2
import numpy as np
from matplotlib import pyplot as plt


class Binarization:

    def OTSU(self, pic):
        im = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        plt.subplot(131), plt.imshow(im, "gray")
        plt.title("source image"), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.hist(im.ravel(), 256)
        plt.title("Histogram"), plt.xticks([]), plt.yticks([])
        ret1, th1 = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)  # 方法选择为THRESH_OTSU
        plt.subplot(133), plt.imshow(th1, "gray")
        plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
        plt.show()
        return th1
    OTSU = staticmethod(OTSU)

    def standard(self,pic):
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
    standard = staticmethod(standard)
