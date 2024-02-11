from sys import exit
import numpy as np
import matplotlib.pyplot as plt
#from skimage.metrics import structural_similarity as compare_ssim 
from skimage.metrics import structural_similarity as compare_ssim

import random

img_rows = img_cols = 28

def plotData(data):
    k = 0
    for i in range(10):
        k += 1
        plt.subplot(2, 5, k)
        plt.imshow(data[i])#, cmap = 'gray')
        plt.title(i, fontsize = 11)
        plt.axis('off')
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

def loadBinData(img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open('images_trn.bin', 'rb') as read_binary:
        x_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('labels_trn.bin', 'rb') as read_binary:
        y_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('images_tst.bin', 'rb') as read_binary:
        x_tst = np.fromfile(read_binary, dtype = np.uint8)
    with open('labels_tst.bin', 'rb') as read_binary:
        y_tst = np.fromfile(read_binary, dtype = np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_trn = np.array(x_trn, dtype = 'float32') / 255
    x_tst = np.array(x_tst, dtype = 'float32') / 255
    x_trn = x_trn.reshape(-1, img_rows, img_cols)
    x_tst = x_tst.reshape(-1, img_rows, img_cols)
    return x_trn, y_trn, x_tst, y_tst

x_trn, y_trn, x_tst, y_tst = loadBinData(img_rows, img_cols)

def get_average(images):
    return np.sum(images, axis = 0) / images.shape[0]

def generalization(x, y):
    result = []
    for i in range(10):
         result.append(get_average(x[y == i]))
    #plotData(result)
    return result
    
def one_plt(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def ToRGB(img):
    import numpy as np    
    data = img.copy()
    data[:, :, 1] *= 0
    data[:, :, 2] *= 0
    return data







     
# вид множества - test
num_class = 6


# find general
#images = x_tst[num_class == y_tst]
#general = np.sum(images, axis = 0) / images.shape[0]

images =[]
general = np.zeros((28,28))
for i in range(len(y_tst)):
    if y_tst[i] == num_class:
        general += x_tst[i]
        images.append(x_tst[i])
general = general / len(images)

#one_plt(general)

# find image
ind_min = 1
for i in range(len(images)):
    s = compare_ssim(images[i], general)
    if s < ind_min:
        ind_min = i
distant_image = x_tst[ind_min]
#one_plt(distant_image)
#print(distant_image.shape)

# color general
gnrl_clr = general.copy()
distant_image = distant_image.reshape(28, 28, 1)
gnrl_clr = gnrl_clr.reshape(28, 28, 1)
gnrl_clr = np.repeat(gnrl_clr, 3, axis = 2)
for i in range(28):
    for j in range(28):
        if distant_image[i, j, 0] > 0:
            gnrl_clr[i, j, 0] = 1
            gnrl_clr[i, j, 1] *= 0
            gnrl_clr[i, j, 2] *= 0

#one_plt(gnrl_clr)


fig, ax = plt.subplots(1, 3)
ax[0].imshow(general, cmap = 'gray')
ax[1].imshow(distant_image, cmap = 'gray')
ax[2].imshow(gnrl_clr)
plt.show()


    
    
