import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rot_img(ang, img_array):
    img_array = np.array(img_array, dtype = 'uint8')
    img = Image.fromarray(img_array, 'L') # Формируем изображение по массиву img_array
    # Поворот изображения на угол ang против часовой стрелки
    img = img.rotate(ang)
    # Переводим изображение в массив
    ix = img.size[0]
    iy = img.size[1]
    img_array_rot = np.array(img.getdata(), dtype = 'uint8').reshape(iy, ix)
    return img_array_rot

def drawline(a1,a2,image):
    k = 10000
    for i in range(k):
        x_ = a1[0]+i/k*(a2[0]-a1[0])+random.uniform(-0.5,0.5)
        y_= a1[1]+i/k*(a2[1]-a1[1])+random.uniform(-0.5,0.5)
        image[int(x_),int(y_)] = random.uniform(75, 255)

def create_image(fun_n):
    size = (64, 64)
    if fun_n == 0 or fun_n == 1:
        point = (random.uniform(20.0, 40.0),random.uniform(20.0, 40.0))
        r = random.uniform (5.0, min(64 - point[0], 64 - point[1], point[0], point[1]) - 2)
        a1 = random.uniform (0, 2*np.pi)
        
        if fun_n == 0:
            a2 = a1 + 2*np.pi
        elif fun_n == 1:
            a2 = a1 + np.pi
        
        k = random.randint(200, 400)
        
        a = np.linspace(a1, a2, k)
        image = np.zeros(size, dtype=np.uint8)
        for i in a:
            cur_x = point[0] + r*np.cos(i) + random.uniform(-1, 1)
            cur_y = point[1] + r*np.sin(i) + random.uniform(-1, 1)
            image[int(cur_x), int(cur_y)] = random.randint(75, 255)
    elif fun_n == 2 or fun_n == 3:
        if fun_n == 2:
            x_noise = np.random.uniform(-1, 1)
            y_noise = np.random.uniform(-4, 4)
            r=random.uniform(0.4,0.75)
            #a = random.uniform(0.0, 40.0)
            nm = 'Эвольвента'
            t = np.linspace(0 , 30, 1000)
            x = (r * (np.cos(t) + t * np.sin(t))) + np.random.uniform(-1, 1)
            y = (r * (np.sin(t) - t * np.cos(t))) + np.random.uniform(-4, 4)
        elif fun_n == 3:
            x_noise = np.random.uniform(-0.3, 0.3)
            y_noise = np.random.uniform(-0.3, 0.3)
            nm = 'Гиперболическая спираль'
            r = random.uniform(20,60)
            z = random.randint(0,1)
            if z == 0:
                t = np.linspace(-30, -1, 1000)
            elif z == 1:
                t = np.linspace(1, 30, 1000)
            x = r * np.cos(t) / t + np.random.uniform(-0.3, 0.3)
            y = r * np.sin(t) / t + np.random.uniform(-0.3, 0.3)
        x_min = int(min(x))
        x_max = int(max(x))
        y_min = int(min(y))
        y_max = int(max(y))
            # Нужно уместить в 64*64
        dx = int((64 - (x_max - x_min)) / 2) # Половина свободного пространства по x
        dy = int((64 - (y_max - y_min)) / 2) # Половина свободного пространства по y
        shift_x = abs(x_min) + dx # Сдвиг по x
        shift_y = abs(y_min) + dy # Сдвиг по y
        w = h = 64 # Ширина и высота рисунка
        image = np.zeros((w, h), dtype = np.uint8)
        clr_mim, clr_max = 75, 255 # Диапазон оттенков серого цвета
        for x, y in zip(x, y):
            ix = (int(x) + shift_x)%64
            iy = (int(y) + shift_y)%64
            clr = np.random.randint(clr_mim, clr_max)
            image[iy, ix] = clr
        angle = random.uniform(0.0, 180.0)
        image = rot_img(angle, image)
    elif fun_n == 4 or fun_n == 5:
        a = (random.uniform(16.0,32.0),random.uniform(16.0,32.0))
        b = (random.uniform(16.0,32.0),random.uniform(32.0,48.0))
        c = (random.uniform(32.0,48.0),random.uniform(16.0,32.0))
        dx = b[0]-a[0]
        dy = b[1]-a[1]
        d = (c[0]+dx,c[1]+dy)
        image = np.zeros(size, dtype = np.uint8)
        if fun_n == 4:
              points = [(a,b),(b,d),(a,c),(c,d)]
              points.pop(random.randint(0,3))
        elif fun_n == 5:
             points = [(a,b),(b,d),(a,c),(c,d)]

        for p in points:
            drawline(p[0], p[1], image)
    
    return image

def print_image(fun_n, image):
    if fun_n == 0:
        nm = "окружность"
    elif fun_n == 1:
        nm = "полуокружность"

    plt.figure(nm)
    plt.imshow(image, cmap='gray')
    plt.show()

def prepareData(fn_t, fn2_t, fn_ch, fn2_ch):
    data_train = open(fn_t, 'wb')
    label_train = open(fn2_t, 'wb')
    data_check = open(fn_ch, 'wb')
    label_check = open(fn2_ch, 'wb')

    n_class = 6 # количество классов
    # 
    for fun_n in range(n_class):
        for i in range(600):
            image = create_image(fun_n)

            data_train.write(image)
            label_train.write(np.uint8(fun_n))

    #
    for fun_n in range(n_class):
        for i in range(150):
            image = create_image(fun_n)

            data_check.write(image)
            label_check.write(np.uint8(fun_n))

    data_train.close()
    label_train.close()

def load_data(fn, fn2):
    with open(fn, 'rb') as read_binary:
        data = np.fromfile(read_binary, dtype = np.uint8)
    with open(fn2, 'rb') as read_binary:
        labels = np.fromfile(read_binary, dtype = np.uint8)
    return data, labels

def plotData(data, labels):
    k = 0
    n = min(16, len(labels))
    for i in range(n):
        k += 1
        plt.subplot(4, 4, k)
        plt.imshow(data[i], cmap = 'gray')
        plt.title(labels[i], fontsize = 11)
        plt.axis('off')
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

# создание нескольких изображений и сохранение в файл
fn_t = 'data_train.bin'
fn2_t = 'label_train.bin'
fn_ch = 'data_check.bin'
fn2_ch = 'label_check.bin'
prepareData(fn_t, fn2_t, fn_ch, fn2_ch)
print("1 part is done")













