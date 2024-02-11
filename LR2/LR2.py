import random
import numpy as np
import matplotlib.pyplot as plt

def create_image(fun_n):
    size = (64, 64)
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

    return image

def print_image(fun_n, image):
    if fun_n == 0:
        nm = "окружность"
    elif fun_n == 1:
        nm = "полуокружность"

    plt.figure(nm)
    plt.imshow(image, cmap='gray')
    plt.show()

def prepareData(n_f, fn, fn2):
    file_data = open(fn, 'wb')
    file_label = open(fn2, 'wb')

    n_class = 2 # количество классов

    for fun_n in range(n_class):
        for i in range(n_f):
            image = create_image(fun_n)

            file_data.write(image)
            file_label.write(np.uint8(fun_n))
    
    file_data.close()
    file_label.close()

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
fn = 'data.bin'
fn2 = 'label.bin'
n_f = 8 # количество фигур в каждом классе
prepareData(n_f, fn, fn2)
print("1 part is done")

# чтение файла
data, labels = load_data(fn, fn2)
n = 2 * n_f # всего фигур
data = data.reshape(n, 64, 64)
print("2 part is done")

# вывод изображений
plotData(data, labels)

print(type(data[0][0][0]))
print("3 part is done")














