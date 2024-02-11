from sys import exit
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from keras import layers
from matplotlib import pyplot as plt
import time
import numpy as np
fn_model = 'lr5_model_100_RGB_v.h5'
epochs = 100
img_rows = img_cols = 64

def loadBinData(img_rows, img_cols):
    #print('Загрузка данных из двоичных файлов...')
    with open('data_train.bin', 'rb') as read_binary:
        x_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('label_train.bin', 'rb') as read_binary:
        y_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('data_check.bin', 'rb') as read_binary:
        x_tst = np.fromfile(read_binary, dtype = np.uint8)
    with open('label_check.bin', 'rb') as read_binary:
        y_tst = np.fromfile(read_binary, dtype = np.uint8)
    return x_trn, y_trn, x_tst, y_tst

# вывод 16 изображений. # imgs.shape = (16, 64*64*3)
def some_plts(imgs):
    fig, axs = plt.subplots(4, 4)
    k = -1
    for i in range(4):
        for j in range(4):
            k += 1
            img = imgs[k].reshape(64, 64, 3)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

# вывод 1 изображения. # imgs.shape = (16, 64*64*3)
def one_plt(img):
    img = img.reshape(64, 64, 3)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Преобразование изображения в цвет в зависимости от метки
def ToRGB(data, lbl):
    import numpy as np
    
    if lbl == 1: # красный / полуокружность
        data[:, :, 1] *= 0
        data[:, :, 2] *= 0
    elif lbl == 2: # зеленый / Эвольвента
        data[:, :, 0] *= 0
        data[:, :, 2] *= 0
    elif lbl == 3: # желтый / Гиперболическая спираль
        data[:, :, 2] *= 0
    elif lbl == 4: # фиолетовый / Параллелограмм 
        data[:, :, 1] *= 0
    elif lbl == 5: # бирюзовый / Параллелограмм без одной стороны
        data[:, :, 0] *= 0
    return data

# Загрузка данных
x_trn, y_trn, x_tst, y_tst = loadBinData(img_rows, img_cols)
# Преобразуем в три канала массив x_trn
x_trn = x_trn / 255
x_trn = x_trn.reshape(-1, 64,64,1)
x_trn = np.repeat(x_trn, 3, axis = 3)
x_trn = x_trn.reshape(len(y_trn), 3*4096)
# Преобразуем в три канала массив x_tst
len_tst = len(y_tst)
x_tst = x_tst / 255
x_tst = x_tst.reshape(-1, 64,64,1)
x_tst = np.repeat(x_tst, 3, axis = 3)
#x_tst = x_tst.reshape(len_tst, 3*4096)
print("1 part is done")

# Преобразование изображений из x_trn в трехканальные
data = x_trn.copy()
data = data.reshape(-1, 64, 64, 3)
for i in range(len(y_trn)):
    data[i] = ToRGB(data[i], y_trn[i])
data = data.reshape(len(y_trn), 3*4096)
print("2 part is done")

# Если модель существует, сразу получим изображение
ExistModel = True
if ExistModel:
    from keras.models import load_model
    model = load_model(fn_model)
    
    one =  False
    if one:
        # Случайное изображение элемента тестового множества
        i = np.random.randint(len(y_tst))
        img = x_tst[i].reshape(-1, 3*4096)
        one_plt(img)
        img_predicted = model.predict(img)
        one_plt(img_predicted)
        exit()
    
    # 16 случайных изображений из тестового множества
    arr_idx = np.random.randint(0, len_tst, 16) # массив из 16 номеров изображений в x_tst
    imgs_for_test = x_tst[arr_idx].reshape(16, 4096*3) # массив из 16 изображений из x_tst
    some_plts(imgs_for_test)
    imgs_pedicted = model.predict(imgs_for_test)
    some_plts(imgs_pedicted) # imgs_pedicted.shape = (16, 12288)
    exit()

add_train = not True
if add_train:
    
    from keras.models import load_model
    model = load_model(fn_model)
    for epoch in range(4):
        print('epoch:', epoch + 97)
        model.fit(x = x_trn, y = data, verbose = 2)
    
    print('Модель сохранена в файле', 'lr5_model_100_RGB.h5')
    model.save('lr5_model_100_RGB.h5')
    exit()

def one_part(units, x):
    x = Dense(units)(x)
    x = LeakyReLU()(x)
    return Dropout(0.25)(x)
#
t1 = time.time()
latent_size = 32 # Размер латентного пространста
inp = Input(shape = 64*64*3)
x = one_part(512, inp)
x = one_part(256, x)
x = one_part(128, x)
x = one_part(64, x)
x = Dense(latent_size)(x)
encoded = LeakyReLU()(x)
x = one_part(64, encoded)
x = one_part(128, x)
x = one_part(256, x)
x = one_part(512, x)
decoded = Dense(64*64*3, activation = 'sigmoid')(x)
model = Model(inputs = inp, outputs = decoded)
model.compile('adam', loss = 'binary_crossentropy') # nadam
model.summary()
print("3 part is done")

plt_epoch = not False
for epoch in range(epochs):
    print('epoch:', epoch + 1)
    model.fit(x = x_trn, y = data, verbose = 2) # data сформ по x_trn
    if plt_epoch and epoch > 0 and epoch % 5 == 0:
        model_name = 'lr5_model_RGB_' + str(epoch + 1) + '.h5'
        model.save(model_name)

t2 = time.time()
print("time: ", t2-t1)
print("4 part is done")
if not plt_epoch:
    print('Модель сохранена в файле', fn_model)
    model.save(fn_model)
