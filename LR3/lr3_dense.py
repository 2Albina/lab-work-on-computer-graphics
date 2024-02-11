from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#
img_rows = img_cols = 64
show_k = not True
pred = True
num_classes = 6
epochs = 20
fn_model = 'lr3_dense_serrah.h5'
#
def show_x(x, y, img_rows, img_cols, N):
    n = int(np.sqrt(N))
    for i, j  in enumerate(np.random.randint(len(x), size = n*n)):
        plt.subplot(n, n, i + 1)
        img = x[j].reshape(img_rows, img_cols)
        plt.imshow(img, cmap = 'gray')
        plt.title(np.argmax(y[j]))
        plt.axis('off')
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

# Вывод графиков
def one_plot(n, y_lb, loss_acc, val_loss_acc):
    plt.subplot(1, 2, n)
    if n == 1:
        lb, lb2 = 'loss', 'val_loss'
        yMin = 0
        yMax = 1.05 * max(max(loss_acc), max(val_loss_acc))
    else:
        lb, lb2 = 'acc', 'val_acc'
        yMin = min(min(loss_acc), min(val_loss_acc))
        yMax = 1.0
    plt.plot(loss_acc, color = 'r', label = lb, linestyle = '--')
    plt.plot(val_loss_acc, color = 'g', label = lb2)
    plt.ylabel(y_lb)
    plt.xlabel('Эпоха')
    plt.ylim([0.95 * yMin, yMax])
    plt.legend()
#
def loadBinData(img_rows, img_cols):
    print('Загрузка данных из двоичных файлов...')
    with open('data_train.bin', 'rb') as read_binary:
        x_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('label_train.bin', 'rb') as read_binary:
        y_trn = np.fromfile(read_binary, dtype = np.uint8)
    with open('data_check.bin', 'rb') as read_binary:
        x_tst = np.fromfile(read_binary, dtype = np.uint8)
    with open('label_check.bin', 'rb') as read_binary:
        y_tst = np.fromfile(read_binary, dtype = np.uint8)
    # Преобразование целочисленных данных в float32 и нормализация; данные лежат в диапазоне [0.0, 1.0]
    x_trn = np.array(x_trn, dtype = 'float32') / 255
    x_tst = np.array(x_tst, dtype = 'float32') / 255
    x_trn = x_trn.reshape(-1, img_rows * img_cols)
    x_tst = x_tst.reshape(-1, img_rows * img_cols)
    print('Преобразуем массивы меток в one-hot представление')
    y_trn = tf.keras.utils.to_categorical(y_trn, num_classes)
    y_tst = tf.keras.utils.to_categorical(y_tst, num_classes)
    return x_trn, y_trn, x_tst, y_tst
#
# Загрузка обучающего и проверочного множества из бинарных файлов
# Загружаются изображения и их метки
x_trn, y_trn, x_tst, y_tst = loadBinData(img_rows, img_cols)
if show_k:
    show_x(x_tst, y_tst, img_rows, img_cols, 16)
    exit()
if pred:
    from sklearn.metrics import classification_report
    from keras.models import load_model
    model = load_model(fn_model)
    # Прогноз
    y_pred = model.predict(x_tst)
    # print(y_pred[0])
    # print(y_tst[0])
    # [6.8e-6 1.5e-10 7.6e-6 1.5e-3 7.0e-9 6.2e-5 2.2e-11 9.9e-1 3.0e-7 5.9e-6]
    # [0.     0.      0.     0.     0.     0.     0.      1.     0.     0.]
    # Заносим в массив predicted_classes метки классов, предсказанных моделью НС
    predicted_classes = np.array([np.argmax(m) for m in y_pred])
    true_classes = np.array([np.argmax(m) for m in y_tst])
    n_tst = len(y_tst)
    # Число верно классифицированных изображений
    true_classified = np.sum(predicted_classes == true_classes)
    # Число ошибочно классифицированных изображений
    false_classified = n_tst - true_classified
    acc = round(100 * true_classified / n_tst, 2)
    print('Точность: {}{}'.format( acc, '%'))
    print('Неверно классифицированно:', false_classified)
    print('Точность по классам:')
    for cls in range(num_classes):
       x_cls = [x for x, y in zip(x_tst, y_tst) if y[cls] == 1]
       x_cls = np.array(x_cls)
       y_pred_cls = model.predict(x_cls)
       predicted_cls = np.array([np.argmax(m) for m in y_pred_cls])
       n_cls = len(y_pred_cls)
       true_cls = [cls]*n_cls
       t_cls = np.sum(predicted_cls == true_cls)
       f_cls = n_cls - t_cls
       acc_cls = round(100 * t_cls / n_cls, 2)
       print(cls, ':', acc_cls)
    print(classification_report(true_classes, predicted_classes, digits = 4))
    exit()
#

import keras # Создание модели нейронной сети
from keras.models import Model
from keras.layers import Input, Dense, Dropout
inp = Input(img_rows * img_cols) # Входной слой
x = Dropout(0.3)(inp)
x = Dense(units = 32, activation = 'relu')(x)
output = Dense(num_classes, activation = 'softmax')(x)
model = Model(inputs = inp, outputs = output)
model.summary()
model.compile(optimizer = 'Adam', loss = 'mse', metrics = ['accuracy'])
##from tensorflow.keras.utils import plot_model
##plot_model(model, to_file = 'mnist_dense.png')
#
# Обучение нейронной сети
history = model.fit(x_trn, y_trn, batch_size = 128, epochs = epochs,
                    verbose = 2, validation_data = (x_tst, y_tst))
print('Модель сохранена в файле', fn_model)
model.save(fn_model)
# Запись истории обучения в текстовые файлы
history = history.history
##for itm in history.items(): print(itm)
# Вывод графиков обучения
plt.figure(figsize = (9, 4))
plt.subplots_adjust(wspace = 0.5)
one_plot(1, 'Потери', history['loss'], history['val_loss'])
one_plot(2, 'Точность', history['accuracy'], history['val_accuracy'])
plt.suptitle('Потери и точность')
plt.show()


