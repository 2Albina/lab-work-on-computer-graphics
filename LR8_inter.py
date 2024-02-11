import numpy as np
import matplotlib.pyplot as plt

def drawBLine(img, x0, y0, x1, y1, c0, c1):
    steep = abs(y1 - y0) > abs(x1 - x0) # Крутизна
    x_new = []
    y_new = []
    c_new = []
    if steep: # Обмен X, Y, если угол наклона отрезка более 45º
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1: # Приводим к базовой форме алгоритма, в которой x0 < x1
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        c0, c1 = c1, c0
    dx = x1 - x0
    dy = abs(y1 - y0)
    dx2 = 2 * dx
    dy2 = 2 * dy
    d = -dx
    y_step = 1 if y0 < y1 else -1 # Шаг по Y
    y = y0
    x = x0
    while x <= x1:
        if steep: # Помним о перестановках
            xp, yp = y, x
        else:
            xp, yp = x, y
        t = (x - x0) / (x1 - x0)
        img[yp, xp, 0] = int((1 - t) * c0[0] + t * c1[0]) / 255
        img[yp, xp, 1] = int((1 - t) * c0[1] + t * c1[1]) / 255
        img[yp, xp, 2] = int((1 - t) * c0[2] + t * c1[2]) / 255
        x_new.append(xp)
        y_new.append(yp)
        c_new.append(img[yp, xp])
        d = d + dy2
        if d > 0:
            y = y + y_step
            d = d - dx2
        x = x + 1
    return x_new, y_new, c_new

def show_x(x):
    plt.figure(figsize = (6, 5))
    plt.imshow(x)
    plt.axis('off')
    plt.show()

img = np.ones((150, 150, 3))

x0, y0, c0 = 10, 14, [255, 0, 0]
x1, y1, c1 = 145, 42, [0, 255, 0]
x2, y2, c2 = 135, 134, [0, 0, 255]
x3, y3, c3 = 27, 126, [255, 255, 0]

mas_x0, mas_y0, mas_c0 = drawBLine(img, x0, 150-y0, x1, 150-y1, c0, c1)
mas_x1, mas_y1, mas_c1 = drawBLine(img, x1, 150-y1, x2, 150-y2, c1, c2)
mas_x2, mas_y2, mas_c2 = drawBLine(img, x3, 150-y3, x2, 150-y2, c3, c2)
mas_x3, mas_y3, mas_c3 = drawBLine(img, x0, 150-y0, x3, 150-y3, c0, c3)

show_x(img)

# закрасим вернюю часть
for i in range(16, 24):
    ind2 = mas_y2.index(i)
    ind = mas_y1.index(i)
    drawBLine(img, mas_x2[ind2], i, mas_x1[ind], i, 255*mas_c2[ind2], 255*mas_c1[ind])

# закрасим среднюю часть
for i in range(24, 108):
    ind3 = mas_y3.index(i)
    ind = mas_y1.index(i)
    drawBLine(img, mas_x3[ind3], i, mas_x1[ind], i, 255*mas_c3[ind3], 255*mas_c1[ind])

# закрасим нижнюю часть
for i in range(108, 136):
    ind3 = mas_y3.index(i)
    ind = mas_y0.index(i)
    drawBLine(img, mas_x3[ind3], i, mas_x0[ind], i, 255*mas_c3[ind3], 255*mas_c0[ind])

show_x(img)
