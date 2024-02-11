import numpy as np
import matplotlib.pyplot as plt

def drawBLine(img, x0, y0, x1, y1, c0, c1):
    steep = abs(y1 - y0) > abs(x1 - x0) # Крутизна
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
        img[yp, xp, 0] = int((1 - t) * c0[0] + t * c1[0])
        img[yp, xp, 1] = int((1 - t) * c0[1] + t * c1[1])
        img[yp, xp, 2] = int((1 - t) * c0[2] + t * c1[2])
        d = d + dy2
        if d > 0:
            y = y + y_step
            d = d - dx2
        x = x + 1
    return img

def show_x(x):
    plt.figure(figsize = (5, 5))
    plt.imshow(x, cmap = 'gray')
    plt.axis('off')
    plt.show()

r = 30
b = 6
size = 2*r+b
img = np.zeros((size, size, 3))
n = 25

# координаты центра окр
x_c = int((2 * r + b) / 2)
y_c = int((2 * r + b) / 2)

x = np.random.randint(x_c-r,x_c+ r, n)
y = []
for i in range(len(x)):
    dig = [-1, 1]
    y_cur = int(np.sqrt(r**2 - (x[i]-x_c)**2)) * dig[np.random.randint(0, 2)] + y_c
    y.append(y_cur)
    img[y_cur][x[i]] = np.array([120, 120, 120])
y = np.array(y)

def find_nearst(x, y, x0, y0):
    ans = [x[0],y[0]]
    for i in range(len(x)):
        if ((x[i] - x0)**2 + (y[i] - y0)**2) < ((ans[0] - x0)**2 + (ans[1] - y0)**2):
            ans = [x[i], y[i]]
    print(ans)
    return ans

x0, y0 = find_nearst(x, y, 0, size)
c0 = [255, 0, 0] # красный
x1, y1 = find_nearst(x, y, 0, 0)
c1 = [0, 255, 0] # зеленый 
x2, y2 = find_nearst(x, y, size, 0)
c2 = [0, 0, 255] # голубой
x3, y3 = find_nearst(x, y, size, size)
c3 = [255, 255, 0] # желтый

def point(y, x, c):
    for k in range(-1, 2):
        for j in range(-1, 2):
            img[x + k][y + j] = c
    return img

point(x0, y0, c0)
point(x1, y1, c1)
point(x2, y2, c2)
point(x3, y3, c3)

drawBLine(img, x0, y0, x1, y1, c0, c1)
drawBLine(img, x0, y0, x2, y2, c0, c2)
drawBLine(img, x0, y0, x3, y3, c0, c3)
drawBLine(img, x1, y1, x2, y2, c1, c2)
drawBLine(img, x1, y1, x3, y3, c1, c3)
drawBLine(img, x2, y2, x3, y3, c2, c3)

img /= 255
show_x(img)
