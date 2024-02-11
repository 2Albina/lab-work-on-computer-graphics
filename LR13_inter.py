import numpy as np
import matplotlib.pyplot as plt
import random

def show_img(img):
    img = np.transpose(img, (1,0,2))
    img = np.flip(img, 0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def add_back(img):
    for i in range(300):
        for j in range(300):
            if np.array_equal(img[i][j], np.array([0, 0, 0])):
                 img[i, j] = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

def drawBLine(img, p1, p2, c0, c1):
    x0, y0 = p1[0], p1[1]
    x1, y1 = p2[0], p2[1]
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
        img[xp, yp, 0] = int((1 - t) * c0[0] + t * c1[0]) 
        img[xp, yp, 1] = int((1 - t) * c0[1] + t * c1[1]) 
        img[xp, yp, 2] = int((1 - t) * c0[2] + t * c1[2]) 
        d = d + dy2
        if d > 0:
            y = y + y_step
            d = d - dx2
        x = x + 1
    return img

def full_inter(img, points, colors):
    for i in range(len(points)):
        img = drawBLine(img, points[i-1], points[i], colors[i-1], colors[i])
    min_x = min(np.array(points[:, 0]))
    max_x = max(np.array(points[:, 0]))
    min_y = min(np.array(points[:, 1]))
    max_y = max(np.array(points[:, 1]))
    flag = True
    for j in range(min_x, max_x+1):
        arr = []
        for i in range(min_y, max_y+1):
            if not(np.array_equal(img[i][j], np.array([0, 0, 0]))) and (
                np.array_equal(img[i-1][j], np.array([0, 0, 0]))):
                arr.append(i)
        len_arr = len(arr)
        if len_arr > 1:
            print(arr)
            if (len_arr == 2) and not flag:
                drawBLine(img, (arr[0], j), (arr[-1], j),  img[arr[0], j], img[arr[-1], j])
            if len_arr == 3:
                drawBLine(img, (arr[0], j), (arr[-1], j),  img[arr[0], j], img[arr[-1], j])
                flag = False
            if flag and len_arr == 4:
                drawBLine(img, (arr[0], j), (arr[1], j),  img[arr[0], j], img[arr[1], j])
                drawBLine(img, (arr[2], j), (arr[3], j),  img[arr[2], j], img[arr[3], j])

img1 = np.zeros((300, 300, 3), dtype = 'uint8')

plg = np.array([[150,85,0],[60,20,0],[100,120,0],[20,160,0],[120,175,0],
       [150,280,0],[180,175,0],[280,160,0],[200,120,0],[240,20,0]])
c2,c1,c3 = np.array([[250,0,0],[250,250,0],[250,175,0]])
cirs = np.array([c2,c1,c2,c3,c2,c1,c2,c3,c2,c1])

full_inter(img1, plg, cirs)
add_back(img1)
show_img(img1)
