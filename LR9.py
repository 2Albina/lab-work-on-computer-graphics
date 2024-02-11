# ЛР 9 по КГ
# Выполнила Белова Альбина (группа А-05-20)

import numpy as np
import matplotlib.pyplot as plt

def drawBLine(img, z_buffer, p1, p2, c, p3):
    x0, y0 = p1[0], p1[1]
    x1, y1 = p2[0], p2[1]
    steep = abs(y1 - y0) > abs(x1 - x0) # Крутизна
    line = []
    if steep: # Обмен X, Y, если угол наклона отрезка более 45º
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1: # Приводим к базовой форме алгоритма, в которой x0 < x1
        x0, x1 = x1, x0
        y0, y1 = y1, y0
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
        zp = find_z(xp,yp,p1,p2,p3)
        if (zp > z_buffer[xp, yp]):
            img[xp, yp] = c
            z_buffer[xp, yp] = zp
        line.append([xp, yp])
        d = d + dy2
        if d > 0:
            y = y + y_step
            d = d - dx2
        x = x + 1
    return np.array(line)

def find_z(x,y,p1,p2,p3):
    x0,y0,z0 = p1
    A,B,C = np.cross(np.array(p2) - np.array(p1),np.array(p3) - np.array(p1))
    return (A*(x0-x)+B*(y0-y))/C + z0

def draw_triangle(p1, p2, p3, color, img, z_buffer):
    p = list([p1, p2, p3])
    print(p)
    p.sort(key = lambda x: x[1])
    line1 = np.transpose(drawBLine(img, z_buffer, p[2], p[0], color, p[1]), (1, 0))
    line2 = np.transpose(drawBLine(img, z_buffer, p[0], p[1], color, p[2]), (1, 0))
    line3 = np.transpose(drawBLine(img, z_buffer, p[1], p[2], color, p[0]), (1, 0))
    print(p)
    part = np.hstack([line2, line3])
    print(line1, line2, line3, part, sep="\n\n", end="\n dfg\n")
    for j in range(p[0][1]+1, p[2][1]):
        ind = list(line1[1]).index(j)
        ind2 = list(part[1]).index(j)
        begin = min(line1[0, ind], part[0, ind2])
        end = max(line1[0, ind], part[0, ind2])
        for i in range(begin+1, end):
            z = find_z(i,j,p1,p2,p3)
            if (z > z_buffer[i][j]):
                img[i][j] = color 
                z_buffer[i, j] = z            
    
def show_x(x):
    plt.figure(figsize = (6, 5))
    plt.imshow(x/255)
    plt.axis('off')
    plt.show()

wx, wy = 150, 80

img = np.zeros((wx, wy, 3))
z_buffer = np.array([[-3 for j in range(wy)] for i in range(wx)])

# координаты вершин синего треугольника
b = ([[60, wy-70, -1], [140, wy-50, 0], [20, wy-10, 0]])
cb = [0, 0, 255]

# координаты вершин красного треугольника
r = [[120, wy-70, 2], [100, wy-10, 0], [10, wy-40, -1]]
cr = [255, 0, 0]

draw_triangle(b[0], b[1], b[2], cb, img, z_buffer)
draw_triangle(r[0], r[1], r[2], cr, img, z_buffer)

show_x(np.transpose(img,(1,0,2)))
