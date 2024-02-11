import numpy as np
import matplotlib.pyplot as plt

def texture():
    W = H = 64 # Размер текстуры равен H * W
    img = np.zeros((H, W, 3), dtype = 'uint8')
    for i in range(H): # Генерация черно-белой карты текстуры
        for j in range(W):
            img[i, j, :] = (i & 16 ^ j & 16) * 255
    return img

def getY(x, p1, p2):
    return (x - p1[0])*(p2[1]-p1[1])/(p2[0]-p1[0]) + p1[1]

# Проверка: лежит ли текущая точка в многоугольнике
def check(x, y, points):
    if points[0][0] == points[1][0]:
        if x < points[0][0]:
            return False
    else:
        y1 = getY(x, points[0], points[1])
        if (y < y1):
            return False
    
    y2 = getY(x, points[1], points[2])
    if (y < y2):
        return False
    
    if points[2][0] == points[3][0]:
        if x > points[2][0]:
            return False
    else:
        y3 = getY(x, points[2], points[3])
        if (y > y3):
            return False
    
    y4 = getY(x, points[0], points[3])
    if (y > y4):
        return False

    return True
    
def overlay(img, repeat, points):
    P1, P2, P3, P4 = points[0], points[1], points[2], points[3]
    txtr = texture()
    min_x = min(P1[0], P2[0], P3[0], P4[0])
    max_x = max(P1[0], P2[0], P3[0], P4[0])
    min_y = min(P1[1], P2[1], P3[1], P4[1])
    max_y = max(P1[1], P2[1], P3[1], P4[1])
    # Если slant = True, то результат будет красивее
    slant = False #True
    if slant:
        a = P1
        e1 = P2 - P1
        e2 = P4 - P1
    else:
        a = np.array([min_x, max_y])
        e1 = np.array([min_x, min_y]) - a
        e2 = np.array([max_x, max_y]) - a
        
    delta = float(e1[0] * e2[1] - e1[1] * e2[0])
    for X in range(min_x, max_x):
        for Y in range(min_y, max_y):
            if (check(X, Y, points)):
                delta_u = float((X - a[0]) * e2[1] - (Y - a[1]) * e2[0])
                delta_v = float((Y - a[1]) * e1[0] - (X - a[0]) * e1[1])
                u, v = delta_u / delta, delta_v / delta
                p = a + u * e1 + v * e2
                x = int(63 * u * repeat) % 64
                y = int(63 * v * repeat) % 64
                img[Y, X] = txtr[y, x]

img = np.full((300, 300, 3), (100, 100, 100), dtype = 'uint8')

H = np.array([[23, 77], [150, 4], [277, 77], [150, 151]])
L = np.array([[23, 223], [23, 77], [150,151], [150,297]])
R = np.array([[150, 296], [150, 151], [277,77], [277,223]])

k = 0
for r in (0.5, 0.75, 1, 2, 3, 4):
    k += 1
    overlay(img, r, H)
    overlay(img, r, L)
    overlay(img, r, R)
    plt.subplot(2, 3, k)
    plt.title(label = r)
    plt.imshow(img)
plt.axis('off')

plt.show()
