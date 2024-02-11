import pyglet
from pyglet import app, gl, graphics
from pyglet.window import Window, key
import numpy as np

width = 400
height = 300
wx, wy = 100, 50
col = True
shade_model = gl.GL_FLAT
hx = hy = 0
cx = 60
cy = 32.5
ang = 0

window = Window(visible = True, width = width, height = height, resizable = True, caption = 'ЛР 10')

def draw():
    #система координат
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-wx, wx, -wy, wy, -3, 3)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glLineWidth(1)
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0, 0, 0)
    gl.glVertex3f(-100, 0, 0)
    gl.glVertex3f(100, 0, 0)
    gl.glEnd()
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(0,0,0)
    gl.glVertex3f(0, -50, 0)
    gl.glVertex3f(0, 50, 0)
    gl.glEnd()

    #синий прямоугольник в 1 квадранте
    gl.glPushMatrix()
    gl.glTranslatef(hx, hy, 0)
    gl.glTranslatef(cx, cy, 0)
    gl.glRotatef(ang,0,0,-2)
    gl.glTranslatef(-cx, -cy, 0) 
    gl.glBegin(gl.GL_QUADS)
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(45, 27.5, -2.)
    gl.glVertex3f(75., 27.5, -2.)
    gl.glVertex3f(75, 37.5, -2.)
    gl.glVertex3f(45, 37.5, -2.)
    gl.glEnd()
    gl.glPopMatrix()

    #красный прямоугольник в 3 квадранте
    gl.glPushMatrix()
    gl.glTranslatef(-hx, -hy, 0)
    gl.glTranslatef(-cx, -cy, 0)
    gl.glRotatef(-ang,0,0,2)
    gl.glTranslatef(cx, cy, 0)
    gl.glBegin(gl.GL_QUADS)
    gl.glColor3f(1, 0, 0)
    gl.glVertex3f(-75, -37.5, 2)
    gl.glVertex3f(-45., -37.5, 2)
    gl.glVertex3f(-45, -27.5, 2)
    gl.glVertex3f(-75, -27.5, 2)
    gl.glEnd()
    gl.glPopMatrix()

    #треугольник
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor3d(1, 0, 0)
    gl.glVertex3f(0, 12.5, 1)
    gl.glColor3d(0, 0, 1)
    gl.glVertex3f(-12.5, -10.85, 1)
    gl.glColor3d(1, 1, 0)
    gl.glVertex3f(12.5, -10.859, 1)
    gl.glEnd()


gl.glClearColor(0.7, 0.7, 0.7, -3)
gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
gl.glEnable(gl.GL_DEPTH_TEST)
@window.event

def on_draw():
    window.clear()
    gl.glShadeModel(shade_model)
    draw()
@window.event

def on_key_press(symbol, modifiers):
    global x, y, col, shade_model, hx, hy, ang
    if symbol == key._1:
        hx -= 5
        hy -= 2.7
        ang -= 15
        if col:
            shade_model = gl.GL_SMOOTH
            col = False
        else:
            shade_model = gl.GL_FLAT
            col = True
    elif symbol == key._2:
        hx += 5
        hy += 2.7
        ang += 15
    elif symbol == key._3:
        shade_model = gl.GL_FLAT
        col = True
        hx = hy = 0
        ang = 0
            
app.run()
