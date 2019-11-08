import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("imagen.png")
source = np.asarray(img).copy()
target = np.zeros(source.shape, dtype=np.uint8)
target[:, :, 3] = 255
fig, ax = plt.subplots()

def rotated(x, y, px, py, t):
    v = np.array([x, y])
    p = np.array([px, py])
    rott = np.array([
        [cos(t), -sin(t)],
        [sin(t), cos(t)]
    ])
    return rott @ (v - p) + p

height = source.shape[0]
width = source.shape[1]

angle = pi / 4
for x in range(width):
    for y in range(height):
        ox, oy = np.rint(rotated(x, y, width / 2, height / 2, -angle)).astype(int)
        if 0 <= ox < width and 0 <= oy < height:
            target[y, x] = source[oy, ox]

ax.imshow(target, cmap='gray')
plt.axis('off')
plt.show()
