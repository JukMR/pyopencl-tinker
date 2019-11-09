import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl

def next_frame(frame):
    state = frame != 0
    N = state.shape[0]
    new_frame = np.empty_like(frame)
    def neighbours(x, y):
        count = 0
        xl, xr, yu, yd = x - 1, x + 1, y - 1, y + 1
        is_valid = lambda x, y: 0 <= x < N and 0 <= y < N
        count += state[yu, xl] if is_valid(yu, xl) else 0
        count += state[yu, x] if is_valid(yu, x) else 0
        count += state[yu, xr] if is_valid(yu, xr) else 0
        count += state[y, xl] if is_valid(y, xl) else 0
        count += state[y, xr] if is_valid(y, xr) else 0
        count += state[yd, xl] if is_valid(yd, xl) else 0
        count += state[yd, x] if is_valid(yd, x) else 0
        count += state[yd, xr] if is_valid(yd, xr) else 0
        return count

    for y in range(N):
        for x in range(N):
            if state[y, x]: # Alive cell
                new_frame[y, x] = 255 if 2 <= neighbours(x, y) <= 3 else 0
            else: # Dead cell
                new_frame[y, x] = 255 if neighbours(x, y) == 3 else 0

    return new_frame

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def main_np():
    N = 32
    frame = np.zeros((N, N)).astype(np.uint8)
    frame[16, :] = 255
    for i in range(16):
        show_image(frame)
        frame = next_frame(frame)

if __name__ == '__main__':
    main_np()
