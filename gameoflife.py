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

def show_image(image, interval=0.01):
    plt.imshow(image, cmap='gray')
    plt.pause(interval)

def main_np():
    k = 16
    N = 32
    frame = np.zeros((N, N)).astype(np.uint8)
    frame[16, :] = 255
    for i in range(16):
        show_image(frame)
        frame = next_frame(frame)

def main_gpu():
    k = 16
    N = 32
    frame = np.zeros((N, N), dtype=np.uint8)
    frame[16, :] = 255

    # GPU Initialization
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context) # In order
    mf = cl.mem_flags
    frame0_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=frame)
    frame1_d = cl.Buffer(context, mf.READ_WRITE, size=frame.nbytes)
    current_frame = 0

    kernel_file = open("gameoflife.cl", "r")
    kernel_source = kernel_file.read()
    kernel_file.close()
    program = cl.Program(context, kernel_source).build()
    kernel = program.next_cell_state
    for i in range(k):
        show_image(frame)
        if current_frame == 0:
            kernel.set_args(frame0_d, frame1_d)
        elif current_frame == 1:
            kernel.set_args(frame1_d, frame0_d)
        cl.enqueue_nd_range_kernel(queue, kernel, (N, N), (1, 1))
        cl.enqueue_copy(
            queue, frame, frame1_d if current_frame == 0 else frame0_d
        ) # by default is_blocking=True so it blocks here
        print("frame1_d" if current_frame == 0 else "frame0_d")
        # TODO assert
        current_frame = (current_frame + 1) % 2
if __name__ == '__main__':
    # main_np()
    main_gpu()
