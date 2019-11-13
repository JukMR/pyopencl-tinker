import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl

def next_temps(temps, sources):
    N = temps.shape[0]
    new_temps = np.empty_like(temps)
    for y in range(N):
        for x in range(N):
            if sources[y, x]: # It is a heat source
                new_temps[y, x] = temps[y, x]
            else:
                new_temps[y, x] = (
                    temps[y, x] + temps[y - 1, x] + temps[y, x + 1] +
                    temps[y + 1, x] + temps[y, x - 1]
                ) / 5
    return new_temps

def show_image(image, interval=0.1):
    plt.imshow(image, cmap='gray')
    # plt.pause(interval)
    plt.show()

def main_np():
    k = 320
    N = 30
    temps, sources = init_temps_sources(N, k)
    show_image(temps)
    for i in range(k):
        temps = next_temps(temps, sources)
    show_image(temps)

def init_temps_sources(N = 32, k = 16, room_temp = 100):
    temps = np.zeros((N + 2, N + 2)).astype(np.double)
    sources = np.zeros((N + 2, N + 2)).astype(np.bool8)

    # temps[15, :] = 15.23
    # temps[18, :] = 20.5123
    # temps[20, 10:20] = -80.5123

    sources[1, 1] = True
    temps[1, 1] = -100

    # sources[28, 28] = True
    # temps[28, 28] = 60

    # Set borders
    temps[0, :] = temps[-1, :] = temps[:, 0] = temps[:, -1] = room_temp
    sources[0, :] = sources[-1, :] = sources[:, 0] = sources[:, -1] = True

    return temps, sources

def main_gpu_iterative():
    k = 16
    N = 32
    temps, sources = init_temps_sources(N, k)
    # GPU Initialization
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context) # TODO: Add profiling options
    mf = cl.mem_flags
    temps0_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=temps)
    sources_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)
    temps1_d = cl.Buffer(context, mf.READ_WRITE, size=temps.nbytes)
    current_temps = 0

    kernel_file = open("heat_transfer.cl", "r")
    kernel_source = kernel_file.read()
    kernel_file.close()
    program = cl.Program(context, kernel_source).build()
    kernel = program.next_cell_temp
    for i in range(k):
        show_image(temps)
        if current_temps == 0:
            kernel.set_args(temps0_d, sources_d, temps1_d)
        elif current_temps == 1:
            kernel.set_args(temps1_d, sources_d, temps0_d)
        cl.enqueue_nd_range_kernel(queue, kernel, (N + 2, N + 2), (1, 1))
        cl.enqueue_copy(
            queue, temps, temps1_d if current_temps == 0 else temps0_d
        ) # TODO: .wait() clBarrier, clFinish, do we need synchronization here?
        print("temps1_d" if current_temps == 0 else "temps0_d")
        # TODO assert
        current_temps = (current_temps + 1) % 2

def main_gpu_batch_global():
    k = 320
    N = 32
    temps, sources = init_temps_sources(N, k)
    # GPU Initialization
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context) # TODO: Add profiling options
    mf = cl.mem_flags
    temps_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=temps)
    sources_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)

    kernel_file = open("heat_transfer.cl", "r")
    kernel_source = kernel_file.read()
    kernel_file.close()
    program = cl.Program(context, kernel_source).build()
    kernel = program.batch_cell_temp
    show_image(temps)
    kernel.set_args(temps_d, sources_d, np.int32(k))
    cl.enqueue_nd_range_kernel(queue, kernel, (N + 2, N + 2), None)
    cl.enqueue_copy(queue, temps, temps_d)
    show_image(temps)
    # TODO assert

def main_gpu_batch_global_double_buffer():
    k = 320
    N = 32
    temps, sources = init_temps_sources(N, k)
    # GPU Initialization
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context) # TODO: Add profiling options
    mf = cl.mem_flags
    temps0_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=temps)
    temps1_d = cl.Buffer(context, mf.READ_WRITE, size=temps.nbytes)
    sources_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)

    kernel_file = open("heat_transfer.cl", "r")
    kernel_source = kernel_file.read()
    kernel_file.close()
    program = cl.Program(context, kernel_source).build()
    kernel = program.batch_cell_temp_double_buffer
    show_image(temps)
    kernel.set_args(temps0_d, temps1_d, sources_d, np.int32(k))
    cl.enqueue_nd_range_kernel(queue, kernel, (N + 2, N + 2), (1, 1))
    cl.enqueue_copy(queue, temps, temps0_d if k % 2 == 0 else temps1_d)
    show_image(temps)
    # TODO assert


if __name__ == '__main__':
    # main_np()
    # main_gpu_iterative()
    main_gpu_batch_global()
    # main_gpu_batch_global_double_buffer()
