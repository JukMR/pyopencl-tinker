import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl
from random import randint
import sys

def show_image(image, interval=0.1):
    plt.imshow(image, cmap='gray')
    # plt.pause(interval)
    plt.show()

def dump_temps(temps):
    for row in temps:
        for temp in row:
            print("{0:.2f} ".format(temp), end="")
        print("\n", end="")

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

def init_temps_sources(N = 32, k = 16, srcs = [(1, 1, 100)], room_temp = 10):
    temps = np.full((N + 2, N + 2), room_temp).astype(np.double)
    sources = np.zeros((N + 2, N + 2)).astype(np.bool8)

    # Set sources
    for x, y, t in srcs:
        sources[y + 1, x + 1] = True
        temps[y + 1, x + 1] = t

    # Set borders
    temps[0, :] = temps[-1, :] = temps[:, 0] = temps[:, -1] = room_temp
    sources[0, :] = sources[-1, :] = sources[:, 0] = sources[:, -1] = True

    return temps, sources

def main_np(N, k, srcs, room_temp):
    temps, sources = init_temps_sources(N, k, srcs, room_temp)
    show_image(temps)
    for i in range(k):
        temps = next_temps(temps, sources)
    show_image(temps)
    dump_temps(temps)

def main_gpu(N, k, srcs, room_temp):
    # Prepare OpenCL
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # Prepare Input
    temps, sources = init_temps_sources(N, k, srcs, room_temp)

    # Copy Input
    mf = cl.mem_flags
    temps_d = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=temps)
    sources_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sources)

    # Run
    kernel_file = open("heat_transfer.cl", "r")
    kernel_source = kernel_file.read()
    kernel_file.close()
    program = cl.Program(context, kernel_source).build()
    kernel = program.batch_cell_temp
    show_image(temps)
    kernel.set_args(temps_d, sources_d, np.int32(k))
    task = cl.enqueue_nd_range_kernel(queue, kernel, (N + 2, N + 2), None)
    copy = cl.enqueue_copy(queue, temps, temps_d)
    task_elapsed = (task.profile.end - task.profile.start) / (10**9)
    copy_elapsed = (copy.profile.end - copy.profile.start) / (10**9)
    show_image(temps)
    dump_temps(temps)
    print(f"Demoró en procesarse {task_elapsed} segundos")
    print(f"Demoró en copiarse {copy_elapsed} segundos")

def main_gpu_queue_in_order(N, k, srcs, room_temp):
    temps, sources = init_temps_sources(N, k, srcs, room_temp)
    # GPU Initialization
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
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
    show_image(temps)
    for i in range(k):
        if current_temps == 0:
            kernel.set_args(temps0_d, sources_d, temps1_d)
        elif current_temps == 1:
            kernel.set_args(temps1_d, sources_d, temps0_d)
        task = cl.enqueue_nd_range_kernel(queue, kernel, (N + 2, N + 2), (1, 1))
        if i == 0:
            initial_task = task
        if i == k - 1:
            end_task = task
        current_temps = (current_temps + 1) % 2
    copy = cl.enqueue_copy(
        queue, temps, temps1_d if current_temps == 0 else temps0_d
    )
    show_image(temps)
    dump_temps(temps)
    copy_elapsed = (copy.profile.end - copy.profile.start) / (10**9)
    task_elapsed = (end_task.profile.end - initial_task.profile.start) / (10**9)
    print(f"Demoró en procesarse {task_elapsed} segundos")
    print(f"Demoró en copiarse {copy_elapsed} segundos")

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
    N = 320
    k = 320
    srcs = [(i + randint(-10, 10), j + randint(-10, 10), 120) for i in range(0, N, 20) for j in range(0, N, 20)]
    room_temp = 10
    # main_np(N, k, srcs, room_temp)
    main_gpu(N, k, srcs, room_temp)
    main_gpu_queue_in_order(N, k, srcs, room_temp)
    # main_gpu_iterative()
    # main_gpu_batch_global_double_buffer()
    # main_gpu_queue_in_order()
