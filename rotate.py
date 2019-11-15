import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl

kernel_file = open("rotate.cl", "r")
kernel_source = kernel_file.read()
kernel_file.close()

def rotate_image_cpu(img, px = 0, py = 0, angle = pi / 4):
    source = np.asarray(img).copy()
    def rotated(x, y, px, py, t):
        v = np.array([x, y])
        p = np.array([px, py])
        rott = np.array([
            [cos(t), -sin(t)],
            [sin(t), cos(t)]
        ])
        return rott @ (v - p) + p

    target = np.zeros(source.shape, dtype=np.uint8)
    height, width, _ = source.shape
    for x in range(width):
        for y in range(height):
            ox, oy = np.rint(rotated(x, y, px, py, -angle)).astype(int)
            if 0 <= ox < width and 0 <= oy < height:
                target[y, x] = source[oy, ox]
    show_image(target)
    return target

def show_image(image, interval=0.01):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()

def show_anim(image, interval=0.01):
    plt.imshow(image, cmap='gray')
    plt.pause(interval)

def rotate_image_gpu(img, px = 0, py = 0, angle = pi / 4):
    # Prepare opencl
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)

    # Prepare Input
    source = np.asarray(img).copy()
    height, width, _ = source.shape
    px = np.int32(px)
    py = np.int32(py)
    angle = np.float32(angle)

    # Copy Input
    mf = cl.mem_flags
    source_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
    target_d = cl.Buffer(context, mf.WRITE_ONLY, size=source.nbytes)

    # Run
    program = cl.Program(context, kernel_source).build()
    kernel = program.rotate_from_pivot
    kernel.set_args(source_d, target_d, px, py, angle)
    cl.enqueue_nd_range_kernel(queue, kernel, (width, height), (1, 1))
    target = np.zeros(source.shape, dtype=np.uint8)
    cl.enqueue_copy(queue, target, target_d)
    show_image(target)
    return target

def continuous_rotate_image_gpu(img, px = 0, py = 0, angle_delta = pi / 36):
    # Prepare opencl
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)

    # Prepare Input
    source = np.asarray(img).copy()
    height, width, _ = source.shape
    px = np.int32(px)
    py = np.int32(py)
    angle_delta = np.float32(angle_delta)

    # Copy Input
    mf = cl.mem_flags
    source_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
    target_d = cl.Buffer(context, mf.WRITE_ONLY, size=source.nbytes)

    # Run
    program = cl.Program(context, kernel_source).build()
    kernel = program.rotate_from_pivot
    for i in range(160):
        kernel.set_args(source_d, target_d, px, py, np.float32(angle_delta * i))
        cl.enqueue_nd_range_kernel(queue, kernel, (width, height), (1, 1))
        target = np.zeros(source.shape, dtype=np.uint8)
        cl.enqueue_copy(queue, target, target_d)
        show_anim(target)
    return target

def print_info():
    platform = cl.get_platforms()[0]
    device = platform.get_devices(device_type=cl.device_type.GPU)[0]

    def print_info(obj, params_obj):
        obj_params = [
            p for p in dir(params_obj)
            if p == str.upper(p)
        ]

        for param in obj_params:
            p = getattr(params_obj, param)
            try:
                print(f"\t {param}: {obj.get_info(p)}\n")
            except:
                print(f"\t [{param}] not available\n")

    print("Platform Information\n\n")
    print_info(platform, cl.platform_info)

    print("\n\nDevice Information\n\n")
    print_info(device, cl.device_info)

if __name__ == '__main__':
    img = Image.open("imagen.png")
    width, height = img.size
    px = width / 2
    py = height / 2
    angle = pi / 4
    # rotate_image_cpu(img, px, py, angle)
    # rotate_image_gpu(img, px, py, angle)
    continuous_rotate_image_gpu(img, px, py, angle_delta=angle / 9)
    print_info()
