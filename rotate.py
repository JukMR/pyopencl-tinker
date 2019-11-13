import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl

def main_rotate_image_cpu(source, img, px, py, angle = pi / 4):
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

    return target

def show_image(image, interval=0.01):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()

def main_rotate_image_gpu():
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)

    img = Image.open("imagen.png")
    source = np.asarray(img).copy()
    height, width, _ = source.shape

    mf = cl.mem_flags
    source_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
    target_d = cl.Buffer(context, mf.WRITE_ONLY, size=source.nbytes)

    kernel_source = """
        __kernel void rotate_from_pivot(
            __global const uchar *source,
            __global uchar *target,
            int px,
            int py,
            float t)
        {
            // TODO: would be nice to write the kernel in its own .cl file
            // TODO: Use opencl vectors and matrices like this:
            // uchar2 v = (float2)(x, y);

            t = -t;
            int x = get_global_id(0);
            int y = get_global_id(1);
            int width = get_global_size(0);
            int height = get_global_size(1);
            float cost = cos(t);
            float sint = sin(t);
            int ox = round(cost * (x - px) - sint * (y - py) + px);
            int oy = round(sint * (x - px) + cost * (y - py) + py);
            if (0 <= ox && ox < width && 0 <= oy && oy < height) {
                target[y * width * 4  + x * 4 + 0] = source[oy * width * 4  + ox * 4 + 0];
                target[y * width * 4  + x * 4 + 1] = source[oy * width * 4  + ox * 4 + 1];
                target[y * width * 4  + x * 4 + 2] = source[oy * width * 4  + ox * 4 + 2];
                target[y * width * 4  + x * 4 + 3] = 255;
            }
    }
    """

    program = cl.Program(context, kernel_source).build()
    kernel = program.rotate_from_pivot
    px = np.int32(width / 2)
    py = np.int32(height / 2)
    angle = np.float32(pi / 4)
    kernel.set_args(source_d, target_d, px, py, angle)
    cl.enqueue_nd_range_kernel(queue, kernel, (width, height), (1, 1))
    target = np.zeros(source.shape, dtype=np.uint8)
    cl.enqueue_copy(queue, target, target_d)
    show_image(target)
    # TODO: The assert does not pass
    # assert np.array_equal(target, target_np), "Not correctly rotated"

def main_print_info():
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
    # main_rotate_image_cpu()
    main_rotate_image_gpu()
    # main_print_info()
