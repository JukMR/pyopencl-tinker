import numpy as np
from numpy import sin, cos, pi
from PIL import Image
import matplotlib.pyplot as plt
import pyopencl as cl

def rotate_image(source, target, img, angle = pi / 4):
    def rotated(x, y, px, py, t):
        v = np.array([x, y])
        p = np.array([px, py])
        rott = np.array([
            [cos(t), -sin(t)],
            [sin(t), cos(t)]
        ])
        return rott @ (v - p) + p

    height, width, _ = source.shape

    for x in range(width):
        for y in range(height):
            ox, oy = np.rint(rotated(x, y, width / 2, height / 2, -angle)).astype(int)
            if 0 <= ox < width and 0 <= oy < height:
                target[y, x] = source[oy, ox]

    return target

def show_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()

def main():
    platform_list = cl.get_platforms()
    devices = platform_list[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=devices)
    queue = cl.CommandQueue(context)

    img = Image.open("imagen.png")
    source = np.asarray(img).copy()
    height, width, _ = source.shape
    # target = np.zeros(source.shape, dtype=np.uint8)
    # target[:, :, 3] = 255

    mf = cl.mem_flags
    source_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
    target_d = cl.Buffer(context, mf.WRITE_ONLY, size=source.nbytes)

    kernel_source = """
        __kernel void rotate_from_pivot(
            __global const uchar *source,
            __global uchar *target)
        {
            // TODO: would be nice to write the kernel in its own .cl file
            // TODO: Use opencl vectors and matrices like this:
            // uchar2 v = (float2)(x, y);

            int x = get_global_id(0);
            int y = get_global_id(1);
            int width = get_global_size(0);
            int height = get_global_size(1);
            float t = M_PI_4;
            float cost = cos(t);
            float sint = sin(t);
            int px = width / 2;
            int py = height / 2;
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
    kernel.set_args(source_d, target_d)
    cl.enqueue_nd_range_kernel(queue, kernel, (width, height), (1, 1))
    target = np.zeros(source.shape, dtype=np.uint8)
    cl.enqueue_copy(queue, target, target_d)

    show_image(target)

if __name__ == '__main__':
    main()
