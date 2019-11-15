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
