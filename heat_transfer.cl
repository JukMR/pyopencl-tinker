__kernel void next_cell_temp(
    __global const double *temps,
    __global const uchar *sources,
    __global double *next_temps
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int N = get_global_size(1);
    int M = get_global_size(0);

    if (sources[y * M + x]) // It is a heat source
      next_temps[y * M + x] = temps[y * M + x];
    else next_temps[y * M + x] = (
        temps[y * M + x] + // Current
        temps[(y - 1) * M + x] + // Top
        temps[y * M + x + 1] + // Right
        temps[(y + 1) * M + x] + // Bottom
        temps[y * M + x - 1] // Left
    ) / 5;
}
