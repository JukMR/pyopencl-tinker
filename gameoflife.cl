bool is_valid(int x, int y, int N, int M) {
   return 0 <= x && x < M && 0 <= y && y < N;
}

__kernel void next_cell_state(
    __global const uchar *frame,
    __global uchar *next_frame
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int N = get_global_size(1);
    int M = get_global_size(0);

    int neighbours = 0;
    int xl = x - 1;
    int xr = x + 1;
    int yu = y - 1;
    int yd = y + 1;
    uchar is_alive = frame[y * M + x];

    neighbours += is_valid(yu, xl, N, M) ? (frame[yu * M + xl] != 0) : 0;
    neighbours += is_valid(yu, x, N, M) ? (frame[yu * M + x] != 0) : 0;
    neighbours += is_valid(yu, xr, N, M) ? (frame[yu * M + xr] != 0) : 0;
    neighbours += is_valid(y, xl, N, M) ? (frame[y * M + xl] != 0) : 0;
    neighbours += is_valid(y, xr, N, M) ? (frame[y * M + xr] != 0) : 0;
    neighbours += is_valid(yd, xl, N, M) ? (frame[yd * M + xl] != 0) : 0;
    neighbours += is_valid(yd, x, N, M) ? (frame[yd * M + x] != 0) : 0;
    neighbours += is_valid(yd, xr, N, M) ? (frame[yd * M + xr] != 0) : 0;

    if (is_alive) {
        next_frame[y * M + x] = (2 <= neighbours && neighbours <= 3) ? 255 : 0;
    } else {
        next_frame[y * M + x] = neighbours == 3 ? 255 : 0;
    }
}
