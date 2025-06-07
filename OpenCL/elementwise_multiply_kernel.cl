__kernel void elementwise_multiply_kernel(
    __global const float* A,    // Input vector/matrix A
    __global const float* B,    // Input vector/matrix B
    __global float* C,          // Output vector/matrix C = A * B (element-wise)
    const int N)                // Total number of elements
{
    int gid = get_global_id(0);

    if (gid < N) {
        C[gid] = A[gid] * B[gid];
    }
}