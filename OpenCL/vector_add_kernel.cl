__kernel void vector_add_kernel(
    __global const float* A,    // Input vector A
    __global const float* B,    // Input vector B (the bias)
    __global float* C,          // Output vector C = A + B
    const int N)                // Size of the vectors
{
    // Get the global ID of the current work-item in the 1D domain
    // This ID corresponds to the index of the element we are processing
    int gid = get_global_id(0);

    // Ensure that the work-item ID is within the bounds of the vector size
    // This is important for cases where the global size is not perfectly divisible
    // by the work-group size, leading to extra work-items.
    if (gid < N) {
        // Perform the element-wise addition and store the result
        C[gid] = A[gid] + B[gid];
    }
}