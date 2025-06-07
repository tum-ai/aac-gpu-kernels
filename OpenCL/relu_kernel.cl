__kernel void relu_kernel(
    __global float* data,   // Input/Output data vector/matrix
    const int N)            // Size of the data vector
{
    // Get the global ID of the current work-item in the 1D domain
    // This ID corresponds to the index of the element we are processing
    int gid = get_global_id(0);

    // Ensure the work-item ID is within the bounds of the data size
    if (gid < N) {
        // Apply the ReLU function: if x < 0, then x = 0; otherwise, x remains x.
        // We can use the fmax OpenCL built-in function for this.
        data[gid] = fmax(0.0f, data[gid]);

        // Alternatively, using an if-statement:
        // if (data[gid] < 0.0f) {
        //     data[gid] = 0.0f;
        // }
    }
}