__kernel void sigmoid_kernel(
    __global float* data,   // Input/Output data vector/matrix
    const int N)            // Size of the data vector
{
    // Get the global ID of the current work-item in the 1D domain
    int gid = get_global_id(0);

    // Ensure the work-item ID is within the bounds of the data size
    if (gid < N) {
        // Apply the Sigmoid function: 1 / (1 + exp(-x))
        // Use the built-in exp() function from OpenCL math library
        data[gid] = 1.0f / (1.0f + exp(-data[gid]));
    }
}