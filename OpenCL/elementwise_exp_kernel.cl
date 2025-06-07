__kernel void elementwise_exp_kernel(
    __global float* data,   // Input/Output data vector/matrix
    const int N)            // Total number of elements
{
    int gid = get_global_id(0);

    if (gid < N) {
        // Apply the exponential function: e^x
        // Use the built-in exp() function
        data[gid] = exp(data[gid]);
    }
}