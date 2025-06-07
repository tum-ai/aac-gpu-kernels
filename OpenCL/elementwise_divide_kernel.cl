__kernel void elementwise_divide_by_scalar_kernel(
    __global float* data,          // Input/Output matrix (e.g., exp(Z))
    __global const float* row_sums, // Vector of sums for each row (batch_size scalars)
    const int num_elements,        // Total number of elements in the data matrix
    const int num_cols)            // Number of columns (e.g., output_dim)
{
    // Get the global ID of the current work-item in the 1D domain
    int gid = get_global_id(0);

    if (gid < num_elements) {
        // Calculate the row index for the current work-item
        // The row_sums array has one sum per row (batch_size)
        int row_idx = gid / num_cols;

        // Get the sum for the current row
        float current_row_sum = row_sums[row_idx];

        // Perform the division. Handle division by zero (though unlikely with exp() sums).
        if (current_row_sum != 0.0f) {
            data[gid] = data[gid] / current_row_sum;
        } else {
            data[gid] = 0.0f; // Or some other appropriate handling
        }
    }
}