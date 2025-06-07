__kernel void matrix_mul_kernel(
    __global const float* A,    // Input matrix A (M x K)
    __global const float* B,    // Input matrix B (K x N)
    __global float* C,          // Output matrix C (M x N)
    const int M,                // Number of rows in A and C
    const int K,                // Number of columns in A, rows in B
    const int N)                // Number of columns in B and C
{
    // Get the global ID for the current work-item along the X-axis (0th dimension)
    // In our 2D execution space, this maps to the row index of the output matrix C.
    int row = get_global_id(0);

    // Get the global ID for the current work-item along the Y-axis (1st dimension)
    // This maps to the column index of the output matrix C.
    int col = get_global_id(1);

    // Check if the current work-item is within the bounds of the output matrix C
    // This handles cases where the global size might be larger than the actual matrix dimensions.
    if (row < M && col < N) {
        // Initialize a sum variable for the dot product calculation for C[row][col]
        float sum = 0.0f;

        // Loop through the 'K' dimension (columns of A and rows of B)
        // This loop performs the dot product of row 'row' from A and column 'col' from B.
        for (int k = 0; k < K; ++k) {
            // Calculate the 1D index for A[row][k]
            // A is stored row-major, so A[row][k] is at index (row * K + k)
            int index_A = row * K + k;

            // Calculate the 1D index for B[k][col]
            // B is stored row-major, so B[k][col] is at index (k * N + col)
            int index_B = k * N + col;

            // Perform the multiplication and accumulate the sum
            sum += A[index_A] * B[index_B];
        }

        // Store the final computed sum (dot product) into the corresponding
        // position in the output matrix C. C is also stored row-major.
        C[row * N + col] = sum;
    }
}