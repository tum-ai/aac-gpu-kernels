import numpy as np
import pyopencl as cl

def test_elementwise_divide_by_scalar_kernel(ctx, queue, prg):
    print("\n--- Testing elementwise_divide_by_scalar_kernel ---")
    # Simulate a matrix of batch_size x num_cols
    batch_size = 8
    num_cols = 10
    num_elements = batch_size * num_cols

    data_np = np.random.rand(batch_size, num_cols).astype(np.float32) * 10.0 + 1.0 # Add 1 to avoid zero
    # Simulate row sums (each element corresponds to a row sum)
    row_sums_np = np.random.rand(batch_size).astype(np.float32) * 5.0 + 0.1 # Add 0.1 to avoid division by zero

    # Expected output: data_np divided by its corresponding row_sum (broadcasted)
    data_np_expected = data_np / row_sums_np[:, np.newaxis]

    data_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_np)
    row_sums_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=row_sums_np)

    try:
        prg.elementwise_divide_by_scalar_kernel(queue, (num_elements,), None,
                                                data_buf, row_sums_buf,
                                                np.int32(num_elements), np.int32(num_cols)).wait()
        data_np_result = np.empty_like(data_np)
        cl.enqueue_copy(queue, data_np_result, data_buf).wait()

        if np.allclose(data_np_expected, data_np_result, rtol=1e-4, atol=1e-6):
            print("SUCCESS: elementwise_divide_by_scalar_kernel matches NumPy.")
        else:
            print("FAILURE: elementwise_divide_by_scalar_kernel does NOT match NumPy.")
            print(f"Input (first 2 rows):\n{data_np[:2]}")
            print(f"Row Sums (first 2): {row_sums_np[:2]}")
            print(f"Expected (first 2 rows):\n{data_np_expected[:2]}")
            print(f"Result (first 2 rows):\n{data_np_result[:2]}")
            print(f"Max diff: {np.max(np.abs(data_np_expected - data_np_result))}")
    except Exception as e:
        print(f"ERROR: elementwise_divide_by_scalar_kernel compilation/execution failed: {e}", file=sys.stderr)