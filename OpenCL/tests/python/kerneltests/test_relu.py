import pyopencl as cl
import numpy as np

def test_relu_kernel(ctx, queue, prg):
    print("\n--- Testing relu_kernel ---")
    N = 100
    data_np = np.random.uniform(-1, 1, N).astype(np.float32) # Mix of positive and negative
    data_np_expected = np.maximum(0, data_np)

    data_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_np)

    try:
        prg.relu_kernel(queue, (N,), None, data_buf, np.int32(N)).wait()
        data_np_result = np.empty_like(data_np)
        cl.enqueue_copy(queue, data_np_result, data_buf).wait()

        if np.allclose(data_np_expected, data_np_result, rtol=1e-4, atol=1e-6):
            print("SUCCESS: relu_kernel matches NumPy.")
        else:
            print("FAILURE: relu_kernel does NOT match NumPy.")
            print(f"Input (first 5):    {data_np[:5]}")
            print(f"Expected (first 5): {data_np_expected[:5]}")
            print(f"Result (first 5):   {data_np_result[:5]}")
            print(f"Max diff: {np.max(np.abs(data_np_expected - data_np_result))}")
    except Exception as e:
        print(f"ERROR: relu_kernel compilation/execution failed: {e}", file=sys.stderr)