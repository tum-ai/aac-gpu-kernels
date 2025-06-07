import pyopencl as cl
import numpy as np

def test_matrix_mul_kernel(ctx, queue, prg):
    print("\n--- Testing matrix_mul_kernel ---")
    M, K, N = 64, 32, 128 # Example dimensions
    a_np = np.random.rand(M, K).astype(np.float32)
    b_np = np.random.rand(K, N).astype(np.float32)
    c_np_expected = a_np @ b_np # NumPy matrix multiplication

    a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
    b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
    c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, M * N * a_np.itemsize)

    try:
        prg.matrix_mul_kernel(queue, (M, N), None, a_buf, b_buf, c_buf,
                              np.int32(M), np.int32(K), np.int32(N)).wait()
        c_np_result = np.empty((M, N), dtype=np.float32)
        cl.enqueue_copy(queue, c_np_result, c_buf).wait()

        if np.allclose(c_np_expected, c_np_result, rtol=1e-4, atol=1e-6):
            print("SUCCESS: matrix_mul_kernel matches NumPy.")
        else:
            print("FAILURE: matrix_mul_kernel does NOT match NumPy.")
            print(f"Expected (top-left 3x3):\n{c_np_expected[:3,:3]}")
            print(f"Result (top-left 3x3):\n{c_np_result[:3,:3]}")
            print(f"Max diff: {np.max(np.abs(c_np_expected - c_np_result))}")
    except Exception as e:
        print(f"ERROR: matrix_mul_kernel compilation/execution failed: {e}", file=sys.stderr)