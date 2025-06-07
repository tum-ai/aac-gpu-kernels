import pyopencl as cl
import numpy as np

def test_elementwise_multiply_kernel(ctx, queue, prg):
    print("\n--- Testing elementwise_multiply_kernel ---")
    N = 100
    a_np = np.random.rand(N).astype(np.float32) * 10
    b_np = np.random.rand(N).astype(np.float32) * 5
    c_np_expected = a_np * b_np

    a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_np)
    b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_np)
    c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a_np.nbytes)

    try:
        prg.elementwise_multiply_kernel(queue, (N,), None, a_buf, b_buf, c_buf, np.int32(N)).wait()
        c_np_result = np.empty_like(a_np)
        cl.enqueue_copy(queue, c_np_result, c_buf).wait()

        if np.allclose(c_np_expected, c_np_result, rtol=1e-4, atol=1e-6):
            print("SUCCESS: elementwise_multiply_kernel matches NumPy.")
        else:
            print("FAILURE: elementwise_multiply_kernel does NOT match NumPy.")
            print(f"Expected (first 5): {c_np_expected[:5]}")
            print(f"Result (first 5):   {c_np_result[:5]}")
            print(f"Max diff: {np.max(np.abs(c_np_expected - c_np_result))}")
    except Exception as e:
        print(f"ERROR: elementwise_multiply_kernel compilation/execution failed: {e}", file=sys.stderr)