# OpenCL Kernel Test Suite

This section provides a framework for testing OpenCL kernels written in C/C++ using PyOpenCL. It includes a test runner to automate the execution of tests and a utility to generate boilerplate test mapping JSON.

## Project Structure

```
.
├── helpers.py                  # Helper functions (e.g., OpenCL context creation)
├── kernel_test_mapping.json    # Maps kernel files to their corresponding test modules
├── generate_test_mapping.py    # Generates/updates the kernel_test_mapping.json
├── test_runner.py              # The main script to run all kernel tests
├── your_kernel_1.cl            # Your OpenCL kernel file
├── your_kernel_2.cl            # Another OpenCL kernel file
└── kerneltests/
    ├── test_kernel_1.py        # Python test file for your_kernel_1.cl
    └── test_kernel_2.py        # Python test file for your_kernel_2.cl
```

## Setup

1.  **Install PyOpenCL and NumPy:**
    ```bash
    pip install pyopencl numpy
    ```

2.  **Ensure OpenCL Drivers:** Make sure you have the appropriate OpenCL drivers installed for your GPU/CPU.

## How to Create a New Test Case

Follow these steps to add a new OpenCL kernel and its corresponding test case to the suite:

### 1. Create Your OpenCL Kernel File

Place your `.cl` OpenCL kernel file in the root directory (or in the `KERNELS_DIR` specified in `generate_test_mapping.py` and `test_runner.py`).

**Example: `my_new_kernel.cl`**

```c
__kernel void add_one_kernel(__global float *data, int N) {
    int gid = get_global_id(0);
    if (gid < N) {
        data[gid] += 1.0f;
    }
}
```

### 2. Generate/Update `kernel_test_mapping.json`

Run the `generate_test_mapping.py` script to add your new kernel to the mapping file. This script will identify new `.cl` files and add them with an empty `test` field, while preserving existing entries.

```bash
python generate_test_mapping.py
```

After running, your `kernel_test_mapping.json` will be updated. For `my_new_kernel.cl`, you'll see an entry like this:

```json
[
    {
        "kernel": "my_new_kernel.cl",
        "test": ""
    }
    // ... other entries
]
```

### 3. Create Your Python Test File

Create a new Python file in the `kerneltests/` directory. The filename should be `test_` followed by a descriptive name (e.g., `test_my_new_kernel.py`).

Inside this file, define one or more test functions. Each test function **must** start with `test_` and accept `ctx`, `queue`, and `prg` as arguments.

* `ctx`: The PyOpenCL context.
* `queue`: The PyOpenCL command queue.
* `prg`: The PyOpenCL program object, built from your kernel source. You can call individual kernels using `prg.your_kernel_name(...)`.

**Example: `kerneltests/test_my_new_kernel.py`**

```python
import pyopencl as cl
import numpy as np
import sys

def test_add_one_kernel(ctx, queue, prg):
    print("\n--- Testing add_one_kernel ---")
    N = 10
    data_np = np.arange(N, dtype=np.float32) # [0, 1, ..., 9]
    data_np_expected = data_np + 1.0        # [1, 2, ..., 10]

    # Create a buffer on the device and copy host data to it
    data_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data_np)

    try:
        # Execute the kernel
        # (N,) specifies the global work size, None for local work size
        prg.add_one_kernel(queue, (N,), None, data_buf, np.int32(N)).wait()

        # Create an empty numpy array to hold the result from the device
        data_np_result = np.empty_like(data_np)
        # Copy the result from the device buffer back to the host array
        cl.enqueue_copy(queue, data_np_result, data_buf).wait()

        # Compare the result with the expected output
        if np.allclose(data_np_expected, data_np_result, rtol=1e-4, atol=1e-6):
            print("SUCCESS: add_one_kernel matches NumPy.")
        else:
            print("FAILURE: add_one_kernel does NOT match NumPy.")
            print(f"Input:        {data_np}")
            print(f"Expected:     {data_np_expected}")
            print(f"Result:       {data_np_result}")
            print(f"Max diff: {np.max(np.abs(data_np_expected - data_np_result))}")

    except Exception as e:
        print(f"ERROR: add_one_kernel compilation/execution failed: {e}", file=sys.stderr)
        # Re-raise the exception to be caught by the runner for proper failure count
        raise
```

### 4. Update `kernel_test_mapping.json` (Manual Step)

Open `kernel_test_mapping.json` and update the `test` field for your new kernel to point to the name of your test file (without the `.py` extension).

**Example: Updated `kernel_test_mapping.json`**

```json
[
    {
        "kernel": "my_new_kernel.cl",
        "test": "test_my_new_kernel"  # <--- Fill this in!
    }
    // ... other entries
]
```

## Running the Tests

Once you've set up your kernels and test files, you can run the entire test suite using `test_runner.py`.

```bash
python test_runner.py
```

The `test_runner.py` script will:
1.  Initialize an OpenCL context and queue.
2.  Read the `kernel_test_mapping.json` file.
3.  For each entry:
    * Load and build the specified OpenCL kernel.
    * Dynamically import the corresponding Python test module.
    * Execute all functions within that module that start with `test_`, passing the context, queue, and built program.
4.  Print a summary of the test results (successes and failures).

If any test fails or encounters an error, the runner will report it, including details if your test function provides them. The script will also exit with a non-zero status code if there are any failures, which is useful for CI/CD pipelines.