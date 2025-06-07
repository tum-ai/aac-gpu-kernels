import pyopencl as cl
import os
import json
import importlib
import sys

# Add the directory containing helpers.py to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helpers import pyopencl_ctx_helper

# Configuration
TEST_MAPPING_FILE = 'kernel_test_mapping.json'
KERNELS_DIR = '../../'  # Assuming kernels are in the current directory or specify a path
TESTS_DIR = './kerneltests' # Directory where your test files are located

def run_tests():
    print("Initializing OpenCL context and queue...")
    try:
        ctx, queue = pyopencl_ctx_helper.create_context_and_queue()
        print("OpenCL context and queue created successfully.")
    except Exception as e:
        print(f"ERROR: Could not create OpenCL context/queue: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(TEST_MAPPING_FILE):
        print(f"ERROR: Test mapping file '{TEST_MAPPING_FILE}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(TEST_MAPPING_FILE, 'r') as f:
        test_mappings = json.load(f)

    # Add the tests directory to sys.path so we can import modules from it
    sys.path.insert(0, TESTS_DIR)

    total_tests_run = 0
    total_failures = 0

    for mapping in test_mappings:
        kernel_path = mapping['kernel']
        test_module_name = mapping['test']

        full_kernel_path = os.path.join(KERNELS_DIR, kernel_path)
        if not os.path.exists(full_kernel_path):
            print(f"WARNING: Kernel file '{full_kernel_path}' not found. Skipping test for this entry.", file=sys.stderr)
            continue

        print(f"\n--- Processing kernel: {full_kernel_path} with test module: {test_module_name} ---")

        try:
            with open(full_kernel_path, 'r') as f:
                kernel_code = f.read()
            prg = cl.Program(ctx, kernel_code).build()
            print(f"Successfully built kernel: {full_kernel_path}")
        except Exception as e:
            print(f"ERROR: Failed to build kernel '{full_kernel_path}': {e}", file=sys.stderr)
            total_failures += 1
            continue

        try:
            test_module = importlib.import_module(test_module_name)
            print(f"Successfully imported test module: {test_module_name}")
        except ImportError as e:
            print(f"ERROR: Could not import test module '{test_module_name}': {e}", file=sys.stderr)
            total_failures += 1
            continue

        # Find and run test functions within the module
        module_failures = 0
        for func_name in dir(test_module):
            if func_name.startswith('test_') and callable(getattr(test_module, func_name)):
                test_func = getattr(test_module, func_name)
                print(f"Running test function: {func_name}")
                total_tests_run += 1
                try:
                    # Pass ctx, queue, and prg to the test function
                    test_func(ctx, queue, prg)
                except Exception as e:
                    print(f"ERROR: Test function '{func_name}' failed with an exception: {e}", file=sys.stderr)
                    module_failures += 1
        if module_failures > 0:
            total_failures += module_failures
            print(f"FAILURE: {module_failures} test(s) failed in module {test_module_name}.")

    sys.path.remove(TESTS_DIR) # Clean up sys.path

    print("\n--- Test Summary ---")
    print(f"Total tests run: {total_tests_run}")
    if total_failures == 0:
        print("All tests passed successfully!")
    else:
        print(f"Total failures: {total_failures}")
        print("Some tests failed.")
        sys.exit(1) # Indicate failure to the calling environment

if __name__ == "__main__":
    run_tests()