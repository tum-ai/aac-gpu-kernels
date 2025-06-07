# Contributing to AAC-GPU-KERNELS

We are thrilled that you're interested in contributing to `AAC-GPU-KERNELS`! Your contributions help build a valuable resource for the GPU computing community. By contributing, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md) (coming soon) and the [MIT License](LICENSE) of this project.

## How to Contribute

The general workflow for contributing to this repository is as follows:

1. **Fork the Repository**: Start by forking the `AAC-GPU-KERNELS` repository to your GitHub account.

2. **Clone Your Fork**: Clone your forked repository to your local machine:

```

git clone [https://github.com/tum-ai/aac-gpu-kernels](https://github.com/tum-ai/aac-gpu-kernels)
cd AAC-GPU-KERNELS

```

3. **Create a New Branch**: Create a new branch for your feature or bug fix. Use a descriptive name (e.g., `feature/add-vector-normalize-opencl`, `fix/matmul-bug-cuda`):

```

git checkout -b feature/your-awesome-contribution

```

4. **Implement Your Changes**:

* Add your new kernel files to the appropriate framework directory (e.g., `OpenCL/`, `CUDA/`).

* **Always include tests** for your new kernels. Refer to the specific `tests/` directory within each framework for how to write tests (e.g., `OpenCL/tests/python/README.md`).

* Update documentation where necessary (e.g., if you add a new framework, update the root `README.md`).

5. **Test Your Changes**: Ensure your new kernels work correctly and that all existing tests pass.

* For OpenCL kernels, follow the instructions in `OpenCL/tests/python/README.md`.

6. **Commit Your Changes**: Write clear and concise commit messages.

```

git add .
git commit -m "feat: Add new vector normalize kernel for OpenCL"

```

7. **Push to Your Fork**: Push your branch to your forked repository on GitHub:

```

git push origin feature/your-awesome-contribution

```

8. **Create a Pull Request (PR)**: Go to your forked repository on GitHub and create a Pull Request against the `main` branch of the original `AAC-GPU-KERNELS` repository. Provide a detailed description of your changes in the PR description.

## What to Contribute

We welcome contributions of high-quality, optimized GPU kernels for various computational tasks. Examples include:

* **Mathematical Operations**: Vector addition, matrix multiplication, element-wise operations (exp, log, sin, cos), reductions (sum, min, max), convolutions.

* **Data Manipulation**: Sorting, filtering, unique elements, prefix sums.

* **Utility Kernels**: Memory copy, initialization, etc.

Contributions for new GPU computing frameworks (e.g., SYCL, Vulkan Compute, WebGPU) are also highly encouraged, provided they include a testing strategy.

## Coding Standards and Guidelines

* **Kernel Code**:

* **Readability**: Write clear, well-commented code.

* **Efficiency**: Aim for optimal performance. Consider memory access patterns, thread divergence, and register usage.

* **Portability**: Where possible, write kernels that are broadly compatible within their respective frameworks.

* **Error Handling**: Consider edge cases and potential errors.

* **Python Test Code (for OpenCL)**:

* Follow PEP 8 for Python code style.

* Ensure tests are comprehensive and cover various input sizes and edge cases.

* Use `numpy.allclose` with appropriate `rtol` and `atol` for floating-point comparisons.

* **Documentation**:

* Add comments to your kernel code explaining its purpose, arguments, and any specific optimizations.

* If you add a new framework or significantly change the testing infrastructure, update the relevant `README.md` files.

## Testing Requirements

**All new kernel contributions must include corresponding test cases.**

* **For OpenCL Kernels**: Please add your test files to the `OpenCL/tests/python/kerneltests/` directory and update the `OpenCL/tests/python/kernel_test_mapping.json` file as described in `OpenCL/tests/python/README.md`.

* **For New Frameworks/Existing Frameworks without a Test Suite**: If you are contributing kernels for a framework that doesn't yet have a testing infrastructure, please consider setting one up as part of your contribution. This ensures the long-term maintainability and correctness of the kernels.

## Reporting Issues

If you find a bug in an existing kernel, have a suggestion for improvement, or encounter any other issues, please open an issue on the GitHub issue tracker. Provide as much detail as possible, including steps to reproduce, expected behavior, and actual behavior.

Thank you for making `AAC-GPU-KERNELS` a better resource for everyone!