# AAC-GPU-KERNELS

A collaborative repository dedicated to providing a curated collection of high-performance GPU kernels across various computing frameworks. The goal is to create a readily available resource for developers to accelerate their applications by reusing optimized kernel implementations for common tasks.

## 🚀 Vision

Writing efficient GPU kernels from scratch can be time-consuming and complex. This repository aims to simplify that process by offering a growing library of pre-written, tested, and functioning kernels. Whether you're working with OpenCL, CUDA, or other GPU programming paradigms, you should be able to find and contribute reusable code snippets here.

## 📁 Repository Structure

The kernels are organized by their respective GPU computing framework:

* **`OpenCL/`**: Contains OpenCL C kernels (`.cl` files) and their associated Python-based test suite (using PyOpenCL).
    * `OpenCL/tests/python/`: Houses the Python testing framework for OpenCL kernels. Refer to its dedicated `README.md` for details on creating and running tests.
* **`CUDA/`**: (Future/Planned) Will contain CUDA C/C++ kernels (`.cu` files) and their respective testing frameworks.
* **`[Other Frameworks]/`**: (Future/Planned) Additional directories for other GPU computing frameworks like SYCL, Vulkan Compute, etc., as contributions grow.

Each framework directory will contain the kernel files themselves and, where applicable, a `tests/` subdirectory with language-specific test runners and examples.

## ✨ Features

* **Diverse Kernel Collection**: A growing set of GPU kernels for common computational tasks (e.g., element-wise operations, matrix multiplication, reductions).
* **Framework Agnostic Organization**: Kernels are categorized by their programming framework for easy navigation.
* **Reusable Code**: Designed for direct integration into your projects.
* **Comprehensive Testing**: Each kernel framework aims to have its own robust testing infrastructure to ensure correctness and performance.
* **Community-Driven**: Open for contributions from developers worldwide.

## 🧪 OpenCL Test Suite

For detailed instructions on how to create, run, and manage test cases for the OpenCL kernels, please refer to the specific `README.md` located at:
[OpenCL Tests README](OpenCL/tests/python/README.md)

This separate `README.md` covers:
* Setting up the PyOpenCL testing environment.
* The structure of OpenCL kernel test files.
* How to use `generate_test_mapping.py` and `test_runner.py`.

## 🤝 Contributing

We welcome contributions from the community! If you have an optimized GPU kernel you'd like to share, or if you want to improve existing ones, please refer to our `CONTRIBUTING.md` for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Start exploring and contributing to faster GPU computing today!**