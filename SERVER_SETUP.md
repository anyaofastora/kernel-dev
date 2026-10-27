# Run on an RTX 4090 server

This project contains independent CUDA course examples. Start with the vector-add
example below, then build other examples as described in their section READMEs.
The original course is at https://github.com/Infatoshi/cuda-course.

## Clone on the server

These commands assume Ubuntu/Linux with Git installed.

```bash
git clone https://github.com/anyaofastora/kernel-dev.git
cd kernel-dev
```

If you prefer SSH and have GitHub SSH access configured, clone
`git@github.com:anyaofastora/kernel-dev.git` instead.

## Check the CUDA environment

```bash
nvidia-smi
nvcc --version
g++ --version
```

You need a working NVIDIA driver, CUDA Toolkit with `nvcc`, and a host C++ compiler
supported by that toolkit. If something is missing, follow
[the setup guide](02_Setup/README.md) and choose the installer for your server's OS.
The CUDA version printed by `nvidia-smi` describes driver support; `nvcc --version`
reports the toolkit actually used to compile.

## Build and run the first example

Run from the repository root:

```bash
mkdir -p build
nvcc -O3 -arch=sm_89 \
  "05_Writing_your_First_Kernels/02 Kernels/00_vector_add_v1.cu" \
  -o build/vector_add
./build/vector_add
```

Look for `Results are correct` after the CPU and GPU timing output.
The RTX 4090 has compute capability 8.9, hence `sm_89`:
[NVIDIA GPU table](https://developer.nvidia.com/cuda/gpus).
If the compiler rejects `sm_89`, install a toolkit that supports Ada (CUDA 11.8 or newer).

For explanations and exercises, see
[the section 05 teaching guide](05_Writing_your_First_Kernels/TEACHING_GUIDE.md).
Later sections may need additional dependencies such as cuDNN, PyTorch, or Triton;
see their individual READMEs.

## Fetch later changes

```bash
git pull --ff-only
```

Recompile the example after pulling source changes.
