
# gemv-test
You can compile cuda/gemv.cu with
```
 nvcc gemv.cu -o gemv -lcublas -arch=sm_80
```
You can compile opencl/gemv-ocl.cu with
```
g++ -o gemv-ocl gemv-ocl.cpp -lOpenCL -std=c++11 -L /usr/local/cuda/lib64/
```