all: 32 64

32: kernel.32.cu cuda_match.32.cu
	nvcc -arch sm_20 kernel.32.cu cuda_match.32.cu -o cuda_match.32

64: kernel.64.cu cuda_match.64.cu
	nvcc -arch sm_20 kernel.64.cu cuda_match.64.cu -o cuda_match.64
