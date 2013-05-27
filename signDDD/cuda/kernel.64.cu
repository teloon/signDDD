#ifndef _HELLOWORLD_KERNEL_H_
#define _HELLOWORLD_KERNEL_H_


#include <stdio.h>


///////////////////////////////////////////////////////////
// Simple Hello World kernel
// @param gpu_odata output data in global memory
///////////////////////////////////////////////////////////

__global__ void CudaTest_kernel(int max_sim_num, int threshold, unsigned long* di_index, unsigned long* di_query, unsigned long* result, char* di_static_table, unsigned int* result_num, unsigned int frame_num_index, unsigned int frame_num_query){
	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx = blockIdx.x;
	const unsigned int block_idy = blockIdx.y;
	const unsigned int di_index_subset_begin = block_idx*1024;
	const unsigned int di_index_subset_size = ((block_idx+1)*1024>frame_num_index)?(frame_num_index%1024):1024;
	const unsigned int di_query_subset_begin = block_idy*512;
	const unsigned int di_query_subset_size = ((block_idy+1)*512>frame_num_query)?(frame_num_query%512):512;
	__shared__ unsigned long di_index_subset[1024];
	__shared__ unsigned long di_query_subset[512];
	__shared__ char shared_static_table[256];

	if(thread_idx<di_index_subset_size)
		di_index_subset[thread_idx] = di_index[di_index_subset_begin+thread_idx];
	if(thread_idx<di_query_subset_size)
		di_query_subset[thread_idx] = di_query[di_query_subset_begin+thread_idx];
	if(thread_idx<256)
		shared_static_table[thread_idx]= di_static_table[thread_idx];
	__syncthreads();

	unsigned long xor_result = 0;
	unsigned int num_diff_bit = 0;
	unsigned int index_local=0;

	if(thread_idx < di_index_subset_size)
		for(int i = 0 ; i < di_query_subset_size ; i ++){
			xor_result = di_query_subset[i] xor di_index_subset[thread_idx];
			num_diff_bit =   shared_static_table[xor_result & 0xff] + shared_static_table[xor_result>>8 & 0xff]
			               + shared_static_table[xor_result>>16 & 0xff] + shared_static_table[xor_result>>24 & 0xff]
			               + shared_static_table[xor_result>>32 & 0xff] + shared_static_table[xor_result>>40 & 0xff]
			               + shared_static_table[xor_result>>48 & 0xff] + shared_static_table[xor_result>>56 & 0xff];
		  if( num_diff_bit < threshold ){
				index_local = atomicAdd(&result_num[(block_idy*512+i)], 1);
				if(index_local>=max_sim_num) break;
				result[(block_idy*512+i)*max_sim_num + index_local] = di_index_subset_begin+thread_idx;
		   }
		}
}



#endif // #ifndef _HELLOWORLD_KERNEL_H_
