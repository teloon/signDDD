

// Host defines
#define NUM_THREADS 1024
#define NUM_GRID 1
#define MAX_SIM_NUM 50000
#define THRESHOLD 2

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <limits.h>
using namespace std;
// GPU Kernels declarations
__global__ void CudaTest_kernel(int max_sim_num, int threshold, unsigned int* di_index, unsigned int* di_query, unsigned int* result, char* di_static_table, unsigned int* result_num, unsigned int frame_num_index, unsigned int frame_num_query);

void bulidSimTable(string hashcodePath, string queryPath, string resultPath, unsigned int frame_num_index, unsigned int frame_num_query, int threshold){
	clock_t cstart=clock();
	//initalize data size
//	unsigned int frame_num_index = 924287;//number of frames in lib
//	unsigned int frame_num_query =1068;//number of frames in query
	unsigned int static_table_num = 256;
	unsigned int mem_size_lib = frame_num_index * sizeof(unsigned int);//the memory size hvectors_in_lib needed
	unsigned int mem_size_query = frame_num_query * sizeof(unsigned int);//the memory size hvectors_query needed
	unsigned int result_size = frame_num_query * MAX_SIM_NUM * sizeof(unsigned int);//the memory size which will store the result
	unsigned int static_table_size = static_table_num * sizeof(char);


	// Host variables
	unsigned int* hi_index;//input data in cpu which contains all feature vectors of frames in lib
	unsigned int* hi_query;//input data in cpu which contains all feature vectors of query frames
	unsigned int* ho_result;//the result in host
	unsigned int* ho_result_cnt;
	char hi_static_table[256]=
	{
					0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
			        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
	};


	// Device variables
	unsigned int* di_index;//input data in gpu which contains all feature vectors of frames
	unsigned int* di_query;//input data in cpu which contains all feature vectors of query frames
	unsigned int* do_result;//the result in device
	unsigned int* do_result_cnt;
	char* di_static_table;


	//allocate memory in host and device
	hi_index = (unsigned int*) malloc(mem_size_lib);//allocate the memory to hvectors_in_lib
	hi_query = (unsigned int*) malloc(mem_size_query);//allocate the memory to hvectors_query
	ho_result_cnt = (unsigned int*) malloc(mem_size_query);

	cudaMalloc((void**) &di_index, mem_size_lib);//allocate dvectors_in_lib to the device memory
	cudaMalloc((void**) &di_query, mem_size_query);//allocate dvectors_query to the device memory
	cudaMalloc((void**) &di_static_table, static_table_size);//allocate device_result to the device memory
	cudaMalloc((void**) &do_result_cnt, mem_size_query);

	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void**)&(ho_result),result_size,cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&(do_result),ho_result,0);


	//initalize host data
	unsigned int n;
	int cnt = 0;
	FILE *fp,*fp2;

	fp = fopen(hashcodePath.c_str(),"r");
	//cout<<hashcodePath.c_str()<<endl;
	while(!feof(fp) && cnt < frame_num_index){
		fscanf(fp, "%u", &n);
		hi_index[cnt] = n;
		cnt++;
	}
	//cout<<"count ="<<cnt<<endl;

	cnt = 0;
	fp = fopen(queryPath.c_str(),"r");
	//cout<<queryPath.c_str()<<endl;
	while(!feof(fp) && cnt < frame_num_query){
		fscanf(fp,"%u", &n);
		hi_query[cnt] = n;
		cnt++;
	}
	//cout<<"count = "<<cnt<<endl;


	memset(ho_result_cnt, 0, frame_num_query*4);


	//initalize device data
	 cudaMemcpy(di_index, hi_index, mem_size_lib, cudaMemcpyHostToDevice);//copy hvectors_in_lib in memory to dvectors_in_lib in gpu
	 cudaMemcpy(di_query, hi_query, mem_size_query, cudaMemcpyHostToDevice);//copy hvectors_query in memory to dvectors_query in gpu
	 cudaMemcpy(di_static_table,hi_static_table,static_table_size,cudaMemcpyHostToDevice);
	 cudaMemcpy(do_result_cnt, ho_result_cnt, mem_size_query, cudaMemcpyHostToDevice);


	//excute kernel in GPU
	//int threshold = THRESHOLD;
	int max_sim_num = MAX_SIM_NUM;
	dim3 grid((int)ceil((double)frame_num_index/1024), (int)ceil((double)frame_num_query/512), 1);


	CudaTest_kernel<<<grid, NUM_THREADS>>>(max_sim_num, threshold, di_index, di_query, do_result, di_static_table, do_result_cnt, frame_num_index, frame_num_query);

	cudaThreadSynchronize();
	//copy the result back to host memory
	cudaMemcpy(ho_result_cnt, do_result_cnt, mem_size_query, cudaMemcpyDeviceToHost);
	cudaMemcpy(ho_result, do_result, result_size, cudaMemcpyDeviceToHost);



	//deal with the result

//	printf("query:%u\n",hi_query[0]);
//	for(int i = 0; i < frame_num_query ; i ++)
//	{
//		//printf("result 0:%u\n",ho_result_cnt[i]);
//		for(int j = 0; j < ho_result_cnt[i]; j++)
//			cout<<ho_result[i*MAX_SIM_NUM+j]<<" ";
//		cout<<endl;
//	}
	fp2 = fopen(resultPath.c_str(),"w+");
	int max_length = 0;
	for(int i = 0 ; i < frame_num_query ; i++)
	{
		if(max_length < ho_result_cnt[i])
			max_length = ho_result_cnt[i];
		for(int j = 0; j < ho_result_cnt[i]; j++)
			fprintf(fp2,"%u ",ho_result[i*MAX_SIM_NUM+j]);
		fprintf(fp2,"\n");
	}
	cout<<"max matched count is: "<<max_length<<endl;
	cout << "size of long: " << sizeof(2l) << endl;
	clock_t cend=clock();
	printf("time:%f\n",(float)(cend-cstart)/CLOCKS_PER_SEC);

////	free up the host memory
//	free(ho_result_cnt);
//	free(hi_index);
//	free(hi_query);
//	free(ho_result);
//
//
////	free up the device memory
//	cudaFree(do_result_cnt);
//	cudaFree(di_index);//
//	cudaFree(di_query);
//	cudaFree(do_result);

}

void sortTable(unsigned int* ho_result_cnt, unsigned int* ho_result, unsigned int frame_num_query){

}

//////////////////////
// Program main
//////////////////////
int main(int argc, char** argv) {
	if(argc == 7)
	{
		string hashcodePath = argv[1];
		string queryPath = argv[2];
		string resultPath = argv[3];
		int frame_num_index = atoi(argv[4]);
		int frame_num_query = atoi(argv[5]);
		int threshold = atoi(argv[6]);
//		unsigned int frame_num_index = 1000001;
//		unsigned int frame_num_query =493;
//		int threshold = 2;
//		string hashcodePath = "/root/hashcode.txt";
//		string queryPath = "/root/query.txt";
//		string resultPath = "/root/result.txt";
		//cout<<hashcodePath<<endl<<queryPath<<endl<<frame_num_index<<endl<<frame_num_query<<endl<<threshold<<endl;
		bulidSimTable(hashcodePath, queryPath, resultPath, frame_num_index, frame_num_query, threshold);
	}
	return 0;
}

