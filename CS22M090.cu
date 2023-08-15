#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void dkernel1(int *d_req_id, int *d_req_cen, int *d_req_start, int *d_req_slots,int *d_capacity, int *d_end, int *d_succ_req, int count){

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < count){
    int arr[24];
    for(int i = 0; i < 24; i++)
      arr[i] = 0;
    if(id == 0){
      for(int i = 0; i < d_end[0]; i++){
        int start = d_req_start[i] - 1;
        int end = start + d_req_slots[i];
        int var = start;
        while(var != end && end <= 24){
          if(arr[var] < d_capacity[id])
            var++;
          else
            break;
        }
        if(var == end){
          for(int j = start; j < end; j++)
            arr[j] += 1;
          
          atomicAdd(&d_succ_req[0],1);
      
        }
      }
    }
    else{
      for(int i = d_end[id-1]; i < d_end[id]; i++){
        int start = d_req_start[i] - 1;
        int end = start + d_req_slots[i];
        int var = start;
        while(var != end && end <= 24){
          if(arr[var] < d_capacity[id])
            var++;
          else 
            break;
        }
        if(var == end){
          for(int j = start; j < end; j++)
            arr[j] += 1;
          
          atomicAdd(&d_succ_req[d_req_cen[i]], 1);
        }
      }
    }
  }
}

int partition(int *req_id, int *req_cen, int *req_fac, int *req_start, int *req_slots, int low , int high){

  int pivot = req_cen[high];
  int pivot_fac = req_fac[high];
  int pivot_id = req_id[high];
  int i = low;
  int j = low;
  while( i <= high){
    if(req_cen[i] > pivot || (req_cen[i] == pivot && req_fac[i] > pivot_fac) || (req_cen[i] == pivot && req_fac[i] == pivot_fac && req_id[i] > pivot_id))
      i++;

    else{
      swap(req_cen[i], req_cen[j]);
      swap(req_id[i], req_id[j]);
      swap(req_fac[i], req_fac[j]);
      swap(req_start[i], req_start[j]);
      swap(req_slots[i], req_slots[j]);
      i++;
      j++;
    }
  }
  return j-1;
}

void quickSort(int *req_id, int *req_cen, int *req_fac, int *req_start, int *req_slots, int low , int high){

  if(low >= high)
    return;

  int part = partition(req_id, req_cen, req_fac, req_start, req_slots, low, high);

  quickSort(req_id, req_cen, req_fac, req_start, req_slots, part + 1, high);                         // left recursive quick sort

  quickSort(req_id, req_cen, req_fac, req_start, req_slots, low , part-1);                           // right recursive quick sort
}


//***********************************************


int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		quickSort(req_id, req_cen, req_fac, req_start, req_slots,0 , R-1);          // sorting the requesto on the basis of center, facility and req_id
    
    int count = 0;                                              // total number of facility in the entire system
    int *fac_count;                                            // stores number of request on each facility of centre
    int *fac_start;                                            // counts at what number the request of each centre start
    int *end;                                                  // use for range of each facility of each centre
    for(int i = 0; i < N; i++)
      count += facility[i];
    
    fac_count = (int *)malloc(count * sizeof(int));
    fac_start = (int *)malloc((N+1) * sizeof(int));
    end = (int *)malloc(count * sizeof(int));

    fac_start[0] = 0;
    for(int i = 0; i < count; i++)
      fac_count[i] = 0;

    for(int i = 1; i <= N; i++)
      fac_start[i] += fac_start[i-1]+facility[i-1];
 
    for(int i = 0; i < R; i++)
      fac_count[fac_start[req_cen[i]]+req_fac[i]]++;

    end[0] = fac_count[0];
    for(int i = 1; i < count; i++){
      end[i] = end[i-1] + fac_count[i];
    }
    
    int block = ceil((float)count/1024);
    int *d_req_id, *d_req_cen, *d_req_start, *d_req_slots, *d_capacity, *d_end, *d_amd, *d_succ_req;
    cudaMalloc(&d_req_id, R * sizeof(int));
    cudaMalloc(&d_req_cen, R * sizeof(int));
    cudaMalloc(&d_req_start, R * sizeof(int));
    cudaMalloc(&d_req_slots, R * sizeof(int));
    cudaMalloc(&d_end, count * sizeof(int));
    cudaMalloc(&d_succ_req, N*sizeof(int));
    cudaMalloc(&d_amd, N*sizeof(int));
    cudaMalloc(&d_capacity, count * sizeof(int));
    cudaMemset(d_succ_req, 0, N*sizeof(int));
    cudaMemcpy(d_req_id, req_id, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start, req_start, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots, req_slots, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_capacity, capacity, count * sizeof(int), cudaMemcpyHostToDevice);

    // //*********************************
    // // Call the kernels here
    dkernel1<<<block, 1024>>>(d_req_id, d_req_cen, d_req_start, d_req_slots,d_capacity, d_end, d_succ_req, count);
    cudaMemcpy(succ_reqs, d_succ_req, N * sizeof(int), cudaMemcpyDeviceToHost);
    // //********************************

    for(int i = 0; i < N; i++)
      success+=succ_reqs[i];                            // counting the total number of request granted
    
    int total = 0;
    for(int i = 0; i < N; i++)
      total += tot_reqs[i];
    fail = total - success;

    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}