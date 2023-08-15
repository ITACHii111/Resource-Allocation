/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/
// this kernel computes the end node at each level
 __global__ void findLevel(int *d_csrList, int *d_offset, int *end, int numberOFNodes, int endPoint, int index){

    int id = blockIdx.x * blockDim.x + threadIdx.x;                      // calculates global id of the thread              
    if(id < numberOFNodes){
        id = id + endPoint + 1;                                          // staring node connected to this thread node
        int endVertex = d_csrList[d_offset[id+1] - 1];
        atomicMax(end, endVertex);                                       // computing max of every connected node
    }
 }

 // this kernel computes the indgree and activeness of the nodes
 __global__ void computesIndegree(int *d_csrList, int *d_offset, int *d_active, int *d_apr, int *inDegree, int numberOFNodes, int startIndex, int current, int next, int level){

  int id = blockIdx.x * blockDim.x + threadIdx.x;               // calculating global id of the thread
  if(id < numberOFNodes){
    if(level == 1){
      int start = d_offset[id];
      int end = d_offset[id+1];
      if(start != end){
        while(start != end){
          int vertex = d_csrList[start++];               
          atomicAdd(&inDegree[vertex], 1);                     // incrementing indegree
        }
      }
    }
    else{

      id = id + startIndex;              
      if(inDegree[id] >= d_apr[id]){                           // checking if a node is active or not
        d_active[id] = 1;
        int start = d_offset[id];
        int end = d_offset[id+1];
        if(start != end){
          while(start != end){
            int vertex = d_csrList[start++];                   // increamenting the indegree of the nodes connected to active nodes 
            atomicAdd(&inDegree[vertex], 1);
          }
        }
      }
    }
   }
 }       

// this kerenl deactivates the node if its left and right node are deactive
 __global__ void deactivateFunction(int *d_csrList, int *d_offset, int *d_active, int *inDegree,int numberOFNodes, int startIndex, int current){

   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id < numberOFNodes){
    id = id + startIndex;
    if((d_active[id]==1)&&((id-1)>=startIndex)&&((id+1)<=current)){
      if(d_active[id-1] == 0 && d_active[id+1] == 0){
        d_active[id] = 0;
        int start = d_offset[id];
        int end = d_offset[id+1];
        if(start != end){
          while(start != end){
            int vertex = d_csrList[start++];
            atomicAdd(&inDegree[vertex], -1);                   // decrementing the indegree of the nodes connected to this deactive node
          }
        }
      }
    }
     
   }
 }  

// this nodes kernel computes the number of active nodes present in every level
 __global__ void countActiveNodes(int *d_active, int *count, int numberOFNodes, int startIndex, int endIndex){


   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id < numberOFNodes){
     id += startIndex;
     if(d_active[id] == 1)
      atomicAdd(count, 1);                             
   }
 }
    
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/


int *end;                                // gpu pointer to store the max of every level
int *levelEnd;                           // array that stores the index of all level expect at the 0th location
int *max;                                // copy end to this variabel to store in levelEnd
int *inDegree;                           // array on gpu to store the active indegree of the vertices
int *d_active;                           // array on gpu to store if the vertices are active or not
int *count;                              // stores the number of active node on every level


// alloting memory on device
cudaMalloc(&end, sizeof(int));
cudaMalloc(&count, sizeof(int));
cudaMalloc(&inDegree, V * sizeof(int));
cudaMalloc(&d_active, V * sizeof(int));

// setting all the device allocated memory to 0
cudaMemset(end, 0, sizeof(int));
cudaMemset(count, 0, sizeof(int));
cudaMemset(inDegree, 0, V * sizeof(int));
cudaMemset(d_active, 0, V * sizeof(int));

// alloting memory on host
levelEnd = (int *)malloc((L+1) * sizeof(int));              // to store the end points of every level 0th level is an 
max = (int *)malloc(sizeof(int));


*max = 0;

levelEnd[0] = -1;
int a = 0;

// getting the first non-zero value of apr of a vertex
while(1){
  if(h_apr[a] == 0)
    a++;
  else
    break;
}

levelEnd[1] = a-1;

// finding the ending vertices of all the level of the given graph 
for(int i = 1; i < L; i++){
    int numberOFNodes = levelEnd[i] - levelEnd[i-1];                     // number of nodes at every level
    int block = ceil((float)numberOFNodes/1024);
    //printf("hi\n");
    findLevel<<<block, 1024>>>(d_csrList, d_offset, end, numberOFNodes, levelEnd[i-1], i);    // calling kernel with total threads equal to number of nodes 
    cudaMemcpy(max, end, sizeof(int), cudaMemcpyDeviceToHost);
    levelEnd[i+1] = *max;                                                                   // storing maximum of evey level
}

for(int i = 1; i < L+1; i++){

  int numberOFNodes = levelEnd[i] - levelEnd[i-1];                       // number of nodes at every level
  int block = ceil(float(numberOFNodes)/1024);
  if(i != L+1){                                                                 
  computesIndegree<<<block, 1024>>>(d_csrList, d_offset, d_active, d_apr, inDegree, numberOFNodes, levelEnd[i-1] + 1, levelEnd[i], levelEnd[i+1], i);     // launching kernel with total threads equal to number of nodes to calcuate indegree of the level and check if the nodes are active or not
  cudaDeviceSynchronize();
  }
  deactivateFunction<<<block, 1024>>>(d_csrList, d_offset, d_active, inDegree, numberOFNodes, levelEnd[i-1] + 1, levelEnd[i]);           // launching the kernel to deactivate the nodes specified in the second rule
  cudaDeviceSynchronize();  
}

*max = 0;                                           // stores the number of nodes active at a particular level
for(int i = 1; i < L+1; i++){
  int numberOFNodes = levelEnd[i] - levelEnd[i-1];
  int block = ceil(float(numberOFNodes)/1024);
  countActiveNodes<<<block, 1024>>>(d_active, count, numberOFNodes, levelEnd[i-1]+1, levelEnd[i]);          // lauching the kernel to and computing active nodes at each level
  cudaMemcpy(max, count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemset(count, 0, sizeof(int));
  h_activeVertex[i-1] = *max;
}

h_activeVertex[0] = a;

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
