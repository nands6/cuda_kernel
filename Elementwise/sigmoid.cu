#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>


void cpu_sigmoid(float *a,float *b,int n){
    for(int i=0;i<n;i++){
        b[i]=1/(1+expf(-a[i]));
    }

}

bool check(float *a,float *b,int n){
    for(int i=0;i<n;i++){
        if(abs(a[i]-b[i])>0.0005){
            printf("%d\n",i);
            return false;
        }
    }
    return true;
}


//naive版本
__global__ void gpu_sigmoidV1(float *a,float *b,int n){
    int tx=blockDim.x*blockIdx.x+threadIdx.x;
    if(tx<n){
        b[tx]=1/(1+expf(-a[tx]));
    }

}
//给每个线程增加工作量，例如1个线程处理4个元素
template <int thread_per_num>
__global__ void gpu_sigmoidV2(float *a,float *b,int n){
    int tx=(threadIdx.x+blockDim.x*blockIdx.x)*thread_per_num;
    if(tx<n){
        for(int i=0;i<thread_per_num;i++){
            b[tx+i]=1/(1+expf(-a[tx+i]));
        }
    }
}

//通过float4的形式
#define FETCH_FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void gpu_sigmoidV3(float *a,float *b,int n){
    int tx=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    float4 tmp_a,tmp_b;
    if(tx<n){
        tmp_a=FETCH_FLOAT4(a[tx]);
        tmp_b.x=1/(1+expf(-tmp_a.x));
        tmp_b.y=1/(1+expf(-tmp_a.y));
        tmp_b.z=1/(1+expf(-tmp_a.z));
        tmp_b.w=1/(1+expf(-tmp_a.w));
        FETCH_FLOAT4(b[tx])=tmp_b;
    }
}

void print_result(float *a,int n){
    for(int i=0;i<n;i++){
        printf("%f \n",a[i]);
    }
}



int main(){
    int n=1280;
    size_t n_bytes=n*sizeof(float);
    
    float *h_a=(float*)malloc(n_bytes);
    float *h_b=(float*)malloc(n_bytes);
    float *gpu_result=(float*)malloc(n_bytes);
    for(int i=0;i<n;i++){
        h_a[i]=1;
        h_b[i]=0;
    }
    

    float *d_a,*d_b;
    cudaMalloc((void**)&d_a,n_bytes);
    cudaMalloc((void**)&d_b,n_bytes);
    cudaMemcpy(d_a,h_a,n_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,n_bytes,cudaMemcpyHostToDevice);

    const int thread_per_num=4;
    int block_size=32;
    int grid_size=(n+block_size-1)/block_size/thread_per_num;
 
    dim3 block(block_size);
    dim3 grid(grid_size);

   
    cpu_sigmoid(h_a,h_b,n);
    // print_result(h_b,n);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // gpu_sigmoidV1<<<grid,block>>>(d_a,d_b,n);
    // gpu_sigmoidV2<thread_per_num><<<grid,block>>>(d_a,d_b,n);
    gpu_sigmoidV3<<<grid,block>>>(d_a,d_b,n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU时间: %f ms\n", gpu_time_ms);

    cudaMemcpy(gpu_result,d_b,n_bytes,cudaMemcpyDeviceToHost);
    // for(int i=0;i<n;i++){
    //     printf("%d %f\n",i,gpu_result[i]);
    // }
    if(check(h_b,gpu_result,n)){
        printf("结果正确\n");
    }else{
        printf("错误\n");
       
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(gpu_result);

    return 0;
}