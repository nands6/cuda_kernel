#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cstdio>

void cpu_add(float *a,float *b,float *c,int n){
    for(int i=0;i<n;i++){
        c[i]=a[i]+b[i];
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
__global__ void gpu_addV1(float *a,float *b,float *c,int n){
    int tx=(threadIdx.x+blockDim.x*blockIdx.x);
    if(tx<n){
        c[tx]=a[tx]+b[tx];
    }

}

//给每个线程增加工作量，例如1个线程处理4个元素
template <int thread_per_num>
__global__ void gpu_addV2(float *a,float *b,float *c,int n){
    int tx=(threadIdx.x+blockDim.x*blockIdx.x)*thread_per_num;
    if(tx<n){
        for(int i=0;i<thread_per_num;i++){
            c[tx+i]=a[tx+i]+b[tx+i];
        }
    }
}

//通过float4的形式
#define FETCH_FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
__global__ void gpu_addV3(float *a,float *b,float *c,int n){
    int tx=(threadIdx.x+blockDim.x*blockIdx.x)*4;
    float4 tmp_a,tmp_b,tmp_c;
    if(tx<n){
        tmp_a=FETCH_FLOAT4(a[tx]);
        tmp_b=FETCH_FLOAT4(b[tx]);
        tmp_c.x=tmp_a.x+tmp_b.x;
        tmp_c.y=tmp_a.y+tmp_b.y;
        tmp_c.z=tmp_a.z+tmp_b.z;
        tmp_c.w=tmp_a.w+tmp_b.w;
        FETCH_FLOAT4(c[tx])=tmp_c;
    }
}


int main(){
    int n=1024000;
    size_t n_bytes=sizeof(float)*n;
    float *h_a=(float*)malloc(n_bytes);
    float *h_b=(float*)malloc(n_bytes);
    float *h_c=(float*)malloc(n_bytes);
    float *gpu_result=(float*)malloc(n_bytes);
    for(int i=0;i<n;i++){
        h_a[i]=1;
        h_b[i]=2;
    }
    memset(h_c,0,n_bytes);

    

    float *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,n_bytes);
    cudaMalloc((void**)&d_b,n_bytes);
    cudaMalloc((void**)&d_c,n_bytes);

    cudaMemset(d_a,0,n_bytes);
    cudaMemset(d_b,0,n_bytes);
    cudaMemset(d_c,0,n_bytes);

    cudaMemcpy(d_a,h_a,n_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,n_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c,n_bytes,cudaMemcpyHostToDevice);
    const int thread_per_num=4;
    int block_size=1024;
    int grid_size=(n+block_size-1)/block_size/thread_per_num;
    dim3 block(block_size);
    dim3 grid(grid_size);

    cpu_add(h_a,h_b,h_c,n);
    
   
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // gpu_addV1<<<grid,block>>>(d_a,d_b,d_c,n);
    // gpu_addV2<thread_per_num><<<grid,block>>>(d_a,d_b,d_c,n);
    gpu_addV3<<<grid,block>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU时间: %f ms\n", gpu_time_ms);
    cudaMemcpy(gpu_result,d_c,n_bytes,cudaMemcpyDeviceToHost);
    if(check(h_c,gpu_result,n)){
        printf("结果正确\n");
    }else{
        printf("错误\n");
       
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
   
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(gpu_result);

    return 0;
}