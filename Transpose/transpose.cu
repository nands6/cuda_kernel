#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>
#include <cstdio>

bool check(float *a,float* b,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(abs(a[i*n+j]-b[i*n+j])>0.005){
                printf("位置:%d\n",i*n+j);
                return false;
            }
        }
    }
    return true;
}

void cpu_transpose(float *a,float *b,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            b[j*m+i]=a[i*n+j];
        }
    }
}

void print_result(float *inp,int M,int N){
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          printf("%f ",inp[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

//naive版本 一个线程处理一个数据
__global__ void gpu_transeposeV1(float *inp,float *out,int m,int n){
    
    int tx=blockDim.x*blockIdx.x+threadIdx.x;
    int ty=blockDim.y*blockIdx.y+threadIdx.y;
    if(tx<n&&ty<m){
        // out[tx*m+ty]=inp[ty*n+tx];   //读取是合并访问的，但是写入不是
        out[ty*m+tx]=inp[tx*n+ty];   //写入是合并访问的，但是读取不是
        // out[ty*m+tx]=__ldg(&inp[tx*n+ty]);  //与第二个一样，目前的架构一般不需要__ldg,因为能够自动判断并优化
    }
}


//共享内存版
template<int BLOCK>
__global__ void gpu_transeposeV2(float *inp,float *out,int m,int n){
    __shared__ float shared[BLOCK][BLOCK];
    int bx=blockDim.x*blockIdx.x;
    int by=blockDim.y*blockIdx.y;

    int tx=threadIdx.x+bx;
    int ty=threadIdx.y+by;
    float *begin_inp=inp+by*n+bx;
    if(tx<n&&ty<m){
        shared[threadIdx.y][threadIdx.x]=begin_inp[threadIdx.y*n+threadIdx.x];
    }
    __syncthreads();

    int tx2=by+threadIdx.x;
    int ty2=bx+threadIdx.y;
    float *begin_out=out+bx*m+by;
    if(tx2<m&&ty2<n){
        begin_out[threadIdx.y*m+threadIdx.x]=shared[threadIdx.x][threadIdx.y];
    }

}

int main(){
    int m=1280;
    int n=1280;
    float *a=(float*)malloc(sizeof(float)*m*n);
    float *cpu_result=(float*)malloc(sizeof(float)*m*n);
    float *gpu_result=(float*)malloc(sizeof(float)*m*n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a[i*n+j]=j+1;
        }
    }
   
    cpu_transpose(a,cpu_result,m,n);
   
    float *device_a,*device_b;
    cudaMalloc((void**)&device_a,sizeof(float)*m*n);
    cudaMalloc((void**)&device_b,sizeof(float)*m*n);
    cudaMemcpy(device_a,a,sizeof(float)*m*n,cudaMemcpyHostToDevice);
    cudaMemset(device_b,0,sizeof(float)*m*n);

    dim3 block(32,32);
    dim3 grid((n+block.x-1)/block.x,(m+block.y-1)/block.y);

    struct timeval t1, t2;
    double time_cuda_pre = 0;
   
    gettimeofday(&t1, NULL);

    // gpu_transeposeV1<<<grid,block>>>(device_a,device_b,m,n);
    gpu_transeposeV2<32><<<grid,block>>>(device_a,device_b,m,n);
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    time_cuda_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("时间:%fms\n",time_cuda_pre);

    cudaMemcpy(gpu_result,device_b,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
   
    if(check(cpu_result,gpu_result,m,n)){
        printf("正确\n");
    }else{
        printf("错误\n");
    }

    free(a);
    free(cpu_result);
    free(gpu_result);
    cudaFree(device_a);
    cudaFree(device_b);


}