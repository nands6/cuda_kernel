#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

void print_result(float *a,int n){
    for(int i=0;i<n;i++){
        printf("%f \n",a[i]);
    }
}

bool check(float *a,float* b,int n){
    for(int i=0;i<n;i++){
        if(abs(a[i]-b[i])>0.00005){
            printf("位置:%d\n",i);
            return false;
        }
    }
    return true;
}

void cpu_gemv(float *a,float *b,float *c,int m,int n){
    for(int i=0;i<m;i++){
        float sum=0;
        for(int j=0;j<n;j++){
            sum+=a[i*n+j]*b[j];
        }
        c[i]=sum;
    }
}

//naive版本，一个线程处理一行数据
__global__ void gpu_gemvV1(float *a,float *b,float *c,int m,int n){
    float *begin_a=a+blockIdx.x*n;
    float sum=0;
    for(int i=0;i<n;i++){
        sum+=begin_a[i]*b[i];
    }
    c[blockIdx.x]=sum;
    
}

//一个线程处理多个数据
__global__ void gpu_gemvV2(float *a,float *b,float *c,int m,int n){
    int tx=threadIdx.x;
    extern __shared__ float shared[];
    float *begin_a=a+blockIdx.x*n;
    float sum=0;
    for(int i=tx;i<n;i=i+blockDim.x){
        sum=sum+begin_a[i]*b[i];
    }
    shared[tx]=sum;
    __syncthreads();
    for(int i=blockDim.x/2;i>0;i=i/2){
        if(tx<i){
            shared[tx]+=shared[tx+i];
        }
        __syncthreads();
    }
    sum=shared[0];
    if(threadIdx.x==0){
        c[blockIdx.x]=sum;
    }
    

}

//warp shuffle
__global__ void gpu_gemvV3(float *a,float *b,float *c,int m,int n){
    int tx=threadIdx.x;
    int warpnum=blockDim.x/32;
    int warpid=threadIdx.x/32;
    int laneid=threadIdx.x%32;
    extern __shared__ float shared[];
    float *begin_a=a+blockIdx.x*n;
    float sum=0;
    for(int i=tx;i<n;i=i+blockDim.x){
        sum=sum+begin_a[i]*b[i];
    }
    // shared[tx]=sum;
    __syncthreads();
    for(int i=16;i>0;i=i/2){
        sum=sum+__shfl_down_sync(0xffffffff,sum,i);
    }
    if(laneid==0){
        shared[warpid]=sum;
    }
    __syncthreads();
    //单个线程规约每个warp的值
    // float tmp=0;
    // if(threadIdx.x==0){
    //     for(int i=0;i<warpnum;i++){
    //         tmp=tmp+shared[i];
    //     }
    // }
    // __syncthreads();
    // if(threadIdx.x==0){
    //     c[blockIdx.x]=tmp;
    // }
    
    //让第一个warp继续规约所有warp的值
    if(warpid==0){
        sum=shared[laneid];
        for(int i=warpnum/2;i>0;i=i/2){
            sum=sum+__shfl_down_sync(0xffffffff,sum,i);
        }
    }
    __syncthreads();
    if(threadIdx.x==0){
        c[blockIdx.x]=sum;
    }
}
int main(){
    int m=128;
    int n=12800;
    float *a=(float*)malloc(sizeof(float)*m*n);
    float *b=(float*)malloc(sizeof(float)*n);
    float *cpu_result=(float*)malloc(sizeof(float)*m);
    float *gpu_result=(float*)malloc(sizeof(float)*m);

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a[i*n+j]=j*0.001;
            b[j]=1;
        }
    }


    float *device_a,*device_b,*device_c;
    cudaMalloc((void**)&device_a,sizeof(float)*m*n);
    cudaMalloc((void**)&device_b,sizeof(float)*n);
    cudaMalloc((void**)&device_c,sizeof(float)*m);
    cudaMemcpy(device_a,a,sizeof(float)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(device_b,b,sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemset(device_c,0,m*sizeof(float));

    int block_size=256;
    int grid_size=m;
    struct timeval t1, t2;
    double time_cuda_pre = 0;
   
    gettimeofday(&t1, NULL);
    //gpu_gemvV1<<<grid_size,block_size>>>(device_a,device_b,device_c,m,n);
    //gpu_gemvV2<<<grid_size,block_size>>>(device_a,device_b,device_c,m,n);
    gpu_gemvV3<<<grid_size,block_size>>>(device_a,device_b,device_c,m,n);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_cuda_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("时间:%fms\n",time_cuda_pre);

    cudaMemcpy(gpu_result,device_c,sizeof(float)*m,cudaMemcpyDeviceToHost);


    cpu_gemv(a,b,cpu_result,m,n);
    if(check(cpu_result,gpu_result,m)){
        printf("正确\n");
    }else{
        printf("错误\n");
    }
    

    return 0;
}