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
        if(abs(a[i]-b[i])>0.005){
            printf("位置:%d\n",i);
            return false;
        }
    }
    return true;
}

//naive版本  偶数线程执行
__global__ void reduce_sumV1(float* input,float *output){
    int tx=threadIdx.x;
    extern __shared__ float shared[];
    float *begin_input=input+blockDim.x*blockIdx.x;
    shared[tx]=begin_input[tx];
    __syncthreads();

    for(int i=1;i<blockDim.x;i=i*2){
        if(tx%(i*2)==0){
            shared[tx]+=shared[tx+i];
        }
        __syncthreads();
    }
    if(tx==0){
        output[blockIdx.x]=shared[tx];
    }

}


//折半规约   前blockDim.x/2个线程执行
__global__ void reduce_sumV2(float* input,float *output){
    int tx=threadIdx.x;
    extern __shared__ float shared[];
    
    float *begin_input=input+blockDim.x*blockIdx.x;
    shared[tx]=begin_input[tx];
    __syncthreads();
    for(int i=blockDim.x/2;i>0;i=i/2){
        if(tx<i){
            shared[tx]+=shared[tx+i];
        }
        __syncthreads();
    }

    if(tx==0){
        output[blockIdx.x]=shared[tx];
        // atomicAdd(output,shared[0]);
    }

}



__device__ float warpreduceSum(float val){
    
    for(int i=16;i>0;i=i/2){
        val=val+__shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}
//warp shuffle   在warp内规约，避免同步
__global__ void reduce_sumV3(float* input,float *output){
    int tx=blockDim.x*blockIdx.x+threadIdx.x;
    int warpnum=blockDim.x/32;
    int warpid=threadIdx.x/32;
    int laneid=threadIdx.x%32;
    extern __shared__ float shared[];
    

    float sum=input[tx];
    sum=warpreduceSum(sum);
    // for(int i=16;i>0;i=i/2){
    //     sum+=__shfl_down_sync(0xffffffff,sum,i);
    // }
    if(laneid==0){
        shared[warpid]=sum;
    }
    __syncthreads();

    if(warpid==0){
        if(laneid<warpnum){
            sum=shared[laneid];
        }else{
            sum=0;
        }
        // for(int i=16;i>0;i=i/2){
        //     sum+=__shfl_down_sync(0xffffffff,sum,i);
        // }
        sum=warpreduceSum(sum);
    }
    if(threadIdx.x==0){
       output[blockIdx.x]=sum;
    }
}



int main(){

    int n=1280000;
    size_t n_bytes=sizeof(float)*n;
    float *a=(float*)malloc(n_bytes);
    for(int i=0;i<n;i++){
        a[i]=1;
    }
    const int block_size=64;
    int grid_size=(n+block_size-1)/block_size;
    float *cpu_result=(float*)malloc(grid_size*sizeof(float));

    for(int i=0;i<grid_size;i++){
        float sum=0;
        for(int j=0;j<block_size;j++){
            sum=sum+a[i*block_size+j];
        }
        cpu_result[i]=sum;
    }
    // print_result(cpu_result,grid_size);

    float *d_a,*d_b,*gpu_result;
    gpu_result=(float*)malloc(sizeof(float)*grid_size);
    cudaMalloc((void**)&d_a,n_bytes);
    cudaMalloc((void**)&d_b,sizeof(float)*grid_size);
    cudaMemcpy(d_a,a,n_bytes,cudaMemcpyHostToDevice);
    struct timeval t1, t2;
    double time_cuda_pre = 0;
   
    gettimeofday(&t1, NULL);
    // reduce_sumV1<<<grid_size,block_size>>>(d_a,d_b);
    // reduce_sumV2<<<grid_size,block_size>>>(d_a,d_b);
    reduce_sumV3<<<grid_size,block_size>>>(d_a,d_b);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_cuda_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("时间:%fms\n",time_cuda_pre);
    
    

    cudaMemcpy(gpu_result,d_b,sizeof(float)*grid_size,cudaMemcpyDeviceToHost);
    // print_result(gpu_result,grid_size);

    if(check(cpu_result,gpu_result,grid_size)){
        printf("正确\n");
    }else{
        printf("错误\n");
    }


    return 0;
}