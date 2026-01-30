#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

void cpu_softmax(float *out, const float *inp, int N, int C) {
    for (int i = 0; i < N; i++) {
      const float *inp_row = inp + i * C;
      float *out_row = out + i * C;
  
      float maxval = -INFINITY;
      for (int j = 0; j < C; j++) {
        if (inp_row[j] > maxval) {
          maxval = inp_row[j];
        }
      }
      float sum = 0.f;
      for (int j = 0; j < C; j++) {
        out_row[j] = expf(inp_row[j] - maxval);
        sum += out_row[j];
      }
      float norm = 1.f / sum;
      for (int j = 0; j < C; j++) {
        out_row[j] *= norm;
      }
    }
}




bool check(float *a,float* b,int n,int c){
    for(int i=0;i<n;i++){
        for(int j=0;j<c;j++){
            if(abs(a[i*c+j]-b[i*c+j])>0.005){
                printf("位置:%d\n",i);
                return false;
            }
        }
    }
    return true;
}

void print_result(float *inp,int N,int C){
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
          printf("%f ",inp[n * C + c]);
        }
        printf("\n");
    }
    printf("\n");
}


//naive版本，一个线程处理一行
__global__ void gpu_softmaxV1(float *inp,float *out,int N,int C){
    
    float *begin_inp=inp+blockIdx.x*C;
    float *begin_out=out+blockIdx.x*C;

    float max=-INFINITY;
    for(int i=0;i<C;i++){
        if(begin_inp[i]>max){
            max=begin_inp[i];
        }
    }
    
    float sum=0;
    for(int i=0;i<C;i++){
        begin_out[i]=expf(begin_inp[i]-max);
        sum=sum+begin_out[i];
    }

    for(int i=0;i<C;i++){
        begin_out[i]=begin_out[i]/sum;
    }

}

//一个线程处理多个元素
template<int BLOCK>
__global__ void gpu_softmaxV2(float *inp,float *out,int N,int C){
    float *begin_inp=inp+blockIdx.x*C;
    float *begin_out=out+blockIdx.x*C;
    __shared__ float shared[BLOCK];
    // extern __shared__ float shared[];

    float max=-INFINITY;
    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        max=fmaxf(max,begin_inp[i]);
    }    
    shared[threadIdx.x]=max;
    __syncthreads();
    
    for(int i=blockDim.x/2;i>0;i=i/2){
        if(threadIdx.x<i){
            shared[threadIdx.x]=fmaxf(shared[threadIdx.x],shared[threadIdx.x+i]);
        }
        __syncthreads();
    }
    __syncthreads();
    max=shared[0];
   
    float sum=0;
    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        begin_out[i]=expf(begin_out[i]-max);
        sum=sum+begin_out[i];
    }    
    shared[threadIdx.x]=sum;
    __syncthreads();

    for(int i=blockDim.x/2;i>0;i=i/2){
        if(threadIdx.x<i){
            shared[threadIdx.x]=shared[threadIdx.x]+shared[threadIdx.x+i];
        }
        __syncthreads();
    }
    __syncthreads();
    sum=shared[0];

    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        begin_out[i]=begin_out[i]/sum;
    }    
}

//wrap shuffle版本
__device__ float wrapreduceMax(float val){
    
    for(int i=16;i>0;i=i/2){
        val=fmaxf(val,__shfl_down_sync(0xffffffff,val,i));
    }
    return val;
}
__device__ float wrapreduceSum(float val){
    
    for(int i=16;i>0;i=i/2){
        val=val+__shfl_down_sync(0xffffffff,val,i);
    }
    return val;
}

__global__ void gpu_softmaxV3(float *inp,float *out,int N,int C){
    float *begin_inp=inp+blockIdx.x*C;
    float *begin_out=out+blockIdx.x*C;
    int wrapnum=blockDim.x/32;
    int wrapid=threadIdx.x/32;
    int laneid=threadIdx.x%32;
    extern __shared__ float shared[];

    float max=-INFINITY;
    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        max=fmaxf(max,begin_inp[i]);
    }    
    __syncthreads();
    max=wrapreduceMax(max);
    // for(int i=16;i>0;i=i/2){
    //     max=fmaxf(max,__shfl_down_sync(0xffffffff,max,i));
    // }
    if(laneid==0){
        shared[wrapid]=max;
    }
    __syncthreads();
    
    if(wrapid==0){
        if(laneid<wrapnum){
            max=shared[laneid];
        }else{
            max=-INFINITY;
        }
        max=wrapreduceMax(max);
        // for(int i=16;i>0;i=i/2){
        //     max=fmaxf(max,__shfl_down_sync(0xffffffff,max,i));
        // }
        shared[0]=max;
    }
    __syncthreads();
    max=shared[0];
    
    float sum=0;
    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        begin_out[i]=expf(begin_inp[i]-max);
        sum=sum+begin_out[i];
    }
    __syncthreads();
    sum=wrapreduceSum(sum);
    // for(int i=16;i>0;i=i/2){
    //     sum=sum+__shfl_down_sync(0xffffffff,sum,i);
    // }
    if(laneid==0){
        shared[wrapid]=sum;
    }
    __syncthreads();
    if(wrapid==0){
        if(laneid<wrapnum){
            sum=shared[laneid];
        }else{
            sum=0;
        }
        sum=wrapreduceSum(sum);
        // for(int i=16;i>0;i=i/2){
        //     sum=sum+__shfl_down_sync(0xffffffff,sum,i);
        // }
        shared[0]=sum;
    }
    __syncthreads();
    sum=shared[0];

    for(int i=threadIdx.x;i<C;i=i+blockDim.x){
        begin_out[i]=begin_out[i]/sum;
    }

}




int main(){

    int N = 20;
    int C = 1280;
    size_t inByteCount=N*C*sizeof(float);
    float *inp = (float *)malloc(inByteCount);
    float *cpu_result = (float *)malloc(inByteCount);
    float *gpu_result = (float *)malloc(inByteCount);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
          inp[n * C + c] = float(c+1);
        }
    }

    int grid_size=N;
    const int block_size=64;
   
    float *device_inp,*device_out;
    cudaMalloc((float**)&device_inp,inByteCount);
    cudaMalloc((float**)&device_out,inByteCount);
    cudaMemset(device_inp,0,inByteCount);
    cudaMemset(device_out,0,inByteCount);
    cudaMemcpy(device_inp,inp,inByteCount,cudaMemcpyHostToDevice);
    cpu_softmax(cpu_result,inp,N,C);
    // print_result(cpu_result,N,C);


    struct timeval t1, t2;
    double time_cuda_pre = 0;
   
    gettimeofday(&t1, NULL);

    //gpu_softmaxV1<<<grid_size,block_size>>>(device_inp,device_out,N,C);
    // gpu_softmaxV2<block_size><<<grid_size,block_size>>>(device_inp,device_out,N,C);
    gpu_softmaxV3<<<grid_size,block_size>>>(device_inp,device_out,N,C);
    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    time_cuda_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("时间:%fms\n",time_cuda_pre);

    cudaMemcpy(gpu_result,device_out,inByteCount,cudaMemcpyDeviceToHost);
     //print_result(gpu_result,N,C);
    if(check(cpu_result,gpu_result,N,C)){
        printf("正确\n");
    }else{
        printf("错误\n");
    }

    return 0;
}