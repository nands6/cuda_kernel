#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>

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

void cpu_gemm(float *a,float *b,float *c,int m,int n,int k){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum=0;
            for(int x=0;x<k;x++){
                sum+=a[i*k+x]*b[x*n+j];
            }
            c[i*n+j]=sum;
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

void InitMatrix(float *a,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            a[i*n+j]=j;
        }
    }
}


//naive版本 一个线程处理一个数据
__global__ void gpu_gemmV1(float *a,float *b,float *c,int m,int n,int k){
    int bx=blockDim.x*blockIdx.x;
    int by=blockDim.y*blockIdx.y;
    int tx=bx+threadIdx.x;
    int ty=by+threadIdx.y;

    if(tx<n&&ty<m){
        float sum=0;
        for(int i=0;i<k;i++){
            sum+=a[ty*k+i]*b[i*n+tx];
        }
        c[ty*n+tx]=sum;
    }

}

//使用共享内存，滑动窗口的形式
template<int BM,int BN,int BK>
__global__ void gpu_gemmV2(float *a,float *b,float *c,int m,int n,int k){
    int bx=blockDim.x*blockIdx.x;
    int by=blockDim.y*blockIdx.y;
    int tx=bx+threadIdx.x;
    int ty=by+threadIdx.y;
    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];
    float *begin_c=c+by*n+bx;
    float *begin_a=a+by*k;
    float *begin_b=b+bx;
    float sum=0;
    if(tx<n&&ty<m){
        for(int s=0;s<k;s=s+BK){
            a_shared[threadIdx.y][threadIdx.x]=begin_a[threadIdx.y*k+s+threadIdx.x];  //a[ty*n+i+threadIdx.x]
            b_shared[threadIdx.y][threadIdx.x]=begin_b[s*n+threadIdx.y*n+threadIdx.x];
            __syncthreads();
            for(int i=0;i<BK;i++){
                sum=sum+a_shared[threadIdx.y][i]*b_shared[i][threadIdx.x];
            }
            __syncthreads();
        }
        begin_c[threadIdx.y*n+threadIdx.x]=sum;
    }
}

//当前访存是瓶颈，增加每个线程的工作量
template<int BM,int BN,int BK,int STRIDE>
__global__ void gpu_gemmV3(float *a,float *b,float *c,int m,int n,int k){
    int STEP=blockDim.x*STRIDE;
    int bx=STEP*blockIdx.x;
    int by=STEP*blockIdx.y;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float *begin_c=c+by*n+bx;
    float *begin_a=a+by*k;
    float *begin_b=b+bx;
    float tmp[STRIDE][STRIDE]={0};
    __shared__ float a_shared[BM*STRIDE][BK*STRIDE];
    __shared__ float b_shared[BK*STRIDE][BN*STRIDE];

    for(int s=0;s<k;s=s+STEP){
        for(int i=0;i<STRIDE;i++){
            for(int j=0;j<STRIDE;j++){
                a_shared[ty+i*blockDim.y][tx+j*blockDim.x]=begin_a[ty*k+i*blockDim.y*k+tx+j*blockDim.x+s];
                b_shared[ty+i*blockDim.y][tx+j*blockDim.x]=begin_b[n*(ty+s+i*blockDim.y)+tx+j*blockDim.x];
            }
        }
        __syncthreads();
        for(int i=0;i<STRIDE;i++){
            for(int j=0;j<STRIDE;j++){
                for(int x=0;x<STEP;x++){
                    tmp[i][j]+=a_shared[ty+i*blockDim.y][x]*b_shared[x][tx+j*blockDim.x];
                }
                
            }
        }
        __syncthreads();
    }
    
    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            begin_c[ty*n+i*blockDim.y*n+tx+j*blockDim.x]=tmp[i][j];
        }
    }

}

#define FETCH_float4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
template <int M_NUM_PER_BLOCK,
int N_NUM_PER_BLOCK,
int K_NUM_PER_BLOCK,
int  NUM_PER_THREAD>
__global__ void gpu_gemmv4(float *A, float *B, float *C, int m, int n, int k)
{
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    float *begin_A=A+blockIdx.y*k*M_NUM_PER_BLOCK;
    float *begin_B=B+blockIdx.x*N_NUM_PER_BLOCK;
    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD]={0};
    
#pragma unroll
    for(int s=0;s<k;s=s+K_NUM_PER_BLOCK){
        FETCH_float4(a_shared[ty][tx*NUM_PER_THREAD])= FETCH_float4(begin_A[ty*k+tx*NUM_PER_THREAD+s]);
        // a_shared[ty][tx*NUM_PER_THREAD]=begin_A[ty*k+tx*NUM_PER_THREAD+s];
        // a_shared[ty][tx*NUM_PER_THREAD+1]=begin_A[ty*k+tx*NUM_PER_THREAD+s+1];
        // a_shared[ty][tx*NUM_PER_THREAD+2]=begin_A[ty*k+tx*NUM_PER_THREAD+s+2];
        // a_shared[ty][tx*NUM_PER_THREAD+3]=begin_A[ty*k+tx*NUM_PER_THREAD+s+3];
        FETCH_float4(b_shared[ty][tx*NUM_PER_THREAD])= FETCH_float4(begin_B[ty*n+s*n+tx*NUM_PER_THREAD]);
        // b_shared[ty][tx*NUM_PER_THREAD]=begin_B[ty*n+s*n+tx*NUM_PER_THREAD];
        // b_shared[ty][tx*NUM_PER_THREAD+1]=begin_B[ty*n+s*n+tx*NUM_PER_THREAD+1];
        // b_shared[ty][tx*NUM_PER_THREAD+2]=begin_B[ty*n+s*n+tx*NUM_PER_THREAD+2];
        // b_shared[ty][tx*NUM_PER_THREAD+3]=begin_B[ty*n+s*n+tx*NUM_PER_THREAD+3];
        __syncthreads();
        #pragma unroll
        for(int i=0;i<NUM_PER_THREAD;i++){
            #pragma unroll
            for(int j=0;j<K_NUM_PER_BLOCK;j++){
                temp[i]+=a_shared[ty][j]*b_shared[j][tx*NUM_PER_THREAD+i];
            }
        }
        __syncthreads();
    }

    float *begin_C=C+blockIdx.y*M_NUM_PER_BLOCK*n+blockIdx.x*N_NUM_PER_BLOCK;
    #pragma unroll
    for(int i=0;i<NUM_PER_THREAD;i++){
        begin_C[ty*n+tx*NUM_PER_THREAD+i]=temp[i];
    }


}

int main(){
    int m=1024;
    int n=1024;
    int k=1024;
    float *h_a=(float*)malloc(sizeof(float)*m*k);
    float *h_b=(float*)malloc(sizeof(float)*k*n);
    float *h_c=(float*)malloc(sizeof(float)*m*n);
    float *gpu_c=(float*)malloc(sizeof(float)*m*n);
    InitMatrix(h_a,m,k);
    InitMatrix(h_b,k,n);
    // print_result(h_a,m,k);
    // print_result(h_b,k,n);
    cpu_gemm(h_a,h_b,h_c,m,n,k);

    // print_result(h_c,m,n);

    float *device_a,*device_b,*device_c;
    cudaMalloc((void**)&device_a,m*k*sizeof(float));
    cudaMalloc((void**)&device_b,k*n*sizeof(float));
    cudaMalloc((void**)&device_c,m*n*sizeof(float));
    cudaMemcpy(device_a,h_a,m*k*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(device_b,h_b,k*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(device_c,0,m*n*sizeof(float));
    
    const int BM=16;
    const int BN=16;
    const int BK=16;
    const int STRIDE=2;
    dim3 block(16,16);
    dim3 grid((n+block.x-1)/block.x/STRIDE,(m+block.y-1)/block.y/STRIDE);
    

    struct timeval t1, t2;
    double time_cuda_pre = 0;
    gettimeofday(&t1, NULL);

    //gpu_gemmV1<<<grid,block>>>(device_a,device_b,device_c,m,n,k);
    //gpu_gemmV2<BM,BN,BK><<<grid,block>>>(device_a,device_b,device_c,m,n,k);
    //gpu_gemmV3<BM,BN,BK,STRIDE><<<grid,block>>>(device_a,device_b,device_c,m,n,k);
    
    // constexpr int M_NUM_PER_BLOCK = 32;
    // constexpr int N_NUM_PER_BLOCK = 32;
    // constexpr int K_NUM_PER_BLOCK = 32;
    // constexpr int NUM_PER_THREAD = 4;
    // dim3 block(8, 32);
    // dim3 grid(n / N_NUM_PER_BLOCK, m / M_NUM_PER_BLOCK);
    // gpu_gemmv4<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK,K_NUM_PER_BLOCK,NUM_PER_THREAD>
    // <<<grid, block>>>(device_a, device_b, device_c, m, n, k);

    cudaDeviceSynchronize();

    gettimeofday(&t2, NULL);
    time_cuda_pre = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("时间:%fms\n",time_cuda_pre);

    cudaMemcpy(gpu_c,device_c,m*n*sizeof(float),cudaMemcpyDeviceToHost);

    if(check(h_c,gpu_c,m,n)){
        printf("正确\n");
    }else{
        printf("错误\n");
    }
    free(gpu_c);
    return 0;
}