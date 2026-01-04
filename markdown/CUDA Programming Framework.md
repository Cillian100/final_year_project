- CUDA is a programming computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. 

- The core of CUDA is based on three abstractions exposed to the programmer: 
	- a hierarchy of thread groups
	- shared memories 
	- barrier synchronisation
- they allow the programmer to partition the problem into coarse sub-problems solved by thread groups in parallel, and each sub-problem is further partitioned into chunks that are solved by threads within a thread group. 

- The GPU started out as a fixed-function hardware to accelerate parallel operations in real-time 3D rendering.
- GPUs and CPUs are desgined with different goals in mind.
	- a CPU is designed to excel at executing a serial sequence of operations as fast as possible and can execute a few tens of these in parallel.
	- a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single thread performance to achieve much greater total throughput 

- CUDA would need to be paired with another tool, such as OpenMP or Pthreads, to utilise the host CPU in a hybrid application.  
### Heterogeneous Systems
- CUDA assumes a heterogeneous system, which means a system that includes both CPUs and GPUs
	- the CPU and the memory directly connected to it are called the host and host memory respectively 
	- the GPU and the memory directly connected to it are called the device and device memory respectively 

- CUDA applications execute some part of their code on the GPU, but applications always start execution on the CPU.
- The CPU and GPU can both be executing code simultaneously, but best performance is usually found by maximising utilisation of both CPU and GPU
- The code an application executes on a GPU is referred to as device code, and a function that is invoked for execution on the GPU is called to kernel.
	- the act of starting a kernel running is called launching a kernel 