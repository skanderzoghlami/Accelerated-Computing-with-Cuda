# Accelerated Computing With Cuda
This repository Contains my notes for parallel computing, it shows how to interact with a cuda enabled GPU.
* Hello.cu script shows the difference between a host and a device function by printing hello from a kernel and from a host function.
* First_parallel.cu shows how to run a kernel with 5 blocks and 5 threads per block.
* vectorAdd.cu shows a very nice way to add the content of two huge vectors by exploiting the GPU, it shows how to find the number of blocks based on a predefined number of threads per block and how to index the kernel to execute it parallely on different cores of the GPU.
* Fiestal Cypher App: is a complete application that does coding/decoding with hashes, we really don't care how Encryption/Decryption is happening, what matters is how we make the decrypt kernel runs on different parallel streams by transforming the data into chunks and running them in multiple non default streams thus achieving Copy Compute overlap.

# Notes
The notes folder is self explanatory it has notes I took while reading the materials, it might not make sense to the reader but it helps me remember what I did, kudos to you if you understand them 🥹🥹


## Cores, Blocks, Threads and other terminologies
The ugly picture below shows that the GPU has many SMs, (Streaming Multiprocessor), an SM has a set of cores, each core is capable of executing a thread and a thread is a set of instructions of code. VOILA !!
![image](https://github.com/user-attachments/assets/04ad2b83-ab1e-46d0-8d89-b8180f48b781)



## Concurrent Streams and Nsight profiling:

Nsight profiler is a tool developed by nvidia to profile a cuda C/C++ app and check if there're any bottlenecks for memory/time.
A Cuda Stream is defined as a set of instructions (can be a kernel for exemple), there're two different types, the default stream and the non default stream"S", 3 Rules to always keep in mind here:
- When the Default stream is running it blocks all other non default streams
- operations in a given stream run by order of launch
- there's no guarentee of any order of execution between any 2 different non default streams executing in parallel.
![image](https://github.com/user-attachments/assets/8ce7eb4f-8706-4ee1-ad00-a5ea81a43db0)

## Memory Prefetching to avoid page faults:
a Page fault signal happens when the program demands access to the memory but it doesn't have it yet (because cudaMallocManaged allocates virtual memory which is memory that isn't physically allocated yet but it's declared), when a page fault occurs the memory becomes phisically allocated to the device (CPU/GPU) that is demanding it
The page fault signal is SLOW and it's better if we can avoid it by telling the GPU or the CPU: Hey man listen, since am the guy who coded this I know that the data will be given to the GPU first to instead of waiting for the page fault signal just physically allocate it to the GPU please.
this is called memory prefetching 😏
![image](https://github.com/user-attachments/assets/a9c9df34-5081-4376-abe6-c2264f72b19d)

## Copy Compute Overlap
Like I said in the into, to achieve this, the data must be divided into chunks and for each chunk we do the async memory allocation + kernel run  + memory liberation in a seperate non default stream as the other chunks.
![image](https://github.com/user-attachments/assets/0a8db559-4274-4543-a092-6f57c2a8010d)

