# HW1: Optimizing Matrix Multiplication

Details on the assignment can be found on CS 5220 Canvas website.

## Version Control

This is a Git repository. 

To use SSH to connect to GitHub,
[follow these instructions](https://help.github.com/en/github/using-git/which-remote-url-should-i-use#cloning-with-ssh-urls).

Every student must create their own PRIVATE GitHub repo to track the changes. One can find a simple instruction here https://junyonglee.me/github/How-to-make-forked-private-repository/ on how to make a private repo that can pull new features from the course public repo. 

The staff members must be given access to every student repository. The staff usernames are: giuliaguidi, XinchenXu, and xinranzhu. When you create your private repo, in the repo settings, find category "Collaborators" and all three staff members. 

## Configuration

Please [follow these instructions](https://github.com/CS5220-SP23/CLASS/tree/main/perlmutter-tutorial) to ensure your module configuration is correct to run the build system.

## Build System

This assignment uses [CMake](https://cmake.org/) to provide a consistent build system.
You should not need to modify the provided build in any way (CMakeLists.txt).
This document describes the basic process for configuring and building the code.

First, note that this file is in the _source directory_.
You will run CMake commands from the _build directory_, which you create by running

```
student@login04:~/hw1> mkdir build
student@login04:~/hw1> cd build
```

## Build Configuration

From this _build directory_, it is now possible to _configure_ the build.
The basic way to do this is:

```
student@login04:~/hw1-starter/build> cmake -DCMAKE_BUILD_TYPE=Release ..
```

This command tells CMake to generate the build files for HW1 in _Release_ mode.
The syntax `-D[VAR]=[VAL]` allows you to set a variable.
Only `CMAKE_BUILD_TYPE` is required, though there are more variables that you might want to change:

1. `CMAKE_BUILD_TYPE` -- this is either `Debug` or `Release`.
2. `CMAKE_C_FLAGS` -- this allows you to specify additional compiler flags.
3. `MAX_SPEED` -- this should be equal to the maximum number of gigaflops-per-second (GF/s) your processor can execute.
It is set to 56 by default, which matches Perlmutter's processors.
4. `TEAM_NO` -- when you are ready to submit your assignment, set this to be your **two-digit** team number.
5. `ALL_SIZES` -- set to `ON` to test against a large set of matrix sizes. `OFF` by default.

When you build in Debug mode, optimizations are disabled.
Yet when writing parallel code, it is often the case that problems arise only when optimizations are enabled.
You can recover debugging symbols in Release mode (for use with `gdb`) by running:

```
student@login04:~/hw1/build> cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-g3" ..
```

Similarly, you can enable optimizations in Debug mode by running:

```
student@login04:~/hw1/build> cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-O2" ..
```

## Compiling

Once your build is configured, you can actually compile by running `make` from the build directory.

```
student@login04:~/hw1-starter/build> make
```

This will produce several files:

```
student@login04:~/hw1-starter/build> ls
benchmark-blas     CMakeCache.txt       job-blas     Makefile
benchmark-blocked  CMakeFiles           job-blocked
benchmark-naive    cmake_install.cmake  job-naive
```

The executables `benchmark-blas`, `benchmark-blocked`, and `benchmark-naive` are the relevant ones here.
You can freely make configuration changes to the build and re-run make however you choose.

## Testing

To run your code on the cluster, you can use the generated `job-blocked` script like so:

```
student@login04:~/hw1-starter/build> sbatch job-blocked
Submitted batch job 9637622
```

The job is now submitted to Cori's job queue.
We can now check on the status of our submitted job using a few different commands.

```
student@login04:~/hw1-starter/build> squeue -u student
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           4613712 regular_m job-naiv   yuedai PD       0:00      1 (QOSMaxJobsPerUserLimit)
           4613708 regular_m job-bloc   yuedai  R       0:07      1 nid004961
           4613705 regular_m job-blas   yuedai  R       0:16      1 nid005254
student@login04:~/hw1-starter/build> sqs
JOBID            ST USER      NAME          NODES TIME_LIMIT       TIME  SUBMIT_TIME          QOS             START_TIME           FEATURES       NODELIST(REASON
4613760          PD yuedai    job-naive     1           2:00       0:00  2023-01-11T21:38:08  debug           2023-01-11T21:38:35  cpu            (QOSMaxJobsPerU
4613758          R  yuedai    job-blocked   1           2:00       0:02  2023-01-11T21:38:04  debug           2023-01-11T21:38:33  cpu            nid004649      
4613754          R  yuedai    job-blas      1           2:00       0:07  2023-01-11T21:37:58  debug           2023-01-11T21:38:28  cpu            nid006483

```

When our job is finished, we'll find new files in our build directory containing the output of our program.
For example, we'll find the files job-blas.o4613758 and job-blas.e4613758.
The first file contains the standard output of our program, and the second file contains the standard error.

Feel free to modify `job-blocked`, but note that changes to it will be overwritten by CMake if you reconfigure your
build.
It might therefore be easier to copy it under a new name like `my-job` and modify it as you desire.

### Interactive Sessions

You may find it useful to launch an [interactive session](https://docs.nersc.gov/jobs/interactive/) when developing
your code.
This lets you compile and run code interactively on a compute node that you've reserved.
In addition, running interactively lets you use the special interactive queue, which means you'll receive your
allocation quicker.

## Submission

We will provide the submission instructions later. 

## Windows Instructions

We recommend using [CLion](https://www.jetbrains.com/clion/) ([free for students](https://www.jetbrains.com/student/))
and [WSL](https://docs.microsoft.com/en-us/windows/wsl/about) (Ubuntu 18.04.3 LTS) for developing on Windows.
CLion provides [instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-clion.html)
for setting up the IDE for use with WSL.
Be sure to install `libopenblas-dev` from within Ubuntu as well.

The starter code will compiler with MSVC and Visual Studio on Windows, but we do not recommend trying to write
first with MSVC and then porting to GCC (the required compiler).
MSVC does not implement many useful features in the C language and is fundamentally a C++ compiler.

## GCC Special Features

If you find that certain compiler flags offer a significant speed up to your code, you should add them to your
source file using the GCC optimize pragma.
For instance if you wanted to specifically enable loop peeling, you could add the following line to the top of your file.

```
#pragma GCC optimize ("peel-loops")
```

This works with any -f flag (eg. -fpeel-loops)
Note that this applies to all functions.
If you want to just tune the optimization of a single function use

```
__attribute__((optimize("peel-loops")))
void my_func() { ... }
```

See it in action here: https://godbolt.org/z/RvXfty.

Read more in the GCC documentation here: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#Common-Function-Attributes 
