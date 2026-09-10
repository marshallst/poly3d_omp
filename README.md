# poly3d_omp
`poly3d_omp` is a parallelized fork of the Poly3D Boundary Element Method (BEM) code for 3D fault and fracture modeling. `poly3d_omp` is upgraded from the original C implementation (Thomas, 1993) using OpenMP for multi-threading, OpenBLAS for parallelized linear algebra, and LLM-assisted refactoring. The usage and mathematics of the code is otherwise unchanged from the original `poly3d`. For details of these changes, see the included `poly3d.c` and refer to the revision history section at the top of the code. This repository builds on the original open source code for poly3d, a Boundary Element Method program first released in the M.S. thesis by Andrew Thomas, 1993. Parallelization and implementation of the OpenBLAS solver has dramatically improved performance, so the goal here is to share this code in the hopes that it is useful to others for structural geology, geomechanics, and active tectonics research. 

The source code posted here includes a sample model for testing and validation purposes 800elts.in and 800elts.out.

The original `poly3d` open source code and documentation (reproduced in modified form below) is currently available on github at
https://github.com/stgl/poly3d

## Documentation

The primary `poly3d` and `poly3d_omp` manual is [Thomas, 1993](http://searchworks.stanford.edu/view/2830996). It covers the details of the boundary value problems behind `poly3d` and explains the input and output file formats.

### Dependencies

On Ubuntu, before compiling `poly3d_omp` you will likely need to install some dependencies to enable full parallelization.<br>

On Ubuntu, the command below should install the dependencies needed to build `poly3d_omp`.<br>
`sudo apt install build-essential libomp-dev liblapack-dev libopenblas-dev`<br>

On Mac OS, using `brew` the install command below may work (unverified).<br>
`brew install openblas lapack libomp`<br>

### Compiling `poly3d_omp`
This version of `poly3d_omp` has been compiled and tested only on Ubuntu Linux. The makefile provided here assumes you are using a recent version of `gcc`, the GNU Compiler Collection and running Ubuntu Linux. The makefile has not been tested in Mac OS, but should theoretically work provided your system has `gcc`, and the dependencies listed above installed and available.

`poly3d_omp` requires several flags, so a makefile is provided. Note that like `poly3d`, `poly3d_omp` must be compiled with the `-O2` flag. Over-optimization (e.g. `-O3` or `-ffast-math`) can lead to unpredictable numerical errors and large singularities in the resulting displacement fields.<br>

To build `poly3d_omp` simply browse to the directory with the code and makefile and type<br>
`make`<br>
This will compile `poly3d_omp`

### A Warning About CPU Usage and Temperatures

Because `poly3d_omp` is fully parallelized, it will by default use every thread that your CPU has. This can result in overheating of your CPU if your system is not sufficiently well cooled (often the case with laptops). To maintain stability of your operating system it recommended to limit `poly3d_omp` to use one less than your max CPU threads. In other words, if your machine has a 32 thread CPU, it is recommended to limit `poly3d_omp` to 31 threads or less. This way your machine can still have at least one CPU thread for background operating system tasks. You can set this as an environment variable in your .bashrc file with the line below, assuming your CPU has 32 threads.
`export OMP_NUM_THREADS=31`
`export OPENBLAS_NUM_THREADS=31`

### Using `poly3d_omp`

`poly3d_omp -i <input> -o <output>`

### Contributors 

`poly3d` and a related commercial software package were developed by researchers and students of Profs. David Pollard and Atilla Aydin associated with the Stanford Rock Fracture Project and the Structural Geology and Geomechanics research group. This repository is intended for academic use only, and is solely derived from the publicly available source code printed in Thomas, 1993 with modifications to fix the "shadow effect.", and the modifications needed to parallelize the code.

### References

Please cite this thesis and related publications if you use `poly3d` or `poly3d_omp` in published work.

Thomas, A. L., 1993, Poly3D: A three-dimensional, polygonal element, displacement discontinuity boundary element computer program with applications to fractures, faults, and cavities in the Earth's crust (M.S. Thesis, Stanford University).

To see a recent example of `poly3d` in active tectonics research, see [this 2022 paper by Devine, Harper, and Marshall](https://doi.org/10.1785/0220220182) on `poly3d` calculated slip rates and distibutions along faults in southern California.

### Contact
Please [open an issue](https://github.com/marshallst/poly3d_omp/issues/new) if you have a question.
