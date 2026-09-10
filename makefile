#-------------------------------------------------------------------------------------------------------------------------
# FILE: makefile
# DATE: June, 1993
# BY:   Andrew L. Thomas
#-------------------------------------------------------------------------------------------------------------------------
# DATE: 2015-03-02
# BY:   Scott T. Marshall
# UPDATE: Updated for poly3d v1.2. Added compiler flags for several different operating systems
# Note that the intel compiler (icc) generates binaries that are up to ~40x faster than some 
# versions of gcc.
#-------------------------------------------------------------------------------------------------------------------------
# DATE: 2025-12-20
# BY:   Scott T. Marshall
# UPDATE: With the assistance of an LLM (and much testing), we now have the first parallelized poly3d version, poly3d_omp!
#  1) Now compiles with gcc version 13.3.0 on WSL running Ubuntu 24.04.3 LTS
#  2) Building the IC matrix is now parallelized with OpenMP
#  3) The IC matrix is now parallelized and solved by LU decomposition using dgetrs_ (factorization) and dgetrs_ (solving).
#     This is much more efficient and accurate than the original numerical recipes routine in the original poly3d.
#     Requires gcc, LAPACK, OpenBLAS, and OpenMP libraries. If you use the regular libblas-dev, only one CPU is used.
#       Install Ubuntu packages with: 
#         sudo apt install build-essential liblapack-dev libblas-dev libopenblas-dev
#   4) Printing the output file is parallelized using OpenMP.
#   5) Now builds poly3d_omp
#-------------------------------------------------------------------------------------------------------------------------
# This Linux makefile can be used to compile and link the source code 
# for the C-language program Poly3D. There are options for both gcc and icc (Intel's compiler).
# Intel's compiler is MUCH faster, but is no longer necessary with the recent parallelization of poly3d. 
# See the README file for details.

# Two commands are supported:
# 1) make
#      This will build the program poly3d_omp, depositing the executable file, poly3d_parallel in BINDIR.
#      Also removes all .o (object files) when complete. 
#
# 2) make clean
#      This will remove the unneeded object files from the source directory.
#
#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------
# Linux (64-bit) Options
#-------------------------------------------------------------------------------------------------------------------------
ifeq ($(shell uname),Linux)
	#Use the intel C compiler and use only 100% safe optimizations. 
	#Adding -ipo and seems to make little difference, and sometimes actually slows things down a bit.
	#adding -static makes the binary larger, but will include intel's faster math libs in the executable,
	#so it is useful for distributing to other computers. If you don't use static, the computer will use
	#its own slower math libs (probably from gcc).
	#This is what I used from July 2016 to December 2025
	#CODE   = poly3d_64
	#CC     = icc
	#CFLAGS = -ansi -I. -O2 -ipo -static
	#LIBS   = -lm
		
	#NON-PARALLELIZED GCC OPTIONS
	#Use gcc C compiler with updated compiler flags
	#CODE   = poly3d_64_gcc_fixed
	#CC     = gcc
	#CFLAGS = -std=c99 -I. -O2 -Wall
	#LIBS   = -lm
	
	#PARALLELIZED GCC OPTIONS USES LAPACK/BLAS and OPENMP
	#Use gcc C compiler with updated compiler flags for parallelization using OpenMP, LAPACK, and BLAS.
	CODE   = poly3d_omp
	CC     = gcc
	CFLAGS = -I. -O2 -Wall -fopenmp
	LIBS   = -lm -llapack -lblas
endif


#-------------------------------------------------------------------------------------------------------------------------
# For All Operating Systems
#-------------------------------------------------------------------------------------------------------------------------
# The directory for the created executable
BINDIR = .
# Source Files
SOURCES  = poly3d.c infcoeff.c matrix.c safetan.c getwords.c getoptPoly3D.c elastic.c nr.c nrutil.c
# Object Files
OBJECTS  = $(SOURCES:.c=.o)
# Include Files
INCLUDES = infcoeff.h matrix.h safetan.h getwords.h getoptPoly3D.h elastic.h nr.h nrutil.h pi.h


#-------------------------------------------------------------------------------------------------------------------------
# Build The Program, poly3d_64
#-------------------------------------------------------------------------------------------------------------------------
poly3d_64:	$(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $(BINDIR)/$(CODE)
	rm -f $(OBJECTS)

#-------------------------------------------------------------------------------------------------------------------------
# Delete Unneeded Object Files If The User Calls "make clean"
#-------------------------------------------------------------------------------------------------------------------------
clean:
	rm -f $(OBJECTS)


#-------------------------------------------------------------------------------------------------------------------------
# Dependency Tree
#-------------------------------------------------------------------------------------------------------------------------
poly3d.o: poly3d.c $(INCLUDES)

infcoeff.o: infcoeff.c infcoeff.h safetan.h pi.h

matrix.o: matrix.c matrix.h nr.h nrutil.h

safetan.o: safetan.c safetan.h pi.h

getwords.o: getwords.c getwords.h

getoptPoly3D.o: getoptPoly3D.c getoptPoly3D.h

elastic.o: elastic.c elastic.h matrix.h

nrutil.o: nrutil.c nrutil.h

nr.o: nr.c nr.h


