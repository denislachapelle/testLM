#
# DL251027
# this file compile all programs developped while experimenting with Lagrange Multiplier
# to implement boundary conditions using mfem.
#

MFEM_DIR ?= /mnt/c/mfem-4.7
MFEM_DIR48 ?= /mnt/c/mfem/mfem-4.8
MFEM_BUILD_DIR ?= /mnt/c/mfem-4.7
MFEM_BUILD_DIR48 ?= /mnt/c/mfem/mfem-4.8

GMSH_DIR ?= /
#MFEM_DIR ?= /home/denislachapelle2003/fem/mfem-4.6
#MFEM_BUILD_DIR ?= /home/denislachapelle2003/fem/mfem-4.6

all: ex1p exp_01 ex1p_02 ex1_03 ex1_04 ex1_05

ex1p: ex1p.cpp
	mpicxx  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	-I$(MFEM_DIR48)/../hypre/src/hypre/include \
	ex1p.cpp -o ex1p \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE \
	-L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
	-lrt


ex1p_01: ex1p_01.cpp
	mpicxx  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	-I$(MFEM_DIR48)/../hypre/src/hypre/include \
	ex1p_01.cpp -o ex1p_01 \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE \
	-L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
	-lrt
	

ex1p_02: ex1p_02.cpp
	mpicxx  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	-I$(MFEM_DIR48)/../hypre/src/hypre/include \
	ex1p_02.cpp -o ex1p_02 \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-L$(MFEM_DIR48)/../hypre-2.33.0/src/hypre/lib -lHYPRE \
	-L$(MFEM_DIR48)/../metis-4.0.3 -lmetis \
	-lrt

ex1_03: ex1_03.cpp
	g++  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	ex1_03.cpp -o ex1_03 \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-lrt

ex1_04: ex1_04.cpp
	g++  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	ex1_04.cpp -o ex1_04 \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-lrt	


ex1_05: ex1_05.cpp
	g++  -g -O0 -Wall -std=c++11 \
	-I$(MFEM_DIR48) \
	ex1_05.cpp -o ex1_05 \
	-L$(MFEM_BUILD_DIR48) -lmfem \
	-lrt	
