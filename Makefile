EXECUTABLE := aceMatchTemplate

CU_FILES   :=

CU_DEPS    :=

CC_FILES   := main.cpp worker.cpp

all: $(EXECUTABLE) $(REFERENCE)

############################################

OBJDIR=objs
#SRCDIR=src

CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++11 -I /usr/local/cuda/include
CXXFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv` -L/usr/local/cuda/lib64 -lcudart

NVCC=nvcc
NVCCFLAGS=-O3 -m64 -arch compute_20 -std=c++11 -I /usr/local/cuda/include --compiler-options -Wall
NVCCFLAGS+=`pkg-config --cflags opencv`

OBJS=$(OBJDIR)/main.o $(OBJDIR)/worker.o
#SOURCE=$(SRCDIR)/main.cpp $(SRCDIR)/worker.cpp

default: $(EXECUTABLE)
#all:
# $(CC) -o c1413_notbb_time 1413_notbb_time.cpp $(CFLAG)
#	$(CC)  -o c1413_cuda 1413_cuda.cu $(CFLAG)
dirs:
		mkdir -p $(OBJDIR)/
		mkdir -p results/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS)

#$(OBJDIR)/%.o: %.cpp
#		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: src/main.cpp src/worker.h
		$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/worker.o: src/worker.cu src/worker.h src/myOwnMatchTemplate.cuh
		$(NVCC) $(NVCCFLAGS) -c -o $@ $<
