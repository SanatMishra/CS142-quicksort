CC = g++

D := 0

ifeq ($(D), 1)
	CFLAGS = -O3 -pg -ftest-coverage -fprofile-arcs -mcx16 -march=native -ffast-math -DCILK -fcilkplus -std=c++17
else
	CFLAGS = -O3 -Drestrict=__restrict__ -funroll-loops -march=native -mavx2 -mcx16 -mfpmath=sse -fopenmp -fopt-info-vec -ffast-math -DCILK -fcilkplus -std=c++17
endif

all:	qsort

q.s:	qsort.cpp qsort.h parse_command_line.h get_time.h 
	$(CC) $(CFLAGS) -DNDEBUG -fverbose-asm -S qsort.cpp -o q.s
qsort:	qsort.cpp qsort.h parse_command_line.h get_time.h 
	$(CC) $(CFLAGS) -DNDEBUG qsort.cpp -o qsort
	
qt:	qt.s
	g++ -c qt.s -o qt.o
	g++ $(CFLAGS) qt.o -o qt
	
clean:
	rm -f qsort
