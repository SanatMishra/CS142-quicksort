#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "get_time.h"
#include "parse_command_line.h"
#include "qsort.h"
using namespace std;

using data_type = uint64_t;

#define NUM_ROUNDS 10

inline data_type hash64o(data_type i)
{
	return i + 99 - 2*(i%100);
}

inline data_type hash64(data_type u )
{
	data_type v = u * 3935559000370003845ul + 2691343689449507681ul;
	v ^= v >> 21;
	v ^= v << 37;
	v ^= v >>  4;
	v *= 4768777513237032717ul;
	v ^= v << 20;
	v ^= v >> 41;
	v ^= v <<  5;
	return v;
}

void dupl(data_type* A, int n) {
	static constexpr int K = 3;
	cilk_for (int i = K; i < n; i++) {
		A[i] = A[hash64(i)%K];
	}
}

void dupl2(data_type* A, int n) {
	static constexpr int J = 3;
	static constexpr int K = 5;
	for (int i = J; i < n; i += 1 + (hash64(i) % K >= J)) {
		A[i] = A[hash64(i)%J];
	}
}

int main(int argc, char** argv) {
	commandLine P(argc, argv,
		"[-n num_elements]");
	int n = P.getOptionLongValue("-n", 100000000);
	
	data_type* A = new data_type[n];
	/*data_type* A_check = new data_type[n];
	cilk_for (int i = 0; i < n; i++) {
		A[i] = hash64(i);
		dupl2(A, n);
		dupl2(A_check, n);
		A_check[i] = A[i];
	}
	
	//correctness check, you can comment this out when you test performance
	std::sort(A_check, A_check+n);
	quicksort(A, n);
	cilk_for (int i = 0; i < n; i++) {
		if (A[i] != A_check[i]) {
			cout << "wrong answer" << endl;
			exit(1);
		}
	}
	delete[] A_check;
	*/
	//first round, not counted in total time
	cilk_for (int i = 0; i < n; i++) {
		A[i] = hash64(i);
		//A[i] = hh(i);
	}
	dupl2(A, n);
	cout << "." << endl;
	quicksort(A, n);
	cout << "." << endl;
	double tot_time = 0.0;
	for (int round = 0; round < NUM_ROUNDS; round++) {
		cilk_for (int i = 0; i < n; i++) {
			A[i] = hash64(i^(round + 2));
		}
		dupl2(A, n);
		timer t; t.start();
		quicksort(A, n);
		t.stop();
		double tm = t.get_total();
		cout << "Round " << round << ", time: " << tm << endl;
		tot_time += tm;
	}
	cout << "Average time: " << tot_time/NUM_ROUNDS << endl;
	return 0;
}
