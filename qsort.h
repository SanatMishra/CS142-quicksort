#include <algorithm>
//#include <immintrin.h>
//#include <xmmintrin.h>
#include <x86intrin.h>
#include <omp.h>
using namespace std;

#define NO_PAR \
0

#if (NO_PAR == 1)
	#define cilk_spawn
	#define cilk_sync
	#define cilk_for for
#endif

#define abbr(x) ((x) / 1000000000000000LLU)
#define cfl(b, x) ((-((b))) & x)

typedef __m256i ipr;

inline int np(int q) {
	q |= q >> 1;
	q |= q >> 2;
	q |= q >> 4;
	q |= q >> 8;
	q |= q >> 16;
	return q + 1;	
}

static uint32_t SEED = 0;
inline uint32_t rr() {
	SEED += 3000000019;
	SEED ^= SEED << 21 | SEED >> 11;
	SEED += 2000000011;
	return SEED;
}

//typedef int64_t T;
template <typename T> 
class qsorter {
public:
	int N;
	void* buf;
	T* A;
	T* B;
	int* ls;
	//int* psf;
	
	constexpr static bool iss = 1 - is_signed<T>::value;
	constexpr static int ust = 10;
	constexpr static int pt = 1 << (ust - 1);
	constexpr static int qt = 100000000/128;

	//struct closure {
	//	int m, n, r, c;	T p;
	//	T* A; T* B; int* ls;	
	
		void us(int m, int n, int r, T p, int k, int &ret1, int &ret2) {
			//cout << "us " << k << endl;
			if (k < (n >> ust)) {
				int r1, r2, r3, r4;
				cilk_spawn us(m, n, r, p, 2*k, r1, r2);
				us(m, n, r, p, 2*k + 1, r3, r4);
				cilk_sync;
				ret1 = r1 + r3;
				ret2 = r2 + r4;
			} else {
				int i, rb;
				if ((k << ust) < n) {
					i = m + ((k << (ust + 1)) + r) - 2*n;
					rb = i + (1 << ust) + (n % (1 << ust));
				} else {
					i = m + ((k << ust) + r) % n;
					rb = i + (1 << ust);
				}
				ret1 = 0;
				ret2 = 0;
				//#pragma omp simd reduction(+:ret1,ret2)
				for (;i < rb; i++) {
					/*
					asm ( 
					"cmpq    %3, %2\n"
					"setb    %r8b\n"
					"seta    %r9b\n"
					"movzbl  %r8b, %0\n"
					"movzbl  %r9b, %1\n"
        				: "=r" (ret1), "=r" (ret2)
        				: "m" (A[i]), "r" (p)
        				: "r8", "r9");
					*/
					ret1 += A[i] < p;
					ret2 += A[i] > p;
				}
			}
			if (k % 2 == 0) {
				ls[2*(m >> ust) + k] = ret1;
				ls[2*(m >> ust) + k + 1] = ret2;
			}
		}

		void ds(int m, int n, int r, int c, T p, int s, int t, int k) {
			//cout << "ds " << s << " " << t << " " << k << endl;
			if (k < (n >> ust)) {
				cilk_spawn ds(m, n, r, c, p, s, t, 2*k);
				ds(m, n, r, c, p, s + ls[2*(m >> ust) + 2*k], t + ls[2*(m >> ust) + 2*k + 1], 2*k + 1);
				cilk_sync;
			} else {
#define DSVEC 1

#if (DSVEC == 0)
				int i, rb;
				if ((k << ust) < n) {
					i = m + ((k << (ust + 1)) + r) - 2*n;
					rb = i + (1 << ust) + (n % (1 << ust));
				} else {
					i = m + ((k << ust) + r) % n;
					rb = i + (1 << ust);
				}
				
				int psf;
				for (; i < rb; i++) {				
					psf = (cfl(A[i] < p, m + s) | cfl(A[i] > p, m + n - t - 1) | cfl(A[i] == p, c + i - s - t));
					//B[(cfl(A[i] < p, m + s) | cfl(A[i] > p, m + n - t - 1) | cfl(A[i] == p, c + i - s - t))] = A[i];
					s += (A[i] < p);
					t += (A[i] > p);
					B[psf] = A[i];
				}
#else					
				int i, rb;
				if ((k << ust) < n) {
					i = m + ((k << (ust + 1)) + r) - 2*n;
					rb = i + (1 << ust) + (n % (1 << ust));
				} else {
					i = m + ((k << ust) + r) % n;
					rb = i + (1 << ust);
				}
				//int psf[8];
				//ipr ns[8], nt[8], f[8], g[8];	
				ipr bridge = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7),
				    sl1 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6),
				    sl2 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5),
				    ind = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
				    inc = _mm256_set1_epi32(8),
				    zeros = _mm256_set1_epi32(0),
				    negate = _mm256_set1_epi32(-1),
				    chsign;
				    if (iss) chsign = _mm256_set1_epi64x(0x8000000000000000LLU);

				//#define bridge  _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6)
				//#define sl1  _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6)
				//#define sl2  _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5)
				//#define ind  _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)
				//#define inc  _mm256_set1_epi32(8)
				//#define zeros  _mm256_set1_epi32(0)
				//#define negate  _mm256_set1_epi32(-1)
				//#define chsign _mm256_set1_epi32(0x8000000000000000LLU)

				ipr k1 =  _mm256_set1_epi32(m - 1), //
				    k2 = _mm256_set1_epi32(m + n),      // potentially in the closure
				    a3 = _mm256_set1_epi64x(iss ? (p - 0x8000000000000000LLU) : p),  //
				    k3 = _mm256_add_epi32(_mm256_set1_epi32(c + i), ind);
				ipr spr = _mm256_set1_epi32(s), tpr = _mm256_set1_epi32(t);
				//cout << i << " " << rb - 1 << endl;			
				while (i + 8 <= rb) {
					
					ipr a1 = _mm256_loadu_si256((ipr*)(A + i)),
					    a2 = _mm256_loadu_si256((ipr*)(A + i + 4));
					// sub 0x8000000000000000LLU
					ipr f1, f2;
					if (iss) {f1 = _mm256_sub_epi64(a1, chsign);
					    f2 = _mm256_sub_epi64(a2, chsign);}
					ipr nt = _mm256_blend_epi32(_mm256_cmpgt_epi64(iss ? f1 : a1, a3), _mm256_cmpgt_epi64(iss ? f2 : a2, a3), 0xAA);
                                        ipr b0 = _mm256_permutevar8x32_epi32(nt, bridge);
                                        nt = _mm256_sub_epi32(zeros, b0);

                                        nt = _mm256_add_epi32(nt, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(nt, sl1), zeros, 0x01));
                                        nt = _mm256_add_epi32(nt, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(nt, sl2), zeros, 0x03));

                                        nt = _mm256_add_epi32(tpr, _mm256_add_epi32(nt, _mm256_permute2x128_si256(nt, nt, 0x08)));
                                        tpr = _mm256_permutevar8x32_epi32(nt, _mm256_set1_epi32(7));
                                        
					ipr resf = _mm256_sub_epi32(k3, nt);
                                        ipr res = _mm256_and_si256(b0, _mm256_sub_epi32(k2, nt));
                                        b0 = _mm256_xor_si256(negate, b0);
 
                                        ipr ns = _mm256_blend_epi32(_mm256_cmpgt_epi64(a3, iss ? f1 : a1), _mm256_cmpgt_epi64(a3, iss ? f2 : a2), 0xAA);
                                        ipr b2 = _mm256_permutevar8x32_epi32(ns, bridge);
                                        ns = _mm256_sub_epi32(zeros, b2);
                                        
                                        ns = _mm256_add_epi32(ns, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(ns, sl1), zeros, 0x01));
                                        ns = _mm256_add_epi32(ns, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(ns, sl2), zeros, 0x03));
                                        
                                        ns = _mm256_add_epi32(spr, _mm256_add_epi32(ns, _mm256_permute2x128_si256(ns, ns, 0x08)));
                                        spr = _mm256_permutevar8x32_epi32(ns, _mm256_set1_epi32(7));
                                        
                                        resf = _mm256_sub_epi32(resf, ns);
                                        res = _mm256_or_si256(res, _mm256_and_si256(b2, _mm256_add_epi32(k1, ns)));
                                        b0 = _mm256_andnot_si256(b2, b0);
                                        
                                        res = _mm256_or_si256(res, _mm256_and_si256(b0, resf));
                                        
					B[_mm256_extract_epi32(res, 0)] = _mm256_extract_epi64(a1, 0);
					B[_mm256_extract_epi32(res, 1)] = _mm256_extract_epi64(a1, 1);
					B[_mm256_extract_epi32(res, 2)] = _mm256_extract_epi64(a1, 2);
					B[_mm256_extract_epi32(res, 3)] = _mm256_extract_epi64(a1, 3);
					B[_mm256_extract_epi32(res, 4)] = _mm256_extract_epi64(a2, 0);
					B[_mm256_extract_epi32(res, 5)] = _mm256_extract_epi64(a2, 1);
					B[_mm256_extract_epi32(res, 6)] = _mm256_extract_epi64(a2, 2);
					B[_mm256_extract_epi32(res, 7)] = _mm256_extract_epi64(a2, 3);
					//cout << i   << " " << (d4[0] & 0x00000000ffffffffLLU) << endl;
					//cout << i+1 << " " << (d4[0]>>32 & 0x00000000ffffffffLLU) << endl;
					//cout << i+2 << " " << (d4[1] & 0x00000000ffffffffLLU) << endl;
					//cout << i+3 << " " << (d4[1]>>32 & 0x00000000ffffffffLLU) << endl;
					//cout << i+4 << " " << (d4[2] & 0x00000000ffffffffLLU) << endl;
					//cout << i+5 << " " << (d4[2]>>32 & 0x00000000ffffffffLLU) << endl;
					//cout << i+6 << " " << (d4[3] & 0x00000000ffffffffLLU) << endl;
					//cout << i+7 << " " << (d4[3]>>32 & 0x00000000ffffffffLLU) << endl;
					k3 = _mm256_add_epi32(k3, inc);
					i += 8;
				}
				int pse;
				s = _mm256_extract_epi32(spr, 0);
				t = _mm256_extract_epi32(tpr, 0);
				//cout << "out" << endl;
				for(;i < rb;i++) {
					pse = (cfl(A[i] < p, m + s) | cfl(A[i] > p, m + n - t - 1) | cfl(A[i] == p, c + i - s - t));
					s += (A[i] < p);
					t += (A[i] > p);
					B[pse] = A[i];
					//cout << i << " " << pse << endl;
				}
#endif				
			}
		}
	//};
	qsorter(T* a, int n) {
		A = a;
		N = n;
		buf = _mm_malloc(2*(N >> ust)*sizeof(int) + N*sizeof(T), 256);
		B = (T*)((uint8_t*)buf);
		ls = (int*)((uint8_t*)buf + N*sizeof(T));
		//psf = (int*)((uint8_t*)buf + 2*N*sizeof(int));
		//B = new T[N];
		//ls = new int[2*N];
		//psf = new int[N];
		//this->pt = qt;
	}

	~qsorter() {
		_mm_free(buf);
		//delete[] B; delete[] ls; delete[] psf;
	}

	inline int med(int a, int b, int c) {
		T aa = A[a]; T bb = A[b]; T cc = A[c];
		bool t = aa > bb;
		bool u = aa > cc;
		if (t^u) return a;
		bool v = bb > cc;
		if (u^v) return c;
		return b;
	}

	void qsr(int m, int n) {
		/*for (int i = 0; i < N; i++) {
			if (i % 20 == 0)
				cout << endl;
			cout << A[i] / 1000000000000000LLU << " ";
		}
		cout << endl << "qsr " << m << " " << n << endl;
		*/
		if (n <= qt) {
			sort(A + m, A + m + n);
			return;
		}
		//uint32_t aa, ab, ac;
		//aa = m + rr() % n; ab = m + rr() % n; ac = m + rr() % n;
		//uint32_t ma = med(aa, ab, ac);
		/*
		//cout << aa << " " << ab << " " << ac << " " << ma << endl;
		aa = m + rr() % n; ab = m + rr() % n; ac = m + rr() % n;
		uint32_t mb = med(aa, ab, ac);
		//cout << aa << " " << ab << " " << ac << " " << ma << endl;
		aa = m + rr() % n; ab = m + rr() % n; ac = m + rr() % n;
		uint32_t mc = med(aa, ab, ac);
		//cout << aa << " " << ab << " " << ac << " " << ma << endl;
		//cout << ma << " " << mb << " " << mc << " " << med(ma, mb, mc) << endl;
		T p = A[med(ma, mb, mc)];
		*/
		//T p = A[ma];
		//int ma = med(m, m + 1, m + 2);
		
		int ma = med(med(m, m + n/2, m + n - 1), med(m + 1, m + n/2 - 1, m + n - 2), med(m + 2, m + n/2 + 1, m + n - 3));
		int mb = med(med(m + 3, m + n/2 - 2, m + n - 4), med(m + 4, m + n/2 + 2, m + n - 5), med(m + 5, m + n/2 - 3, m + n - 6));
		int mc = med(med(m + 6, m + n/2 + 3, m + n - 7), med(m + 7, m + n/2 - 3, m + n - 8), med(m + 8, m + n/2 + 4, m + n - 9));
		ma = med(ma, mb, mc);
		T p = A[ma];	
		//T p = A[med(m, m + n/2, m + n - 1)];
		//T p = A[med(m, m + n/2, m + n - 1), med(m + n/8, m + 3*n/8, m + 7*n/8), med(m + n/4, m + 3*n/8, m + 5*n/8)];
		int c, d;
		//cout << "part " << m << " " << n << " " << p << endl;

		//cout << "call us " << m << " " << p << " " << n << " " << 2*n - np(n) << endl;
		int r = 2*n - np(n);
		//closure st {m, n, 2*n - np(n), 0, p, A, B, ls};
		us(m, n, r, p, 1, c, d);
		//for (int i = 2*m; i < 2*m + 2*n; i++) {
		//	cout << ls[i] << " ";
		//}
		//cout << endl;
		//cout << "call ds " << c << " " << d << endl;
		//st.c = c;
		ds(m, n, r, c, p, 0, 0, 1);
		//for (int i = m; i < m + n; i++)
		//	cout << psf[i] << " ";
		//cout << endl;
		cilk_for(int j = m; j < m + n; j += pt) {
			//A[j] = B[j];
			//A[j+1] = B[j+1];
			//A[j+2] = B[j+2];
			//A[j + 3] = B[j+3];
			
			int im = min(m + n, j + pt);
			for (int i = j; i < im; i++)
				A[i] = B[i];
			
			//int i = j;
			//while ((A + i) & 0xFF) {
			//	A[i] = B[i++]; 
			//}
			//#pragma omp simd simdlen(4)
			/*while(i + 4 <= im) {
				ipr bp = _mm256_loadu_si256((__m256i*)(B + i));
			//	cout << hex << (data_type)_mm256_extract_epi64(bp, 0) << " " << 
			//	(data_type)_mm256_extract_epi64(bp, 1) << " " << (data_type)_mm256_extract_epi64(bp, 2) <<
			//	" " << (data_type)_mm256_extract_epi64(bp, 3) << " " << endl;
			//	cout << B[i] << " " << B[i + 1] << " " << B[i + 2] << " " << B[i + 3] << endl;
				_mm256_storeu_si256((__m256i*)(A + i), bp);
			//	cout << A[i] << " " << A[i + 1] << " " << A[i + 2] << " " << A[i + 3] << endl;
				i += 4;
			}
			for (; i < im; i++)
				A[i] = B[i];
			*/
		}
		//moveToA(m, 0, p, n);
		cilk_spawn qsr(m, c);
		qsr(m + n - d, d);
		cilk_sync;
	}
};

template <typename T>
void quicksort(T* a, size_t n) {
	qsorter q(a, n);
	q.qsr(0, n);
}

