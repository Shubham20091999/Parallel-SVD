#define NUM_THREADS 8
#include <iostream>
#include <algorithm>
#include "Mat.h"

int main()
{
	int n = 50;
	Mat A(n);
	for (int i = 0; i < n; i++)
	{
		A.Get(i, i) = (((double)rand() / RAND_MAX) * 20);
		for (int j = i + 1; j < n; j++)
		{
			A.Get(i, j) = (((double)rand() / RAND_MAX) * 20);
			A.Get(j, i) = A.Get(i, j);
		}
	}

	Mat::SVD(A, Mat::EigenParallel2JAC, true);
	Mat::SVD(A, Mat::EigenOptimizedSerial2JAC, true);
}