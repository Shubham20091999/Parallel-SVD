#pragma once
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include "Utils.h"
#include <functional>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif // !NUM_Threads


using namespace std;

struct SVDMats;
class Mat
{
	using vec = std::vector<double>;

private:
	int nrows;
	vec m_mat;

public:
	Mat(const int m);

	template <typename T>
	Mat(std::initializer_list<std::initializer_list<T>> e);

	friend std::ostream& operator<<(std::ostream& os, const Mat& a);

	double& operator()(int i, int j);
	double operator()(int i, int j) const;

	static Mat Identity(int m);
	void Identity();

	double Get(const int m, const int n) const;
	double& Get(const int m, const int n);

	Mat operator*(const Mat& alt) const;

	Mat Transpose();
	static Mat Transpose(const Mat& A);
	Mat Mult_UTxAxU(const Mat& U);

	//Algorithms
	double errorVal();

	//HouseHolder's Algorithm
	Mat QR() const;
	void EigenQR(const bool info = false, const double tol = pow(10, -8));

	//Using Hessenberg Matrix
	void Hessenberg();
	void EigenHQR(const bool info = false, const double tol = pow(10, -8));

	//Serial Two Sided Jacobi
	void IterSeial2JAC(int i, int j, Mat& U);
	static Mat EigenOptimizedSerial2JAC(Mat& A, const bool info = false, const double tol = pow(10, -8));

	//Parallel Two Sided Jacobi
	void IterParallel2JAC(Mat& U, int i, int j);
	static Mat EigenParallel2JAC(Mat& A, const bool info = false, const double tol = pow(10, -8));

	//SVD
	static void getSortedEigVec(Mat& Val, Mat& Vec);
	static void getSortedEigVal(Mat& Val);
	static SVDMats SVD(Mat& A, function<Mat(Mat&, bool, double)> EigenFn, const bool info = false, const double tol = pow(10, -8));
};

struct SVDMats
{
	Mat V, S, U;
};

//---------------------------------------------QR for Eigen Values---------------------------------------
Mat Mat::QR() const
{
	//Mat R = *this;
	Mat QT = Identity(nrows);

	//Mat tmp_R(m_nrow, m_ncol);
	Mat tmp_QT(nrows);

	vec v(nrows);
	Mat H(nrows);

	for (int i = 0; i < nrows; i++)
	{
		double magW = 0;
		double magY = 0;
		for (int j = i; j < nrows; j++)
		{
			magW += pow((*this).Get(j, i), 2.0);
		}

		magY = copysign(1.0, (*this).Get(i, i)) * sqrt(magW);
		magW = sqrt(magW - pow((*this).Get(i, i), 2.0) + pow((*this).Get(i, i) + magY, 2));
		v[i] = ((*this).Get(i, i) + magY) / magW;

		for (int j = i + 1; j < nrows; j++)
		{
			v[j] = (*this).Get(j, i) / magW;
		}
		H.Identity();
#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for collapse(2)
			for (int m = i; m < nrows; m++)
			{
				for (int n = i; n < nrows; n++)
				{
					H.Get(m, n) -= 2.0 * v[m] * v[n];
				}
			}

#pragma omp for collapse(2)
			for (int m = 0; m < nrows; m++)
			{
				for (int n = 0; n < nrows; n++)
				{
					double tmpQ = 0;
					// double tmpR = 0;

					for (int k = 0; k < nrows; k++)
					{
						tmpQ += H.Get(m, k) * QT.Get(k, n);
						//tmpR += H.Get(m, k) * R.Get(k, n);
					}
					tmp_QT.Get(m, n) = tmpQ;
					//tmp_R.Get(m, n) = tmpR;
				}
			}
		}
		QT.m_mat.swap(tmp_QT.m_mat);
		//R.m_mat.swap(tmp_R.m_mat);
	}

	QT.Transpose();

	return QT;
}

void Mat::EigenQR(const bool info, const double tol)
{
	double ratio = 1;
	int numiter = 0;

	Timer t;
	t.begin();
	Mat tmpM(nrows);
	while (ratio > tol)
	{
		auto Q = QR();
		Mult_UTxAxU(Q);
		ratio = errorVal();
		// cout<<ratio<<"\n";
		numiter += 1;
	}

	double delta = t.end();
	if (info)
	{
		std::cout << "QR\n"
			<< "Time Taken: " << delta << "\n"
			<< "Number of Iteration: "
			<< numiter << "\n";
	}
}

//---------------------------------------------Hessenberg QR for Eigen Values---------------------------------------
void Mat::Hessenberg()
{
	/*for k = 1:n - 2
	x = A(k + 1:n, k : k);
	x(1) = x(1) + sign(x(1)) * norm(x);
	u = x / norm(x);
	A(k + 1:n, k : n) = A(k + 1:n, k : n) - 2 * u * (u.'*A(k+1:n,k:n));
		A(1:n, k + 1 : n) = A(1:n, k + 1 : n) - 2 * (A(1:n, k + 1 : n) * u) * u.';
		end
		H = A;*/

	vec u(nrows);
	for (int i = 0; i < nrows - 2; i++)
	{
		double magU = 0.0;
		//Can be parallelized
		for (int j = i + 1; j < nrows; j++)
		{
			magU += pow(Get(j, i), 2);
		}

		double magX = copysign(1.0, Get(i + 1, i)) * sqrt(magU);
		magU = sqrt(magU + 2 * magX * Get(i + 1, i) + pow(magX, 2));

		u[i + 1] = (Get(i + 1, i) + magX) / magU;

		//Can be parallelized
		for (int j = i + 2; j < nrows; j++)
		{
			u[j] = Get(j, i) / magU;
		}

#pragma omp parallel num_threads(NUM_THREADS)
		{
#pragma omp for
			for (int l = i; l < nrows; l++)
			{
				double tmp = 0;

				//Can be parallelized
				for (int m = i + 1; m < nrows; m++)
				{
					tmp += u[m] * Get(m, l);
				}
				//Can be parallelized
				for (int m = i + 1; m < nrows; m++)
				{
					Get(m, l) -= 2 * u[m] * tmp;
				}
			}

#pragma omp for
			for (int l = 0; l < nrows; l++)
			{
				double tmp = 0;
				//Can be parallelized
				for (int m = i + 1; m < nrows; m++)
				{
					tmp += u[m] * Get(l, m);
				}
				//Can be parallelized
				for (int m = i + 1; m < nrows; m++)
				{
					Get(l, m) -= 2 * u[m] * tmp;
				}
			}
		}
	}
}

void Mat::EigenHQR(const bool info, const double tol)
{

	/*for k = 1:1000
	for i = 1 : n - 1
	a = H(i, i);
	b = H(i + 1, i);
	c(i) = a / sqrt(a * a + b * b);
	s(i) = b / sqrt(a * a + b * b);
	H(i:i + 1, i : n) = [c(i) s(i); -s(i) c(i)] * H(i:i + 1, i : n);
	H(1:i + 1, i : i + 1) = H(1:i + 1, i : i + 1) * [c(i) - s(i); s(i) c(i)];
	end
	end*/

	int numiter = 0;
	double ratio = 1;

	Timer t;

	t.begin();
	Hessenberg();

	vec c(nrows);
	vec s(nrows);

	while (ratio > tol)
	{
		for (int i = 0; i < nrows - 1; i++)
		{
			double a = Get(i, i);
			double b = Get(i + 1, i);
			double mag = sqrt(pow(a, 2) + pow(b, 2));
			c[i] = a / mag;
			s[i] = b / mag;

			for (int j = i; j < nrows; j++)
			{
				double tmp = c[i] * Get(i, j) + s[i] * Get(i + 1, j);
				Get(i + 1, j) = -s[i] * Get(i, j) + c[i] * Get(i + 1, j);
				Get(i, j) = tmp;
			}
		}

		for (int i = 0; i < nrows - 1; i++)
		{
			for (int j = 0; j < i + 2; j++)
			{
				double tmp = c[i] * Get(j, i) + s[i] * Get(j, i + 1);
				Get(j, i + 1) = -s[i] * Get(j, i) + c[i] * Get(j, i + 1);
				Get(j, i) = tmp;
			}
		}

		ratio = errorVal();
		numiter += 1;
	}
	double delta = t.end();

	if (info)
		std::cout << "Hessenberg QR\n"
		<< "Time Taken: " << delta << "\n"
		<< "Number of Iteration: " << numiter << "\n";
}

//---------------------------------------------Serial Two Sided Jacobi---------------------------------------

void Mat::IterSeial2JAC(int i, int j, Mat& U)
{
	/*function A = do(A, i, j)
		n = length(A);
	th = (A(j, j) - A(i, i)) / 2 / A(i, j);
	t = sign(th) / (abs(th) + sqrt(power(th, 2) + 1));
	c = 1 / sqrt(1 + power(t, 2));
	s = c * t;

	A(i, i) = A(i, i) - t * A(i, j);
	A(j, j) = A(j, j) + t * A(i, j);
	for r = 1:n
		if (r == j || r == i)
			continue;
	end
		tmp = A(i, r);
	A(i, r) = c * A(i, r) - s * A(j, r);
	A(j, r) = c * A(j, r) + s * tmp;
	A(r, i) = A(i, r);
	A(r, j) = A(j, r);
	end
		A(i, j) = 0;
	A(j, i) = 0;
	end*/
	double th;
	double t;
	if (Get(i, j) != 0)
	{
		th = (Get(j, j) - Get(i, i)) / 2 / Get(i, j);
		t = copysign(1.0, th) / (abs(th) + sqrt(1 + pow(th, 2)));
	}
	else
	{
		t = 0;
	}
	double c = 1 / sqrt(1 + pow(t, 2));
	double s = c * t;

	Get(i, i) = Get(i, i) - t * Get(i, j);
	Get(j, j) = Get(j, j) + t * Get(i, j);

	for (int r = 0; r < nrows; r++)
	{
		double tmp_U = U(r, i);
		U(r, i) = U(r, i) * c - s * U(r, j);
		U(r, j) = tmp_U * s + c * U(r, j);
		if (r == j or r == i)
			continue;

		double tmp = Get(i, r);
		Get(i, r) = c * Get(i, r) - s * Get(j, r);
		Get(j, r) = c * Get(j, r) + s * tmp;
		Get(r, i) = Get(i, r);
		Get(r, j) = Get(j, r);
	}
	Get(i, j) = 0;
	Get(j, i) = 0;
}

Mat Mat::EigenOptimizedSerial2JAC(Mat& A, const bool info, const double tol)
{
	/*for niter = 1:20
		for k = 1 : n - 1
			for i = 1 : ceil((n - k) / 2)
				j = n - k + 2 - i;
	A = do(A, i, j);
	end
		if (k > 2)
			for i = (n - k + 2) : (n - floor(1 / 2 * k))
				j = 2 * n - k + 2 - i;
	A = do(A, i, j);
	end
		end
		end
		for i = 2:ceil(1 / 2 * n)
			j = n + 2 - i;
	A = do(A, i, j);
	end
		end*/

	int numiter = 0;
	double ratio = 1.0f;
	Timer t;
	int nrows = A.nrows;
	t.begin();
	Mat J = Identity(nrows);
	while (ratio > tol)
		//for (size_t i = 0; i < 1; i++)
	{
		for (int i = 0; i < nrows; i++)
		{
			for (int j = i + 1; j < nrows; j++)
			{
				A.IterSeial2JAC(i, j, J);
			}
		}
		ratio = A.errorVal();
		numiter += 1;
	}
	double delta = t.end();

	if (info)
	{
		std::cout << "Serial Two Sided Jacobi\n"
			<< "Time Taken: " << delta << "\n"
			<< "Number of Iteration: " << numiter << "\n";
	}

	return J;
}

//---------------------------------------------Parallel Two Sided Jacobi-------------------------------------

void Mat::IterParallel2JAC(Mat& U, int i, int j)
{
	/*function U = do(A, U, i, j)
	th = (A(j, j) - A(i, i)) / 2 / A(i, j);
	t = sign(th) / (abs(th) + sqrt(power(th, 2) + 1));
	c = 1 / sqrt(1 + power(t, 2));
	s = c * t;
	U(i, i) = c;
	U(j, j) = c;
	U(i, j) = s;
	U(j, i) = -s;
	end*/

	double th;
	double t;
	if (Get(i, j) != 0)
	{
		th = (Get(j, j) - Get(i, i)) / 2 / Get(i, j);
		t = copysign(1.0, th) / (abs(th) + sqrt(1 + pow(th, 2)));
	}
	else
	{
		t = 0;
	}
	double c = 1 / sqrt(1 + pow(t, 2));
	double s = c * t;

	U(i, i) = U(j, j) = c;
	U(i, j) = s;
	U(j, i) = -s;
}

//We tried to make the code such that it does not initalize and destroy threads
//multiple times in a single iteration but that code took almost the same amount of time
//as this current code, so for better readability we kept the simplified code.
Mat Mat::EigenParallel2JAC(Mat& A, const bool info, const double tol)
{
	/*J = eye(n);
	for niter = 1:30
		% U = eye(n);
	for k = 1:n - 1
		U = eye(n);
	for i = 1:ceil((n - k) / 2)
		j = n - k + 2 - i;
	U = do(A, U, i, j);
	end
		A = transpose(U) * A * U;
	J = J * U;

	if (k > 2)
		U = eye(n);
	for i = (n - k + 2) : (n - floor(1 / 2 * k))
		j = 2 * n - k + 2 - i;
	U = do(A, U, i, j);
	end
		A = transpose(U) * A * U;
	J = J * U;
	end
		end
		U = eye(n);
	for i = 2:ceil(1 / 2 * n)
		j = n + 2 - i;
	U = do(A, U, i, j);
	end
		A = transpose(U) * A * U;
	J = J * U;
	end*/

	int numiter = 0;
	double ratio = 1;
	Timer t;

	int nrows = A.nrows;

	t.begin();
	Mat U(nrows);
	Mat J = Identity(nrows);
	Mat tmpM(nrows);
	while (ratio > tol)
	{
		for (int k = 0; k < nrows - 1; k++)
		{
			int end = (int)ceil((float)(nrows - k - 1) / 2.0f);
			U.Identity();
#pragma omp parallel for num_threads(NUM_THREADS)
			for (int i = 0; i < end; i++)
			{
				int j = nrows - k - i - 1;
				A.IterParallel2JAC(U, i, j);
			}
			J = J * U;
			A.Mult_UTxAxU(U);

			if (k > 1)
			{
				int start = (nrows - k);
				int end = (int)(nrows - (int)floor((float)(1 + k) / 2.0f));
				U.Identity();
#pragma omp parallel for num_threads(NUM_THREADS)
				for (int i = start; i < end; i++)
				{
					int j = 2 * nrows - k - i - 1;
					A.IterParallel2JAC(U, i, j);
				}
				J = J * U;
				A.Mult_UTxAxU(U);
			}
		}
		int end = (int)ceil((float)nrows / 2.0f);
		U.Identity();
#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 1; i < end; i++)
		{
			int j = nrows - i;
			A.IterParallel2JAC(U, i, j);
		}
		J = J * U;
		A.Mult_UTxAxU(U);

		ratio = A.errorVal();
		numiter += 1;
	}

	double delta = t.end();

	if (info)
	{
		std::cout << "Parallel Two Sided Jacobi\n"
			<< "Time Taken: " << delta << "\n"
			<< "Number of Iteration: "
			<< numiter << "\n";
	}

	return J;
}

//---------------------------------------------SVD----------------------------------------------------

void Mat::getSortedEigVec(Mat& Val, Mat& Vec)
{
	vector<int> pos(Vec.nrows, 0);
	vector<int> bucket(Vec.nrows, 0);

	//Prallalizable
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < Val.nrows; i++)
	{
		int tmp = 0;
		for (int j = 0; j < Val.nrows; j++)
		{
			if (Val(i, i) < Val(j, j))
			{
				tmp += 1;
			}
		}
		pos[i] = tmp;
	}

	for (int i = 0; i < Val.nrows; i++)
	{
		for (int j = 0; j < Val.nrows; j++)
		{
			//We dont need Val anymore so recycling it
			Val(j, pos[i] + bucket[pos[i]]) = Vec(j, i);
		}
		bucket[pos[i]] += 1;
	}
}

void Mat::getSortedEigVal(Mat& Val)
{
	//Implement parallalized Bubble sort
	for (int i = 0; i < Val.nrows; i++)
	{
		for (int j = 0; j < Val.nrows - i - 1; j++)
		{
			if (abs(Val(j, j)) < abs(Val(j + 1, j + 1)))
			{
				swap(Val(j, j), Val(j + 1, j + 1));
			}
		}
		for (int j = 0; j < Val.nrows; j++)
		{
			if (j != i)
				Val(i, j) = 0;
		}
	}
}

SVDMats Mat::SVD(Mat& A, function<Mat(Mat&, bool, double)> EigenFn, const bool info, const double tol)
{
	Timer t;
	t.begin();
	Mat AT = Mat::Transpose(A);
	Mat ATA = AT * A;
	Mat AAT = A * AT;

	//As we dont need the AT I just aliased it as tmp_V(same for tmp_U later)
	Mat& tmp_V = AT;
	tmp_V = EigenFn(ATA, false, tol);
	//We dont need ATA anymore so recycling it
	getSortedEigVec(ATA, tmp_V);
	Mat& V = ATA;

	//cout << V;

	Mat& tmp_U = AT;
	tmp_U = EigenFn(AAT, false, tol);
	//We dont need AAT anymore so recycling it
	getSortedEigVec(AAT, tmp_U);
	Mat& U = AAT;

	EigenFn(A, false, tol);
	getSortedEigVal(A);

	double delta = t.end();

	if (info)
	{
		std::cout << "SVD\n"
			<< "Time Taken: " << delta << "\n";
	}

	return SVDMats{
		V,
		A,
		U,
	};
}

//---------------------------------------------Misc functions---------------------------------------
Mat::Mat(const int m) : nrows(m), m_mat(vec(m* m, 0)) {}

template <typename T>
Mat::Mat(std::initializer_list<std::initializer_list<T>> e)
	: nrows(e.size()), m_mat(vec(nrows* nrows))
{
	unsigned int i = 0;
	for (auto& row : e)
	{
		for (auto& col : row)
		{
			m_mat[i] = col;
			i += 1;
		}
		if (i % nrows != 0)
			throw("Mismatch Column Size");
	}
}

Mat Mat::operator*(const Mat& alt) const
{
	Mat ret(nrows);
#pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < alt.nrows; j++)
		{
			double tmp = 0;
			for (int k = 0; k < nrows; k++)
			{
				tmp += Get(i, k) * alt.Get(k, j);
			}
			ret.Get(i, j) = tmp;
		}
	}
	return ret;
}

double& Mat::operator()(int i, int j)
{
	return Get(i, j);
}

double Mat::operator()(int i, int j) const
{
	return Get(i, j);
}

std::ostream& operator<<(std::ostream& os, const Mat& a)
{
	os << "\n[";
	for (int i = 0; i < a.nrows; i++)
	{
		for (int j = 0; j < a.nrows; j++)
		{
			os << a.m_mat[i * a.nrows + j];
			os << " ";
		}
		os << "\n";
	}
	os << "]\n";
	return os;
}

Mat Mat::Identity(int m)
{
	Mat ret(m);
	for (int i = 0; i < m; i++)
	{
		ret.Get(i, i) = 1.0;
	}
	return ret;
}

void Mat::Identity()
{
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < nrows; i++)
	{
		Get(i, i) = 1;
		for (int j = i + 1; j < nrows; j++)
		{
			Get(i, j) = 0;
			Get(j, i) = 0;
		}
	}
}

Mat Mat::Transpose()
{
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < nrows; i++)
		for (int j = i + 1; j < nrows; j++)
			swap(Get(i, j), Get(j, i));

	return *this;
}

Mat Mat::Transpose(const Mat& A)
{
	Mat ret(A.nrows);

	for (int i = 0; i < A.nrows; i++)
	{
		for (int j = 0; j < A.nrows; j++)
		{
			ret(i, j) = A(j, i);
		}
	}
	return ret;
}

double Mat::Get(const int m, const int n) const
{
	return m_mat[m * nrows + n];
}

double& Mat::Get(const int m, const int n)
{
	return m_mat[m * nrows + n];
}

Mat Mat::Mult_UTxAxU(const Mat& U)
{
	Mat tmpM(nrows);
#pragma omp parallel num_threads(NUM_THREADS)
	{
		//A*U
#pragma omp for collapse(2)
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < nrows; j++)
			{
				double tmp = 0;
				for (int k = 0; k < nrows; k++)
				{
					tmp += Get(i, k) * U(k, j);
				}
				tmpM.Get(i, j) = tmp;
			}
		}

		//UT*A
#pragma omp for collapse(2)
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < nrows; j++)
			{
				double tmp = 0;
				for (int k = 0; k < nrows; k++)
				{
					tmp += U(k, i) * tmpM.Get(k, j);
				}
				Get(i, j) = tmp;
			}
		}
	}

	return *this;
}

double Mat::errorVal()
{
	double neum = 0;
	double deno = 0;

#pragma omp parallel for num_threads(NUM_THREADS) reduction(+ \
															: neum, deno)
	for (int i = 0; i < nrows; i++)
	{
		deno += pow((float)Get(i, i), 2);
		for (int j = i + 1; j < nrows; j++)
		{
			neum += pow((float)Get(j, i), 2);
		}
	}
	return sqrt(neum / deno);
}
