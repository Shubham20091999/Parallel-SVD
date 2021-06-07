#pragma once
#ifdef _WIN32
#define WIN 1
#else
#define WIN 0
#endif

#include <iostream>

#if WIN
#include <ctime>
#endif

#if !WIN
#include <sys/time.h>
#endif

#if WIN
class Timer
{
	double start = 0.0;

public:
	void begin()
	{
		start = clock();
	}

	double end()
	{
		double stop = clock();
		return (stop - start) / CLOCKS_PER_SEC;
	}
};
#endif

#if !WIN
class Timer
{
	struct timeval start, stop;

public:
	void begin()
	{
		gettimeofday(&start, NULL);
	}

	double end()
	{
		gettimeofday(&stop, NULL);
		return ((stop.tv_sec - start.tv_sec) * 1000000u +
				stop.tv_usec - start.tv_usec) /
			   1.e6;
	}
};
#endif