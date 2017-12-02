#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <math.h>

double hyperbolicTangent(double x) {
	return tanh(x);
 }

double deriviateHyperbolicTangent(double valueTanh) {
	return (1 - valueTanh) * (1 + valueTanh);
}

double* softmax(double *g, int numberNeurons) {
	double* valueFunction = new double[numberNeurons];
	double sumExp = 0;

	for (int m = 0; m < numberNeurons; m++) {
		sumExp += exp(g[m]);
	}

	for (int j = 0; j < numberNeurons; j++) {
		valueFunction[j] = exp(g[j]) / sumExp;
	}

	return valueFunction;
}

#endif