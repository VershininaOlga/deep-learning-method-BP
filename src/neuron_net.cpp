
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include "neuron_net.h"
#include "activation_function.h"

NeuronNet::NeuronNet(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons) {
	numberInputNeurons = _numberInputNeurons;
	numberHiddenNeurons = _numberHiddenNeurons;
	numberOutputNeurons = _numberOutputNeurons;

	inputX = new double[numberInputNeurons];
	outputY = new double[numberOutputNeurons];

	hiddenOutputs = new double[numberHiddenNeurons];
	outputsNet = new double[numberOutputNeurons];

	weightsLayer1 = new double* [numberInputNeurons];
	for (int i = 0; i < numberInputNeurons; i++)
		weightsLayer1[i] = new double [numberHiddenNeurons];
	
	weightsLayer2 = new double* [numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++)
		weightsLayer2[s] = new double [numberOutputNeurons];

	initializeWeights();
}

NeuronNet::~NeuronNet() {
	delete[] inputX;
	delete[] outputY;
	delete[] hiddenOutputs;
	delete[] outputsNet;

	for (int i = 0; i < numberInputNeurons; i++)
		delete[] weightsLayer1[i];
	delete[] weightsLayer1;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] weightsLayer2[i];
	delete[] weightsLayer2;
}

void NeuronNet::initializeWeights() {
	srand(time(NULL));
	for (int i = 0; i < numberInputNeurons; i++)
		for (int j = 0; j < numberHiddenNeurons; j++)
			weightsLayer1[i][j] = (double(rand()) / (double)RAND_MAX) / 100.0;

	for (int i = 0; i < numberHiddenNeurons; i++)
		for (int j = 0; j < numberOutputNeurons; j++)
			weightsLayer2[i][j] = (double(rand()) / (double)RAND_MAX) / 100.0;
}

void NeuronNet::calculateHiddenOutputs() {
	double *f = new double[numberHiddenNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		f[s] = 0;
		for (int i = 0; i < numberInputNeurons; i++) {
			f[s] += weightsLayer1[i][s] * inputX[i];
		}
		hiddenOutputs[s] = hyperbolicTangent(f[s]);
	}
	hiddenOutputs[0] = 1;

	delete[] f;
}

void NeuronNet::calculateOutputs() {
	double *g = new double[numberOutputNeurons];

	calculateHiddenOutputs();

	for (int j = 0; j < numberOutputNeurons; j++) {
		g[j] = 0;
		for (int s = 0; s < numberHiddenNeurons; s++) {
			g[j] += weightsLayer2[s][j] * hiddenOutputs[s];
		}
	}

	outputsNet = softmax(g, numberOutputNeurons);

	delete[] g;
}

void NeuronNet::calculateGradientErrorFunction(double **gradientWeightsLayer1, double **gradientWeightsLayer2) {
	double *sigmaLayer2 = new double[numberOutputNeurons];
	double *summa = new double[numberHiddenNeurons];
	double *dActFuncHiddenLayer = new double[numberHiddenNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			sigmaLayer2[j] = outputsNet[j] - outputY[j];
			gradientWeightsLayer2[s][j] = sigmaLayer2[j] * hiddenOutputs[s];
		}
	}
	
	for (int s = 0; s < numberHiddenNeurons; s++) {
		dActFuncHiddenLayer[s] = deriviateHyperbolicTangent(hiddenOutputs[s]);
	}

	for (int s = 0; s < numberHiddenNeurons; s++) {
		summa[s] = 0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			summa[s] += sigmaLayer2[j] * weightsLayer2[s][j];
		}
	}

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			gradientWeightsLayer1[i][s] = dActFuncHiddenLayer[s] * summa[s] * inputX[i];
		}
	}

	delete[] sigmaLayer2;
	delete[] summa;
	delete[] dActFuncHiddenLayer;	
}

void NeuronNet::correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate) {

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			weightsLayer1[i][s] -= learningRate * gradientWeightsLayer1[i][s];
		}
	}

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			weightsLayer2[s][j] -= learningRate * gradientWeightsLayer2[s][j];
		}
	}
}

void NeuronNet::backPropagation(double learningRate) {

	double **gradientWeightsLayer1, **gradientWeightsLayer2;
	gradientWeightsLayer1 = new double* [numberInputNeurons];
	for (int i = 0; i < numberInputNeurons; i++)
		gradientWeightsLayer1[i] = new double [numberHiddenNeurons];

	gradientWeightsLayer2 = new double* [numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++)
		gradientWeightsLayer2[s] = new double [numberOutputNeurons];

	calculateOutputs();
	calculateGradientErrorFunction(gradientWeightsLayer1, gradientWeightsLayer2);
	correctWeights(gradientWeightsLayer1, gradientWeightsLayer2, learningRate);

	for (int i = 0; i < numberInputNeurons; i++)
		delete[] gradientWeightsLayer1[i];
	delete[] gradientWeightsLayer1;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] gradientWeightsLayer2[i];
	delete[] gradientWeightsLayer2;
}

double NeuronNet::calculateValueErrorFunction(double **trainData, double *trainLabel, int numberTrainImage) {
	double crossEntropy = 0;

	for (int image = 0; image < numberTrainImage; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			inputX[i] = trainData[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			outputY[j] = 0;
		}
		outputY[(int)trainLabel[image]] = 1;

		calculateOutputs();

		for (int j = 0; j < numberOutputNeurons; j++) {
			crossEntropy += outputY[j] * log(outputsNet[j]);
		}
	}

	crossEntropy = -1 * crossEntropy / numberTrainImage;

	return crossEntropy;

}

void NeuronNet::setRandomOrder(int *order, int size) {
	int randomNumber, tmp;
	for (int i = 0; i < size; i++) {
		order[i] = i;
	}

	for (int i = 0; i < size; i++) {
		randomNumber = i + rand() % (size - i);
		tmp = order[i];
		order[i] = order[randomNumber];
		order[randomNumber] = tmp;
	}
}

void NeuronNet::trainNeuronNetwork(double **trainData, double *trainLabel, int numberTrainImage, int numberEpochs, double learningRate, double errorCrossEntropy) {
	
	double currentCrossEntropy = 0;
	int numberImage = 0;

	int *order = new int[numberTrainImage];

	for (int epoch = 0; epoch < numberEpochs; epoch++) {
		printf("# epoch = %d \n", epoch);

		setRandomOrder(order, numberTrainImage);

		for (int image = 0; image < numberTrainImage; image++) {
			numberImage = order[image];
			for (int i = 0; i < numberInputNeurons; i++) {
				inputX[i] = trainData[numberImage][i];
			}

			for (int j = 0; j < numberOutputNeurons; j++) {
				outputY[j] = 0;
			}
			outputY[(int)trainLabel[numberImage]] = 1;

			backPropagation(learningRate);
		}

		currentCrossEntropy = calculateValueErrorFunction(trainData, trainLabel, numberTrainImage);
		printf("    currentCrossEntropy = %f \n", currentCrossEntropy);
		
 		if (currentCrossEntropy < errorCrossEntropy) {
			break;
		}
	}

	delete[] order;
}

double NeuronNet::calculatePrecision(double **data, double *label, int numberImage) {
	double precision = 0;
	int truePositive = 0, falsePositive = 0;
	int maxIndex;

	for (int image = 0; image < numberImage; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			inputX[i] = data[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			outputY[j] = 0;
		}
		outputY[(int)label[image]] = 1;

		calculateOutputs();

		maxIndex = 0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			if (outputsNet[j] > outputsNet[maxIndex]) {
				maxIndex = j;
			}
		}		

		if (outputY[maxIndex] == 1.0) {
			truePositive++;
		}
		else {
			falsePositive++;
		}
	}

	precision = (double)truePositive / (double)(truePositive + falsePositive);

	return precision;
}


