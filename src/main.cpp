
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include "neuron_net.h"
#include "read_mnist.h"
#include "config.h"

int main(int argc, char* argv[])
{
	read_config();
	
	char *fileTrainImageMNIST = task.fileTrainImageMNIST;
	char *fileTrainLabelsMNIST = task.fileTrainLabelsMNIST;
	char *fileTestImageMNIST = task.fileTestImageMNIST;
	char *fileTestLabelsMNIST = task.fileTestLabelsMNIST;
	int numberHidden = task.numberHiddenNeurons + 1;
	int numberEpochs = task.numberEpochs;
	double learningRate = task.learningRate;
	double errorCrossEntropy = task.errorCrossEntropy;

	int width = 28, height = 28;
	int numberTrainImage = 60000;
	int numberTestImage = 10000;

	int numberInput = width * height + 1;
	int numberOutput = 10;

	double **trainData = new double*[numberTrainImage];
	for (int i = 0; i < numberTrainImage; i++)
		trainData[i] = new double[numberInput];
	readSetImage(fileTrainImageMNIST, trainData);

	double *trainLabel = new double[numberTrainImage];
	readSetLabel(fileTrainLabelsMNIST, trainLabel);

	double **testData = new double*[numberTestImage];
	for (int i = 0; i < numberTestImage; i++)
		testData[i] = new double[numberInput];
	readSetImage(fileTestImageMNIST, testData);

	double *testLabel = new double[numberTestImage];
	readSetLabel(fileTestLabelsMNIST, testLabel);

	printf("\n Run training algorithm ... \n \n");
	NeuronNet network = NeuronNet(numberInput, numberHidden, numberOutput); 
	network.trainNeuronNetwork(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	double precision = network.calculatePrecision(trainData, trainLabel, numberTrainImage);
	printf("precision train = %f \n", precision);

	precision = network.calculatePrecision(testData, testLabel, numberTestImage);
	printf("precision test = %f \n", precision);

	for (int i = 0; i < numberTrainImage; i++)
		delete[] trainData[i];
	delete[] trainData;

	for (int i = 0; i < numberTestImage; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabel;
	delete[] testLabel;
	
	printf("runtime = %f min \n", (clock()/1000.0) / 60);

	return 0;
}