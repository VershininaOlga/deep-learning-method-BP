#include "config.h"
#include <stdio.h>
#include "string.h"

TaskConfig task;

void output_setting()
{
	printf("############# parameters task #############\n");
    printf("path to fileTrainImageMNIST = %s \n", task.fileTrainImageMNIST);
	printf("path to fileTrainLabelsMNIST = %s \n", task.fileTrainLabelsMNIST);
	printf("path to fileTestImageMNIST = %s \n", task.fileTestImageMNIST);
	printf("path to fileTestLabelsMNIST = %s \n", task.fileTestLabelsMNIST);
	printf("numberHiddenNeurons = %d \n", task.numberHiddenNeurons);
	printf("numberEpochs = %d \n", task.numberEpochs);
	printf("learningRate = %f \n", task.learningRate);
	printf("errorCrossEntropy = %f \n", task.errorCrossEntropy);
}

void set_path(char *s, char *value)
{
	if(strcmp(s, "fileTrainImageMNIST,") == 0)
	{
		strcpy(task.fileTrainImageMNIST, value);
	}
	if(strcmp(s, "fileTrainLabelsMNIST,") == 0)
	{
		strcpy(task.fileTrainLabelsMNIST, value);
	}
	if(strcmp(s, "fileTestImageMNIST,") == 0)
	{
		strcpy(task.fileTestImageMNIST, value);
	}
	if(strcmp(s, "fileTestLabelsMNIST,") == 0)
	{
		strcpy(task.fileTestLabelsMNIST, value);
	} 
}

void set_param(char *s, double value)
{
	if(strcmp(s, "numberHiddenNeurons,") == 0)
	{
		task.numberHiddenNeurons = (int)value;
	}
	if(strcmp(s, "numberEpochs,") == 0)
	{
		task.numberEpochs = (int)value;
	}
	if(strcmp(s, "learningRate,") == 0)
	{
		task.learningRate = value;
	}
	if(strcmp(s, "errorCrossEntropy,") == 0)
	{
		task.errorCrossEntropy = value;
	}
}

void read_file_conf(FILE *f_conf)
{
	char s[500];
	char val_str[500];
	double val = 0;
	int cnt = 0;
	int k = 0;
	while(!feof(f_conf))
	{
		cnt = fscanf(f_conf, "%s", s);
		k++;
		if ((cnt > 0) && (k <= 4))
		{
			fscanf(f_conf, "%s", val_str);
			set_path(s, val_str);
		}
		
		if ((cnt > 0) && (k > 4))
		{
			fscanf(f_conf, "%lf", &val);
			set_param(s, val);
		}
		
	}
  
	fclose(f_conf);
}

void read_config()
{
  FILE * f_conf = fopen("config.txt", "r");
  if(f_conf == NULL)
  {
    printf("Use default setting \n");
  } 
  else
  {
    read_file_conf(f_conf);
  }
  
  output_setting();  
}