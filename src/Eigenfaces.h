#ifndef Eigenfaces_H
#define Eigenfaces_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;

class Eigenfaces
{
public:
	Eigenfaces();
	void eigenfacesTestSetup(Mat pEeigenspace, vector<Mat> pEigenTrainSet, vector<string> pTrainSet);
	bool addTrainSet(vector<Mat> trainSet, vector<string> labels);
	bool clearTrainSet();
	bool train(vector<Mat> trainSet, vector<string> labels);
	bool train();
	bool setK(int x);
	void getFacesTrain(vector<Mat> &output);
	void getEigenfacesTrain(vector<Mat> &output);
	void reconstructDemo(Mat image, Mat &output);
	void recognition(Mat image, string &answer, Mat &match);
	void displayEigenspace();
	bool fitNormalDistrib();
	void recognitionProb(Mat image, string &answer);
	void recognitionPose(Mat image, int &errorMin);

private:
	int k;
	Mat eigenspace;
	Mat meanImage;

	typedef struct
	{
		Mat img;
		string name;
	}Sample;
	vector<Sample> eigenTrainSet;

	typedef struct
	{
		string name;
		Mat mean;
		Mat covar;
	}NormDistrib;
	vector<NormDistrib> eigenNormDistrib;

	void computeEigenspace();
	void vectorize(Mat image, Mat &output, bool x);
	void project(Mat image, Mat &output);
	void reconstruct(Mat image, Mat &output);
	bool sampleCompare(Sample s1, Sample s2);
	double gaussian(Mat x, Mat mean, Mat covar);
};

#endif