#ifndef Eigenfaces_H
#define Eigenfaces_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "Helper_Class.h"

using namespace cv;
using namespace std;

class Eigenfaces
{
public:
	Eigenfaces();
	void eigenfacesTestSetup(Mat pEeigenspace, vector<ImageSample> pTrainSet);
	bool addTrainSet(vector<ImageSample> trainSet);
	bool clearTrainSet();
	bool train(vector<ImageSample> trainSet);
	bool train();
	bool setK(int x);
	void getFacesTrain(vector<ImageSample> &output);
	void getEigenfacesTrain(vector<ImageSample> &output);
	void reconstructDemo(Mat image, Mat &output);
	void recognition(ImageSample sample, string &answer, ImageSample &match);
	void displayEigenspace();
	bool fitNormalDistrib();
	void recognitionProb(ImageSample query, string &answer);
	void recognitionPose(ImageSample query, double &errorMin);

private:
	int k;
	Mat eigenspace;
	Mat meanImage;

	vector<ImageSample> eigenTrainSet;
	vector<NormDistrib> eigenNormDistrib;

	void computeEigenspace();
	void vectorize(Mat image, Mat &output, bool x);
	void project(Mat image, Mat &output);
	void reconstruct(Mat image, Mat &output);
};

#endif