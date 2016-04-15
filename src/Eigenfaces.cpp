#include "Eigenfaces.h"

Eigenfaces::Eigenfaces(){k = 0;}

void Eigenfaces::eigenfacesTestSetup(Mat pEigenspace, vector<ImageSample> pTrainSet)
{
	eigenspace = pEigenspace;
	eigenTrainSet = pTrainSet;
}

bool Eigenfaces::addTrainSet(vector<ImageSample> trainSet)
{
	eigenTrainSet.insert(eigenTrainSet.end(), trainSet.begin(), trainSet.end());
	return true;
}

bool Eigenfaces::clearTrainSet()
{
	eigenTrainSet.clear();
}

bool Eigenfaces::train(vector<ImageSample> trainSet)
{
	eigenTrainSet.clear();
	addTrainSet(trainSet);

	train();

	return true;
}

bool Eigenfaces::train()
{
	k = 10;

	eigenspace = Mat();
	meanImage = Mat();
	
	computeEigenspace();

	return true;
}

bool Eigenfaces::setK(int x)
{
	if (x < 1)
	{
		cout << "Propsed k must be greater than 0" << endl;
		return false;
	}
	if (x > eigenspace.rows)
	{
		cout << "Propsed k must not be greater than " << eigenspace.rows << endl;
		return false;
	}

	k = x;
	return true;
}

void Eigenfaces::getFacesTrain(vector<ImageSample> &output)
{
	output = eigenTrainSet;
}

void Eigenfaces::getEigenfacesTrain(vector<ImageSample> &output)
{
	ImageSample aSample;
	Mat image, eigen;
	int size = (int) eigenTrainSet.size();
	output = vector<ImageSample>(size);

	for (int i = 0; i < size; i++)
	{
		aSample = eigenTrainSet[i];
		image = aSample.image();
		project(image, eigen);
		reconstruct(eigen, image);
		output.push_back(ImageSample(image, aSample.label()));
	}
}

void Eigenfaces::reconstructDemo(Mat image, Mat &output)
{
	Mat eigen;
	project(image, eigen);
	reconstruct(eigen, output);
}

void Eigenfaces::recognition(ImageSample sample, string &answer, ImageSample &match)
{
	Mat eigen, eigenTrain;
	project(sample.image(), eigen);

	int size = (int) eigenTrainSet.size();
	int index;
	double distance;
	double min = DBL_MAX;

	for (int i = 0; i < size; i++)
	{
		project(eigenTrainSet[i].image(), eigenTrain);

		distance = norm(eigenTrain, eigen);
		if (distance < min)
		{
			min = distance;
			index = i;
		}
	}

	ImageSample selectedSample = eigenTrainSet[index];
	answer = selectedSample.name();
	match = selectedSample;
}

void Eigenfaces::displayEigenspace()
{
	Mat image;
	image = meanImage.reshape(1, 100);
	normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);

	string imageName = "Mean Image";
	imshow(imageName, image);
	waitKey(0);
	destroyWindow(imageName);
	
	int size = 10;
	if (k < 10) size = k;
	int imageRow = eigenTrainSet[0].image().rows;
	for (int i = 0; i < size; i++)
	{
		image = eigenspace.row(i).reshape(1, imageRow);
		normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);

		imageName = "Eigenface " + to_string(i+1);
		imshow(imageName, image);
		waitKey(0);
		destroyWindow(imageName);
	}
}

bool Eigenfaces::fitNormalDistrib()
{
	eigenNormDistrib.clear();
	sort(eigenTrainSet.begin(), eigenTrainSet.end(), compareImgByLabel);

	Mat data;
	string name = eigenTrainSet[0].name();
	Mat image = eigenTrainSet[0].image();
	vectorize(image, image, true);
	data.push_back(data);

	int range = (int) eigenTrainSet.size() - 1;
	for (int i = 1; i < range; i++)
	{
		if (name != eigenTrainSet[i].name())
		{
			NormDistrib nd (name, data);
			eigenNormDistrib.push_back(nd);
			
			data = Mat();
			name = eigenTrainSet[i].name();
		}

		image = eigenTrainSet[i].image();
		vectorize(image, image, true);
		data.push_back(image);
	}
	NormDistrib nd (name, data);
	eigenNormDistrib.push_back(nd);

	return true;
}

void Eigenfaces::recognitionProb(ImageSample query, string &answer)
{
	Mat eigen;
	project(query.image(), eigen);

	double likelyhood;
	double max = 0;
	NormDistrib nd;
	int index;

	int size = (int) eigenNormDistrib.size();
	for (int i = 0; i < size; i++)
	{
		nd = eigenNormDistrib[i];
		likelyhood = nd.gaussian(eigen);

		if (likelyhood > max)
		{
			max = likelyhood;
			index = i;
		}
	}

	answer = eigenNormDistrib[index].name();
}

void Eigenfaces::recognitionPose(ImageSample query, double &errorMin)
{
	Mat eigen, eigenTrain;
	project(query.image(), eigen);

	int size = (int) eigenTrainSet.size();
	double distance;
	errorMin = DBL_MAX;

	for (int i = 0; i < size; i++)
	{
		project(eigenTrainSet[i].image(), eigenTrain);

		distance = norm(eigenTrain, eigen);
		if (distance < errorMin) errorMin = distance;
	}
}

void Eigenfaces::computeEigenspace()
{
	Mat data, image;
	int size = (int) eigenTrainSet.size();

	for (int i = 0; i < size; i++)
	{
		image = eigenTrainSet[i].image();
		vectorize(image, image, true);
		data.push_back(image);
	}
	
	Mat covar, eigenvalues, eigenvectors;
	calcCovarMatrix(data, covar, meanImage, CV_COVAR_ROWS|CV_COVAR_SCRAMBLED);

	Mat row;
	for (int i = 0; i < size; i++)
	{
		row = data.row(i);
		row -= meanImage;
	}
	meanImage = meanImage.t();

	eigen(covar, eigenvalues, eigenvectors);

	Mat vector, vectorTemp;
	for (int i = 0; i < size; i++)
	{
		vector = eigenvectors.row(i);
		vectorTemp = vector * data;
		vectorTemp = vectorTemp/norm(vectorTemp);
		eigenspace.push_back(vectorTemp);
	}

	if (size < k) k = size;
}

void Eigenfaces::vectorize(Mat image, Mat &output, bool x)
{
	cvtColor(image, image, CV_BGR2GRAY);
	image.convertTo(image, CV_64FC1);
	if (x) output = image.reshape(1, 1);
	else output = image.reshape(1, image.rows * image.cols);
}

void Eigenfaces::project(Mat image, Mat &output)
{
	vectorize(image, image, false);
	image -= meanImage;
	output = eigenspace * image;
}

void Eigenfaces::reconstruct(Mat eigen, Mat &output)
{
	output = eigenspace.t() * eigen;
	output += meanImage;

	output = output.reshape(1, 100);
	normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
}