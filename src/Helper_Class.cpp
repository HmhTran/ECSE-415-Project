#include "Helper_Class.h"

Mat ImageAnnot::annotImage()
{
	Mat output;
	aImage.copyTo(output);
	rectangle(output, aAnnot, Scalar(255, 0, 255), 5);
	return output;
}

Mat ImageAnnot::cropImage()
{
	Mat output;
	Mat roi = aImage(aAnnot);
	roi.copyTo(output);
	return output;
}

ImageAnnot::ImageAnnot(Mat pImage, Rect pAnnot)
{
	aImage = pImage;
	aAnnot = pAnnot;
}

int Pose::tilt()
{
	return aTilt;
}

int Pose::pan()
{
	return aPan;
}

string Pose::str()
{
	return '(' + to_string(aTilt) + ',' + to_string(aPan) + ')';
}

Pose::Pose(int pTilt, int pPan)
{
	aTilt = pTilt;
	aPan = pPan;	
}

string Label::name()
{
	return aName;
}

Pose Label::pose()
{
	return aPose;
}

string Label::str()
{
	return aName + ',' + to_string(aPose.tilt()) + ',' + to_string(aPose.pan());
}

Label::Label(string pName, int tilt, int pan)
{
	aName = pName;
	aPose = Pose(tilt, pan);
}

Label::Label(string pName, Pose pPose)
{
	aName = pName;
	aPose = pPose;
}

Mat ImageSample::image()
{
	return aImage;
}

Label ImageSample::label()
{
	return aLabel;
}

string ImageSample::name()
{
	return aLabel.name();
}

Pose ImageSample::pose()
{
	return aLabel.pose();
}

int ImageSample::tilt()
{
	return aLabel.pose().tilt();
}

int ImageSample::pan()
{
	return aLabel.pose().pan();
}

ImageSample::ImageSample(Mat pImage, string pName, int pTilt, int pPan)
{
	aImage = pImage;
	Pose aPose = Pose(pTilt, pPan);
	aLabel = Label(pName, aPose);
}

ImageSample::ImageSample(Mat pImage, string pName, Pose pPose)
{
	aImage = pImage;
	aLabel = Label(pName, pPose);
}

ImageSample::ImageSample(Mat pImage, Label pLabel)
{
	aImage = pImage;
	aLabel = pLabel;
}

ImageSample::ImageSample(Mat pImage)
{
	aImage = pImage;
	aLabel = Label();
}

string NormDistrib::name()
{
	return aName;
}

Mat NormDistrib::mean()
{
	return aMean;
}
	
Mat NormDistrib::covar()
{
	return aCovar;
}
	
double NormDistrib::gaussian(Mat x)
{
	const double pi = 3.14159265358979323846;

	double n = x.rows;
	exp((aMean.t().dot(aCovar.inv() * aMean)) / -2) * sqrt(pow(2*pi, n) * determinant(aCovar));
}

NormDistrib::NormDistrib(string pName, Mat pMean, Mat pCovar)
{
	aName = pName;
	aMean = pMean;
	aCovar = pCovar;
}

NormDistrib::NormDistrib(string pName, Mat data)
{
	calcCovarMatrix(data, aCovar, aMean, CV_COVAR_ROWS | CV_COVAR_NORMAL);
}


bool comparePose(Pose pPose1, Pose pPose2)
{
	if (pPose1.tilt() == pPose2.tilt()) return pPose1.pan() < pPose2.pan();
	else return pPose1.tilt() < pPose2.tilt();
}

bool compareLabel(Label pLabel1, Label pLabel2)
{
	if (pLabel1.name() == pLabel2.name()) return comparePose(pLabel1.pose(), pLabel2.pose());
	else return pLabel1.name() < pLabel2.name();
}

bool compareImgByLabel(ImageSample sample1, ImageSample sample2)
{
	return  compareLabel(sample1.label(), sample2.label());
}

void writeMatrix(Mat m, string filePath)
{
	int rows = m.rows;
	int cols = m.cols;

	ofstream file;
	file.open(filePath, ios::trunc);
	double *ptrRow;
	string line;

	for (int i = 0; i < rows; i++)
	{
		ptrRow = m.ptr<double>(i);
		for (int j = 0; j < cols; j++)
		{
			
		}
	}
}