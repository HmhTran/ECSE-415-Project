#ifndef Face_Dataset_H
#define Face_Dataset_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "K_Fold_Cross_Set.h"

using namespace cv;
using namespace std;

class Face_Dataset
{
public:
	Face_Dataset(string pathQMUL, string pathPose);
	bool isSuccessfullyLoaded();
	void dispImageSetQMUL(string subject);
	void dispImageSetPose(int person, int series);
	void getImageSubjectQMUL(string subject, vector<Mat> &output);
	void getImagePoseQMUL(int tilt, int pan, vector<Mat> &output);
	void getImagePosePose(int tilt, int pan, vector<Mat> &output);
	void printSettingsQMUL();
	void printSubjectQMUL();
	void printPoseQMUL();
	void printTiltQMUL();
	void printPanQMUL();
	void printSettingsPose();
	void printSubjectPose();
	void printSeriesPose();
	void printPosePose();
	void printTiltPose();
	void printPanPose();
	K_Fold_Cross_Set get7FoldCrossSetQMUL();

private:
	bool successfullyLoaded;
	int maxTilt;
	int maxPan;

	vector<string> subjectsQMUL;
	int deltaQMUL;
	vector<vector<vector<Mat>>> QMUL;

	int numPersPose;
	int deltaPose;
	vector<vector<vector<vector<Mat>>>> Pose;
	vector<vector<vector<vector<Rect>>>> annotPose;

	bool loadSubjectsQMUL();
	bool loadQMUL(string pathQMUL);
	bool loadPose(string pathPose);
	string formatIntStr(int i, int width);
	void printAngles(int max, int delta);
};

#endif