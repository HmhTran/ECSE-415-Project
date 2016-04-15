#ifndef Face_Dataset_H
#define Face_Dataset_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "Helper_Class.h"
#include "K_Fold_Cross_Set.h"

using namespace cv;
using namespace std;

class Face_Dataset
{
public:
	Face_Dataset(string pathQMUL, string pathPose);
	bool isSuccessfullyLoaded();

	void dispImageSetQMUL(string subject, string filePath);
	void dispImageSetPose(int person, int series, string filePath);

	void getImageQMUL(string subject, int tilt, int pan, ImageSample &output);
	void getImageSubjectQMUL(string subject, vector<ImageSample> &output);
	void getImagePoseQMUL(int tilt, int pan, vector<ImageSample> &output);

	void getImagePose(int person, int series, int tilt, int pan, ImageSample &output);
	void getImagePosePose(int tilt, int pan, vector<ImageSample> &output);

	void getSettingsQMUL(vector<string> &subjects, vector<int> &tilt, vector<int> &pan);
	void getSubjectsQMUL(vector<string> &subjects);
	void getPoseQMUL(vector<int> &tilt, vector<int> &pan);
	void getTiltQMUL(vector<int> &tilt);
	void getPanQMUL(vector<int> &pan);

	void printSettingsQMUL();
	void printSubjectsQMUL();
	void printPoseQMUL();
	void printTiltQMUL();
	void printPanQMUL();

	void getSettingsPose(vector<string> &person,  vector<int> &series, vector<int> &tilt, vector<int> &pan);
	void getPersonPose(vector<string> &subjects);
	void getSeriesPose(vector<int> &ser);
	void getPosePose(vector<int> &tilt, vector<int> &pan);
	void getTiltPose(vector<int> &tilt);
	void getPanPose(vector<int> &pan);

	void printSettingsPose();
	void printPersonPose();
	void printSeriesPose();
	void printPosePose();
	void printTiltPose();
	void printPanPose();

	K_Fold_Cross_Set get7FoldCrossSetQMUL();
	K_Fold_Cross_Set get7FoldCrossSetQMULx();

private:
	bool successfullyLoaded;
	int maxTilt;
	int maxPan;

	vector<string> subjectsQMUL;
	int deltaQMUL;
	vector<vector<vector<Mat>>> QMUL;

	int numPerPose;
	int numSerPose;
	int deltaPose;
	vector<vector<vector<vector<ImageAnnot>>>> Pose;

	bool loadSubjectsQMUL();
	bool loadQMUL(string pathQMUL);
	bool loadPose(string pathPose);

	string formatIntStr(int i, int width);
	int subjectToIndexQMUL(string subject);
	string indexToPersonPose(int index);

	void poseToIndex(int tilt, int pan, int delta, int &indexTilt, int &indexPan);
	int angleToIndex(int angle, int max, int delta);

	void indexToPose(int indexTilt, int indexPan, int delta, int &tilt, int &pan);
	int indexToAngle(int index, int max, int delta);

	void getAngles(int max, int delta, vector<int> &output);
	void printAngles(int max, int delta);
};

#endif