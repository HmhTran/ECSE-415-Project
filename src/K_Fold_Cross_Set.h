#ifndef K_Fold_Cross_Set_H
#define K_Fold_Cross_Set_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "Helper_Class.h"

using namespace cv;
using namespace std;

class K_Fold_Cross_Set
{
public:
	K_Fold_Cross_Set(int x);
	bool add(ImageSample sample);
	bool addSet(vector<ImageSample> sampleSet);
	bool create();
	bool addFoldSet(K_Fold_Cross_Set kset);
	bool addFoldSet(vector<vector<ImageSample>> pFoldSet);
	void fold(int index, vector<ImageSample> &trainSet, vector<ImageSample> &testSet);
	void getFoldSet(vector<vector<ImageSample>> &output);
	void getFoldSetAt(int setIndex, vector<ImageSample> &output);
	void writeKFoldSet(string filePath);
	bool clearAll();
	bool clearDataSet();
	bool clearFoldSet();

private:
	int k;
	vector<int> indices;
	vector<ImageSample> dataSet;
	vector<vector<ImageSample>> foldSet;

	void randomShuffle();
};

#endif