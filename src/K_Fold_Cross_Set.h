#ifndef K_Fold_Cross_Set_H
#define K_Fold_Cross_Set_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

class K_Fold_Cross_Set
{
public:
	K_Fold_Cross_Set(int x);
	bool add(Mat m, string info);
	bool addSet(vector<Mat> vM, vector<string> vInfo);
	bool clearAll();
	bool clearBuffer();
	bool clearSet();
	bool create();
	void getAllSet(vector<vector<Mat>> &outputTrainSet, vector<vector<string>> &outputTrainInfo, vector<Mat> &outputTestSet, vector<string> &outputTestInfo);
	void getTrainSet(vector<vector<Mat>> &outputSet, vector<vector<string>> &outputInfo);
	void getTestSet(vector<Mat> &outputSet, vector<string> &outputInfo);
	void getTrainSetAt(int setIndex, vector<Mat> &outputSet, vector<string> &outputInfo);
	void trainSetAt(int setIndex, int index, Mat &m, string &str);
	void testSetAt(int index, Mat &m, string &str);

private:
	int k;
	vector<int> indices;

	vector<Mat> bufferSet;
	vector<string> bufferInfo;

	vector<vector<Mat>> trainSet;
	vector<vector<string>> trainInfo;
	vector<Mat> testSet;
	vector<String> testInfo;
	void randomShuffle();
};

#endif