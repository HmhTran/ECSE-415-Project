#ifndef K_Fold_Cross_Set_H
#define K_Fold_Cross_Set_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;

class K_Fold_Cross_Set
{
public:
	K_Fold_Cross_Set(int x);
	void add(Mat m, string info);
	void addSet(vector<Mat> vM, vector<string> vInfo);
	void clearAll();
	void clearBuffer();
	void clearSet();
	void create();
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