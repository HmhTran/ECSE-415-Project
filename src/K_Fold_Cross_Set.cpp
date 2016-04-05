#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;

class K_Fold_Cross_Set
{
public:
	K_Fold_Cross_Set::K_Fold_Cross_Set(int x)
	{
		if (x < 2) 
		{
			cout << "k-fold cross set must have at least 2 subsets" << endl;
			return;
		}

		k = x;
	}

	void add(Mat m, string info)
	{
		bufferSet.push_back(m);
		bufferInfo.push_back(info);
	}

	void addSet(vector<Mat> vM, vector<string> vInfo)
	{
		bufferSet.insert(bufferSet.end(), vM.begin(), vM.end());
		bufferInfo.insert(bufferInfo.end(), vInfo.begin(), vInfo.end());
	}

	void clearAll()
	{
		clearBuffer();
		clearSet();
	}

	void clearBuffer()
	{
		bufferSet.clear();
		bufferInfo.clear();
	}

	void clearSet()
	{
		trainSet.clear();
		trainInfo.clear();
		testSet.clear();
		testInfo.clear();
	}

	void create()
	{
		int size = (int) bufferSet.size();
		if (size < k) {
			cout << k << "-fold cross set must have at least " << k << " elements" << endl;
			return;
		}
		if (size%k != 0)
		{
			cout << k << "-fold cross set must have elements divisble by " << k << endl;
			cout << "Add " << k - (size%k) << " elements to the set" << endl;
		}

		clearSet();

		int q = size / k;

		randomShuffle();

		Mat m;
		string info;
		int shuffledIndex, setIndex, index;	
		
		for (int i = 0; i < size; i++)
		{
			shuffledIndex = indices[i];
			m = bufferSet[shuffledIndex];
			info = bufferInfo[shuffledIndex];

			setIndex = shuffledIndex/q;
			index = shuffledIndex%q;

			if (setIndex == k-1)
			{
				testSet[index] = m;
				testInfo[index] = info;
			}
			else
			{
				trainSet[setIndex][index] = m;
				trainInfo[setIndex][index] = info;
			}
		}

		clearBuffer();
		indices.clear();
	}

	void getAllSet(vector<vector<Mat>> &outputTrainSet, vector<vector<string>> &outputTrainInfo, vector<Mat> &outputTestSet, vector<string> &outputTestInfo)
	{
		getTrainSet(outputTrainSet, outputTrainInfo);
		getTestSet(outputTestSet, outputTestInfo);
	}

	void getTrainSet(vector<vector<Mat>> &outputSet, vector<vector<string>> &outputInfo)
	{
		outputSet = trainSet;
		outputInfo = trainInfo;
	}

	void getTestSet(vector<Mat> &outputSet, vector<string> &outputInfo)
	{
		outputSet = testSet;
		outputInfo = testInfo;
	}

	void getTrainSetAt(int setIndex, vector<Mat> &outputSet, vector<string> &outputInfo)
	{
		outputSet = trainSet[setIndex];
		outputInfo = trainInfo[setIndex];
	}

	void trainSetAt(int setIndex, int index, Mat &m, string &str)
	{
		m = trainSet[setIndex][index];
		str = trainInfo[setIndex][index];
	}

	void testSetAt(int index, Mat &m, string &str)
	{
		m = testSet[index];
		str = testInfo[index];
	}
	
private:
	int k;
	vector<int> indices;

	vector<Mat> bufferSet;
	vector<string> bufferInfo;

	vector<vector<Mat>> trainSet;
	vector<vector<string>> trainInfo;
	vector<Mat> testSet;
	vector<String> testInfo;

	void randomShuffle()
	{
		int size = (int) bufferSet.size();
		for (int i = 0; i <= size; i++) indices.push_back(i);
		random_shuffle(indices.begin(), indices.end());		
	}

};