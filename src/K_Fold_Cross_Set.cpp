#include "K_Fold_Cross_Set.h"

K_Fold_Cross_Set::K_Fold_Cross_Set(int x)
{
	if (x < 2) 
	{
		cout << "k-fold cross set must have at least 2 subsets" << endl;
		k = 0;
		return;
	}

	k = x;
}

bool K_Fold_Cross_Set::add(Mat m, string info)
{
	bufferSet.push_back(m);
	bufferInfo.push_back(info);

	return true;
}

bool K_Fold_Cross_Set::addSet(vector<Mat> vM, vector<string> vInfo)
{
	if(vM.size() != vInfo.size())
	{
		cout << "Sample Size and Info Size must be equal" << endl;
		return false;
	}

	bufferSet.insert(bufferSet.end(), vM.begin(), vM.end());
	bufferInfo.insert(bufferInfo.end(), vInfo.begin(), vInfo.end());

	return true;
}

bool K_Fold_Cross_Set::clearAll()
{
	clearBuffer();
	clearSet();

	return true;
}

bool K_Fold_Cross_Set::clearBuffer()
{
	bufferSet.clear();
	bufferInfo.clear();

	return true;
}

bool K_Fold_Cross_Set::clearSet()
{
	trainSet.clear();
	trainInfo.clear();
	testSet.clear();
	testInfo.clear();

	return true;
}

bool K_Fold_Cross_Set::create()
{
	int size = (int) bufferSet.size();
	if (size < k) {
		cout << k << "-fold cross set must have at least " << k << " elements" << endl;
		return false;
	}
	if (size%k != 0)
	{
		cout << k << "-fold cross set must have elements divisble by " << k << endl;
		cout << "Add " << k - (size%k) << " elements to the set" << endl;
		return false;
	}

	clearSet();

	int q = size / k;

	randomShuffle();

	trainSet = vector<vector<Mat>>(k-1, vector<Mat>(q));
	trainInfo = vector<vector<string>>(k-1, vector<string>(q));
	testSet = vector<Mat>(q);
	testInfo = vector<string>(q);

	int shuffledIndex;
	int count = 0;

	for (int i = 0; i < k-1; i++)
	{
		for (int j = 0; j < q; j++)
		{
			shuffledIndex = indices[count];
			trainSet[i][j] = bufferSet[shuffledIndex];
			trainInfo[i][j] = bufferInfo[shuffledIndex];

			count++;
		}
	}

	for (int i = 0; i < q; i++)
	{
		shuffledIndex = indices[count];
		testSet[i] = bufferSet[shuffledIndex];
		testInfo[i] = bufferInfo[shuffledIndex];

		count++;
	}

	clearBuffer();
	indices.clear();

	return true;
}

void K_Fold_Cross_Set::getAllSet(vector<vector<Mat>> &outputTrainSet, vector<vector<string>> &outputTrainInfo, vector<Mat> &outputTestSet, vector<string> &outputTestInfo)
{
	getTrainSet(outputTrainSet, outputTrainInfo);
	getTestSet(outputTestSet, outputTestInfo);
}

void K_Fold_Cross_Set::getTrainSet(vector<vector<Mat>> &outputSet, vector<vector<string>> &outputInfo)
{
	outputSet = trainSet;
	outputInfo = trainInfo;
}

void K_Fold_Cross_Set::getTestSet(vector<Mat> &outputSet, vector<string> &outputInfo)
{
	outputSet = testSet;
	outputInfo = testInfo;
}

void K_Fold_Cross_Set::getTrainSetAt(int setIndex, vector<Mat> &outputSet, vector<string> &outputInfo)
{
	outputSet = trainSet[setIndex];
	outputInfo = trainInfo[setIndex];
}

void K_Fold_Cross_Set::trainSetAt(int setIndex, int index, Mat &m, string &str)
{
	m = trainSet[setIndex][index];
	str = trainInfo[setIndex][index];
}

void K_Fold_Cross_Set::testSetAt(int index, Mat &m, string &str)
{
	m = testSet[index];
	str = testInfo[index];
}
	
void K_Fold_Cross_Set::randomShuffle()
{
	int size = (int) bufferSet.size();
	for (int i = 0; i < size; i++) indices.push_back(i);
	random_shuffle(indices.begin(), indices.end());		
}