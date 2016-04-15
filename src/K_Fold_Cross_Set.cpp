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

	foldSet = vector<vector<ImageSample>>(k);
}

bool K_Fold_Cross_Set::add(ImageSample sample)
{
	dataSet.push_back(sample);

	return true;
}

bool K_Fold_Cross_Set::addSet(vector<ImageSample> sampleSet)
{
	dataSet.insert(dataSet.end(), sampleSet.begin(), sampleSet.end());

	return true;
}

bool K_Fold_Cross_Set::create()
{
	int size = (int) dataSet.size();
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

	clearFoldSet();

	int q = size / k;

	randomShuffle();

	foldSet = vector<vector<ImageSample>>(k, vector<ImageSample>(q));

	int shuffledIndex;
	int count = 0;

	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < q; j++)
		{
			shuffledIndex = indices[count];
			foldSet[i][j] = dataSet[shuffledIndex];

			count++;
		}
	}

	clearDataSet();
	indices.clear();

	return true;
}

bool K_Fold_Cross_Set::addFoldSet(K_Fold_Cross_Set kset)
{
	vector<vector<ImageSample>> pFoldSet;
	kset.getFoldSet(pFoldSet);

	int pK = (int) pFoldSet.size();

	if (pK != k)
	{
		cout << "New Fold Set size must be of size: "<< k << endl;
		return false;
	}
	
	for (int i = 0; i < k; i++)
	{
		foldSet[i].insert(foldSet[i].end(), pFoldSet[i].begin(), pFoldSet[i].end()); 
	}

	return true;
}

bool K_Fold_Cross_Set::addFoldSet(vector<vector<ImageSample>> pFoldSet)
{
	int pK = (int) pFoldSet.size();
	if (pK != k)
	{
		cout << "New Fold Set size must be of size: "<< k << endl;
		return false;
	}

	int sizeSub = pFoldSet[0].size();
	for (int i = 1; i < pK; i++)
	{
		if ((int) pFoldSet[i].size() != sizeSub)
		{
			cout << "All subvectors in new Fold Set must have equal size" << endl;
			return false;
		}
	}
	
	for (int i = 0; i < k; i++)
	{
		foldSet[i].insert(foldSet[i].end(), pFoldSet[i].begin(), pFoldSet[i].end()); 
	}

	return true;
}

void K_Fold_Cross_Set::fold(int index, vector<ImageSample> &trainSet, vector<ImageSample> &testSet)
{
	if (index < 0 || index > k-1)
	{
		cout << "index must be from 0 to " << k-1 << endl;
		return;
	}

	trainSet.clear();
	testSet.clear();

	vector<ImageSample> inputSet;

	for (int i = 0; i < k; i++)
	{
		inputSet = foldSet[i];

		if (i == index)
		{
			testSet.insert(testSet.end(), inputSet.begin(), inputSet.end());
		}
		else
		{
			trainSet.insert(trainSet.end(), inputSet.begin(), inputSet.end());
		}
	}
}

void K_Fold_Cross_Set::getFoldSet(vector<vector<ImageSample>> &output)
{
	output = foldSet;
}

void K_Fold_Cross_Set::getFoldSetAt(int setIndex, vector<ImageSample> &output)
{
	output = foldSet[setIndex];
}

void K_Fold_Cross_Set::writeKFoldSet(string filePath)
{
	int q = (int) foldSet[0].size();

	ofstream file;
	file.open(filePath, ios::trunc);
	string line;

	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < q; j++)
		{
			line = foldSet[i][j].label().str() + ' ';
		}
		file << line << endl;
	}
	file.close();
}

bool K_Fold_Cross_Set::clearAll()
{
	clearDataSet();
	clearFoldSet();

	return true;
}

bool K_Fold_Cross_Set::clearDataSet()
{
	dataSet.clear();

	return true;
}

bool K_Fold_Cross_Set::clearFoldSet()
{
	foldSet.clear();

	return true;
}

void K_Fold_Cross_Set::randomShuffle()
{
	int size = (int) dataSet.size();
	for (int i = 0; i < size; i++) indices.push_back(i);
	random_shuffle(indices.begin(), indices.end());		
}