#include "EigenfacesTest.h"

void kFoldEigenfacesTest1(K_Fold_Cross_Set kset)
{
	int kFold = 7;
	vector<ImageSample> trainSet;
	vector<ImageSample> testSet;

	int kMax = 31*6*19;
	int kTestSet[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, kMax};
	int kSize = 13;
	int k;

	Eigenfaces efaces = Eigenfaces();
	
	double kFoldErrorTrain[13];
	double errorTrain;
	double kFoldErrorTest[13];
	double errorTest;

	int sizeTrain;
	int sizeTest;

	vector<ImageSample> reconstructFacesTrain;
	ImageSample imageTest, reconstructFaceTest;

	for (int i = 0; i < kFold; i++)
	{
		kset.fold(i, trainSet, testSet);
		efaces.train(trainSet);

		sizeTrain = (int) trainSet.size();
		sizeTest = (int) testSet.size();

		for (int j = 0; j < kSize; j++)
		{
			k = kTestSet[j];
			efaces.setK(k);

			efaces.getEigenfacesTrain(reconstructFacesTrain);
			
			errorTrain = 0;

			for (int i = 0; i < sizeTrain; i++)
			{
				errorTrain += norm(trainSet[i].image(), reconstructFacesTrain[i].image());
			}

			errorTrain /= sizeTrain;
			kFoldErrorTrain[j] = errorTrain;

			errorTest = 0;

			for (int i = 0; i < sizeTest; i++)
			{
				imageTest = testSet[i];
				efaces.reconstructDemo(imageTest.image(), reconstructFaceTest.image());
				errorTest += norm(testSet[i].image(), reconstructFaceTest.image());
			}

			errorTest /= sizeTest;
			kFoldErrorTest[j] = errorTest;
		}
	}

	for (int i = 0; i < kSize; i++) kFoldErrorTrain[i] /= kFold;
	cout << "7-Fold Cross Validation Eigenfaces Reconstruction Error for Training Images: " << kFoldErrorTrain << endl;

	for (int i = 0; i < kSize; i++) kFoldErrorTest[i] /= kFold;
	cout << "7-Fold Cross Validation Eigenfaces Reconstruction Error for Training Images: " << kFoldErrorTest << endl;
}

void kFoldEigenfacesTest1extra(K_Fold_Cross_Set kset, int x)
{
	int i = rand() % 7;
	vector<ImageSample> trainSet;
	vector<ImageSample> testSet;

	int kTestSet[] = {1, 2, x};
	int kSize = 3;
	int k;

	Eigenfaces efaces = Eigenfaces();

	kset.fold(i, trainSet, testSet);
	efaces.train(trainSet);

	int sizeTrain = (int) trainSet.size();
	int sizeTest = (int) testSet.size();

	Mat img;
	string imgName;

	int trainIndex1, trainIndex2, testIndex1, testIndex2;
	rand2Num(trainIndex1, trainIndex2, sizeTrain);
	rand2Num(testIndex1, testIndex2, sizeTest);

	efaces.displayEigenspace();

	vector<ImageSample> reconstructFacesTrain;
	ImageSample imageTest, reconstructFaceTest;

	for (int j = 0; j < kSize; j++)
	{
		k = kTestSet[j];
		efaces.setK(k);

		efaces.getEigenfacesTrain(reconstructFacesTrain);

		for (int i = 0; i < sizeTrain; i++)
		{
			hconcat(trainSet[i].image(), reconstructFacesTrain[i].image(), img);
			imgName = "Reconstructed Training Face " + to_string(i+1);

			if (i == trainIndex1 || i == trainIndex2) displayImage(img, imgName, true);
			else displayImage(img, imgName, false);
		}

		for (int i = 0; i < sizeTest; i++)
		{
			imageTest = testSet[i];
			efaces.reconstructDemo(imageTest.image(), reconstructFaceTest.image());

			hconcat(imageTest.image(), reconstructFaceTest.image(), img);
			imgName = "Reconstructed Test Face " + to_string(i+1);

			if (i == testIndex1 || i == testIndex2) displayImage(img, imgName, true);
			else displayImage(img, imgName, false);
		}

	}
}

void kFoldEigenfacesTest2(K_Fold_Cross_Set kset)
{
	int kFold = 7;
	vector<ImageSample> trainSet;
	vector<ImageSample> testSet;

	int kMax = 31*6*19;
	int kTestSet[] = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, kMax};
	int kSize = 13;
	int k;

	Eigenfaces efaces = Eigenfaces();
	
	double kFoldRecognitionRate[13];
	double recognitionRate;

	ImageSample query;
	string answer;

	int sizeTest;
	
	for (int i = 0; i < kFold; i++)
	{
		kset.fold(i, trainSet, testSet);
		efaces.train(trainSet);
		sizeTest = (int) testSet.size();

		for (int j = 0; j < kSize; j++)
		{
			k = kTestSet[j];
			efaces.setK(k);

			recognitionRate = 0;

			for (int i = 0; i < sizeTest; i++)
			{
				query = testSet[i];
				efaces.recognition(query, answer, ImageSample());
				if(answer == query.name()) recognitionRate++;
			}

			recognitionRate /= sizeTest;
			kFoldRecognitionRate[j] = recognitionRate;
		}
	}

	for (int i = 0; i < kSize; i++) kFoldRecognitionRate[i] /= kFold;
	cout << "7-Fold Cross Validation Recognition Rate for Eigenfaces: " << kFoldRecognitionRate << endl;
}

void kFoldEigenfacesTest2extra(K_Fold_Cross_Set kset, int x)
{
	int i = rand() % 7;
	vector<ImageSample> trainSet;
	vector<ImageSample> testSet;

	int k = x;

	Eigenfaces efaces = Eigenfaces();

	kset.fold(i, trainSet, testSet);
	efaces.train(trainSet);
	efaces.setK(k);

	int sizeTest = (int) testSet.size();
	int testIndex1 = rand() % sizeTest;
	int testIndex2 = testIndex1 + (((rand() % (sizeTest-1))+1) % sizeTest);

	ImageSample query, match;
	Mat img;
	string answer, imgName;

	vector<Mat> correctFaces, wrongFaces;
	vector<string> correctNames, wrongNames;

	for (int i = 0; i < sizeTest; i++)
	{
		query = testSet[i];
		efaces.recognition(query, answer, match);

		hconcat(query.image(), match.image(), img);
		imgName = query.name() + '/' + answer;

		if(answer == query.name())
		{
			correctFaces.push_back(img);
			correctNames.push_back(imgName);
		}
		else
		{
			wrongFaces.push_back(img);
			wrongNames.push_back(imgName);
		}
	}

	int sizeCorrect = (int) correctFaces.size();
	int sizeWrong = (int) wrongFaces.size();

	int indexCorrect1, indexCorrect2, indexWrong1, indexWrong2;
	rand2Num(indexCorrect1, indexCorrect2, sizeCorrect);
	rand2Num(indexWrong1, indexWrong2, sizeWrong);

	for (int i = 0; i < sizeCorrect; i++)
	{
		if(i == indexCorrect1 || i == indexCorrect2) displayImage(correctFaces[i], correctNames[i], true);
		else displayImage(correctFaces[i], correctNames[i], false);
	}

	for (int i = 0; i < sizeCorrect; i++)
	{
		if (i == indexWrong1 || i == indexWrong2) displayImage(wrongFaces[i], wrongNames[i], true);
		else displayImage(wrongFaces[i], wrongNames[i], false);
	}
}

void kFoldEigenfacesTest3(K_Fold_Cross_Set kset, int x)
{
	int kFold = 7;
	vector<ImageSample> trainSet;
	vector<ImageSample> testSet;

	Eigenfaces efaces = Eigenfaces();
	efaces.setK(x);

	double kFoldRecognitionRate = 0;
	double recognitionRate;

	ImageSample query;
	string answer;

	int sizeTest;
	
	for (int i = 0; i < kFold; i++)
	{
		kset.fold(i, trainSet, testSet);
		efaces.train(trainSet);
		sizeTest = (int) testSet.size();

		for (int j = 0; j < sizeTest; j++)
		{
			sizeTest = (int) testSet.size();
			recognitionRate = 0;

			for (int i = 0; i < sizeTest; i++)
			{
				query = testSet[i];
				efaces.recognitionProb(query, answer);
				if(answer == query.name()) recognitionRate++;
			}

			recognitionRate /= sizeTest;
			kFoldRecognitionRate += recognitionRate;
		}
	}

	kFoldRecognitionRate /= kFold;
	cout << "7-Fold Cross Validation Recognition Rate for Eigenfaces: " << kFoldRecognitionRate << endl;
}

void displayImage(Mat image, string imageName, bool x)
{
	string path = "Eigenfaces Test Images/";

	imshow(imageName, image);
	waitKey(0);
	destroyWindow(imageName);

	if (x) imwrite(path + imageName + ".png", image);
}

void rand2Num(int &x1, int &x2, int range)
{
	x1 = rand() % range;
	x2 = x1 + (((rand() % (range-1))+1) % range);
}