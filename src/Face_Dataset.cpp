#include "Face_Dataset.h"

Face_Dataset::Face_Dataset(string pathQMUL, string pathPose)
{
	cout << "Loading Face Dataset" << endl;
	successfullyLoaded = false;
	maxTilt = 30;
	maxPan = 90;

	// Load QMUL Dataset
	cout << "\tLoading QMUL Dataset" << endl;
	if(!loadSubjectsQMUL()) return;
	deltaQMUL = 10;
	if(!loadQMUL(pathQMUL)) return;
	cout << "\t\tDone!" << endl;

	// Load Pose Dataset
	cout << "\tLoading Pose Dataset" << endl;
	numPersPose = 15;
	deltaPose = 15;
	if(!loadPose(pathPose)) return;
	cout << "\t\tDone!" << endl;

	successfullyLoaded = true;
	cout << "Dataset successfully loaded: QMUL Dataset and Pose Dataset" << endl << endl;
}

bool Face_Dataset::isSuccessfullyLoaded()	{  return successfullyLoaded; }

void Face_Dataset::dispImageSetQMUL(string subject)
{
	vector<string>::iterator iter = find(subjectsQMUL.begin(), subjectsQMUL.end(), subject);
	if (iter == subjectsQMUL.end())
	{
		cout << "Images of " << subject << "'s faces not found" << endl;
		return;
	}
	int index = distance(subjectsQMUL.begin(), iter);
		
	Mat output, temp, image;
	int rows = (int) QMUL[0].size();
	int cols = (int) QMUL[0][0].size();

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			QMUL[index][i][j].copyTo(image);
			image = image.t();
			temp.push_back(image);
		}

		temp = temp.t();
		output.push_back(temp);
		temp = Mat();
	}

	int scale = 2;
	string imgName = subject + "'s Face";
	namedWindow(imgName, WINDOW_NORMAL);
	resizeWindow(imgName, output.cols/scale, output.rows/scale);
	imshow(imgName, output);
	waitKey(0);
	destroyWindow(imgName);
}

void Face_Dataset::dispImageSetPose(int person, int series)
{
	if (person < 1 || person > numPersPose || 
		series < 1 || series > 2)
	{
		cout << "Series " << series <<" Images of Person" << setfill('0') << setw(2) << person << "'s faces not found" << endl;
		return;
	}
	int per = person - 1;
	int ser = series - 1;
	string perID = to_string(person);
	if (person < 10) perID = '0' + perID;
		
	Mat output, temp, image;
	int rows = (int) Pose[0][0].size();
	int cols = (int) Pose[0][0][0].size();

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			Pose[per][ser][i][j].copyTo(image);
			Rect annotation = annotPose[per][ser][i][j];
			rectangle(image, annotation, Scalar(255, 0, 255), 5);
			image = image.t();
			temp.push_back(image);
		}

		temp = temp.t();
		output.push_back(temp);
		temp = Mat();
	}

	int scale = 5;
	string imgName = "Person" + perID + "'s Face Series " + to_string(series);
	namedWindow(imgName, WINDOW_NORMAL);
	resizeWindow(imgName, output.cols/scale, output.rows/scale);
	imshow(imgName, output);
	waitKey(0);
	destroyWindow(imgName);
}

void Face_Dataset::getImageSubjectQMUL(string subject, vector<Mat> &output)
{
	vector<string>::iterator iter = find(subjectsQMUL.begin(), subjectsQMUL.end(), subject);
	if (iter == subjectsQMUL.end())
	{
		cout << "Images of " << subject << "'s faces not found" << endl;
		return;
	}
	int index = distance(subjectsQMUL.begin(), iter);

	int rows = (int) QMUL[0].size();
	int cols = (int) QMUL[0][0].size();

	int count = 0;
	output = vector<Mat>(rows*cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output[count] = QMUL[index][i][j];
			count++;
		}			
	}
}
	
void Face_Dataset::getImagePoseQMUL(int tilt, int pan, vector<Mat> &output)
{
	if (tilt < -maxTilt || tilt > maxTilt || tilt % deltaQMUL != 0 || 
		pan < -maxPan || pan > maxPan || pan % deltaQMUL != 0)
	{
		cout << "Images of faces with tilt angle of " << tilt << " degrees and pan angle of " << " degrees not found" << endl;
		return;
	}
	int indexTilt = (tilt + maxTilt) / deltaQMUL;
	int indexPan = (pan + maxPan) / deltaQMUL;
		
	int numPers = (int) QMUL.size();
	output = vector<Mat>(numPers);

	for (int per = 0; per < numPers; per++)
	{
		output[per] = QMUL[per][indexTilt][indexPan];
	}
}

void Face_Dataset::getImagePosePose(int tilt, int pan, vector<Mat> &output)
{
	if (tilt < -maxTilt || tilt > maxTilt || tilt % deltaPose != 0 || 
		pan < -maxPan || pan > maxPan || pan % deltaPose != 0)
	{
		cout << "Images of faces with tilt angle of " << tilt << " degrees and pan angle of " << " degrees not found" << endl;
		return;
	}
	int indexTilt = (tilt + maxTilt) / deltaPose;
	int indexPan = (pan + maxPan) / deltaPose;
	
	int numPers = (int) Pose.size();
	int numSer = (int) Pose[0].size();

	int count = 0;
	Mat image, roi;
	output = vector<Mat>(numPers*numSer);

	for (int per = 0; per < numPers; per++)
	{
		for (int ser = 0; ser < numSer; ser++)
		{
			image = Pose[per][ser][indexTilt][indexPan];
			roi = image(annotPose[per][ser][indexTilt][indexPan]);
			output[count] = roi;
			count++;
		}
	}
}

void Face_Dataset::printSettingsQMUL()
{
	printSubjectQMUL();
	printPoseQMUL();
}

void Face_Dataset::printSubjectQMUL()
{
	int size = (int) subjectsQMUL.size();
	cout << "List of Subjects in QMUL Dataset" << endl;
	
	for (int i = 0; i < size; i++)
	{
		cout << subjectsQMUL[i] << endl;
	}

	cout << endl;
}

void Face_Dataset::printPoseQMUL()
{
	printTiltQMUL();
	printPanQMUL();
}

void Face_Dataset::printTiltQMUL()
{
	cout << "List of Tilt Angles in QMUL Dataset" << endl;
	printAngles(maxTilt, deltaQMUL);
}

void Face_Dataset::printPanQMUL()
{
	cout << "List of Pan Angles in QMUL Dataset" << endl;
	printAngles(maxPan, deltaQMUL);
}

void Face_Dataset::printSettingsPose()
{
	printSubjectPose();
	printSeriesPose();
	printPosePose();
}

void Face_Dataset::printSubjectPose()
{
	cout << "List of Subjects in QMUL Dataset" << endl;
	
	for (int i = 0; i < numPersPose; i++)
	{
		cout << "Person" << setfill('0') << setw(2) << i+1 << endl;
	}

	cout << endl;
}

void Face_Dataset::printSeriesPose()
{
	int size = (int) Pose[0].size();
	cout << "List of Series in QMUL Dataset" << endl;
		
	for (int i = 0; i < size; i++)
	{
		cout << "Series " << i+1 << endl;
	}

	cout << endl;
}

void Face_Dataset::printPosePose()
{
	printTiltPose();
	printPanPose();
}

void Face_Dataset::printTiltPose()
{
	cout << "List of Tilt Angles in Pose Dataset" << endl;
	printAngles(maxTilt, deltaPose);
}

void Face_Dataset::printPanPose()
{
	cout << "List of Pan Angles in Pose Dataset" << endl;
	printAngles(maxTilt, deltaPose);
}

K_Fold_Cross_Set Face_Dataset::get7FoldCrossSetQMUL()
{
	K_Fold_Cross_Set kSet(7);

	int numPers = (int) QMUL.size();
	int numTilt = (int) QMUL[0].size();
	int numPan = (int) QMUL[0][0].size();

	string subject;
	vector<string> info;

	for (int i = 0; i < numPers; i++)
	{
		subject = subjectsQMUL[i];

		for (int j = 0; j < numPan; j++)
		{
			info.push_back(subject);
		}

		for (int j = 0; j < numTilt; j++)
		{
			if(!kSet.addSet(QMUL[i][j], info))
			{
				cout << "Unable to add images from QMUL to 7-fold cross set" << endl;
				return NULL;
			}
		}

		info.clear();
	}

	if(!kSet.create())
	{
		cout << "Unable to create 7-fold cross set for QMUL" << endl;
		return NULL;
	}
	return kSet;
}

bool Face_Dataset::loadSubjectsQMUL()
{
	subjectsQMUL.push_back("AdamB");
	subjectsQMUL.push_back("AndreeaV");
	subjectsQMUL.push_back("CarlaB");
	subjectsQMUL.push_back("ColinP");
	subjectsQMUL.push_back("DanJ");
	subjectsQMUL.push_back("DennisP");
	subjectsQMUL.push_back("DennisPNoGlasses");
	subjectsQMUL.push_back("DerekC");
	subjectsQMUL.push_back("GrahamW");
	subjectsQMUL.push_back("HeatherL");
	subjectsQMUL.push_back("Jack");
	subjectsQMUL.push_back("JamieS");
	subjectsQMUL.push_back("JeffN");
	subjectsQMUL.push_back("John");
	subjectsQMUL.push_back("Jon");
	subjectsQMUL.push_back("KateS");
	subjectsQMUL.push_back("KatherineW");
	subjectsQMUL.push_back("KeithC");
	subjectsQMUL.push_back("KrystynaN");
	subjectsQMUL.push_back("PaulV");
	subjectsQMUL.push_back("RichardB");
	subjectsQMUL.push_back("RichardH");
	subjectsQMUL.push_back("SarahL");
	subjectsQMUL.push_back("SeanG");
	subjectsQMUL.push_back("SeanGNoGlasses");
	subjectsQMUL.push_back("SimonB");
	subjectsQMUL.push_back("SueW");
	subjectsQMUL.push_back("TasosH");
	subjectsQMUL.push_back("TomK");
	subjectsQMUL.push_back("YogeshR");
	subjectsQMUL.push_back("YongminY");

	return true;
}

bool Face_Dataset::loadQMUL(string pathQMUL)
{
	Mat image;
	vector<Mat> tempPan;
	vector<vector<Mat>> tempTilt;

	int perSize = (int) subjectsQMUL.size();
	string subject, fileName, filePath;

	for (int i = 0; i < perSize; i++)
	{
		subject = subjectsQMUL[i];

		for (int tilt = 60; tilt <= 120; tilt+=deltaQMUL)
		{
			for (int pan = 0; pan <= 180; pan+=deltaQMUL)
			{
				fileName = subject + "_" + formatIntStr(tilt, 3) + "_" + formatIntStr(pan, 3) + ".ras"; 
				filePath = pathQMUL + '/' + subject + "Grey/" + fileName;
				image = imread(filePath);
				if (!image.data)
				{
					cout << "\t\tError loading image in " << filePath << " for QMUL" << endl;
					return false;
				}
				tempPan.push_back(image);
			}

			tempTilt.push_back(tempPan);
			tempPan.clear();
		}

		QMUL.push_back(tempTilt);
		tempTilt.clear();
	}

	return true;
}

bool Face_Dataset::loadPose(string pathPose)
{
	Mat image;
	vector<Mat> tempPan;
	vector<vector<Mat>> tempTilt;
	vector<vector<vector<Mat>>> tempSer;

	Rect annotRect;
	vector<Rect> annotPan;
	vector<vector<Rect>> annotTilt;
	vector<vector<vector<Rect>>> annotSer;

	int count;
	string tiltPlus, panPlus;
	string line, fileName, filePath, imagePath, annotPath;
		
	int x;
	int y;
	int width = 100;
	int height = 100;

	for (int per = 1; per <= numPersPose; per++)
	{
		for (int ser = 1; ser <= 2; ser++)
		{
			count = 14;
			tiltPlus = "";

			for (int tilt = -30; tilt <= 30; tilt+=deltaPose)
			{
				if (tilt >= 0) tiltPlus = '+';
				panPlus = "";

				for (int pan = -90; pan <= 90; pan+=deltaPose)
				{
					if (pan >= 0) panPlus = '+';

					fileName = "person" + formatIntStr(per, 2) + to_string(ser) + formatIntStr(count, 2) + 
						tiltPlus + to_string(tilt) + panPlus + to_string(pan);
					filePath = pathPose + "/Person" + formatIntStr(per, 2) +'/' + fileName;

					imagePath = filePath + ".jpg";
					image = imread(imagePath);
					if (!image.data)
					{
						cout << "\t\tError loading image in " << imagePath << endl;
						return false;
					}
					tempPan.push_back(image);

					annotPath = filePath + ".txt";
					ifstream annotIfs(annotPath);
					if (!annotIfs.is_open())
					{
						cout << "\t\tError: Error loading annotation in " << annotPath << endl;
						return false;
					}
					getline(annotIfs, line);
					getline(annotIfs, line);
					getline(annotIfs, line);
					annotIfs >> x;
					annotIfs >> y;
					annotRect = Rect(x - width/2, y - height/2, width, height);
					annotPan.push_back(annotRect);

					count++;
				}

				tempTilt.push_back(tempPan);
				tempPan.clear();
				annotTilt.push_back(annotPan);
				annotPan.clear();
			}

			tempSer.push_back(tempTilt);
			tempTilt.clear();
			annotSer.push_back(annotTilt);
			annotTilt.clear();
		}

		Pose.push_back(tempSer);
		tempSer.clear();
		annotPose.push_back(annotSer);
		annotSer.clear();
	}

	return true;
}

string Face_Dataset::formatIntStr(int i, int width)
{
	stringstream output;
	output << setfill('0') << setw(width) << i;
	return output.str();
}

void Face_Dataset::printAngles(int max, int delta)
{
	int x = -max;
	cout << '[' << x;

	for (x+=delta; x <= max; x+=delta)
	{
		cout << ", " << x;
	}

	cout << ']' << endl << endl;
}