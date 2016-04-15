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
	/*cout << "\tLoading Pose Dataset" << endl;
	numPerPose = 15;
	numSerPose = 2;
	deltaPose = 15;
	if(!loadPose(pathPose)) return;
	cout << "\t\tDone!" << endl;*/

	successfullyLoaded = true;
	cout << "Dataset successfully loaded: QMUL Dataset and Pose Dataset" << endl << endl;
}

bool Face_Dataset::isSuccessfullyLoaded()	{  return successfullyLoaded; }

void Face_Dataset::dispImageSetQMUL(string subject)
{
	int index = subjectToIndexQMUL(subject); 
	if (index < 0) return;
		
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
	if(subject.find("NoGlasses") != string::npos) subject = subject.substr(0, subject.find("NoGlasses"));
	string imgName = subject + "'s Face";
	namedWindow(imgName, WINDOW_NORMAL);
	resizeWindow(imgName, output.cols/scale, output.rows/scale);
	imshow(imgName, output);
	waitKey(0);
	destroyWindow(imgName);
}

void Face_Dataset::dispImageSetPose(int person, int series)
{
	if (person < 1 || person > numPerPose || 
		series < 1 || series > numSerPose)
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
			image = Pose[per][ser][i][j].annotImage();
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

void Face_Dataset::getImageQMUL(string subject, int tilt, int pan, ImageSample &output)
{
	int indexSubject = subjectToIndexQMUL(subject); 
	if (indexSubject < 0) return;

	int indexTilt, indexPan;
	poseToIndex(tilt, pan, deltaQMUL, indexTilt, indexPan);

	output = ImageSample(QMUL[indexSubject][indexTilt][indexPan], subject, tilt, pan);
}

void Face_Dataset::getImageSubjectQMUL(string subject, vector<ImageSample> &output)
{
	int indexSubject = subjectToIndexQMUL(subject); 
	if (indexSubject < 0) return;

	int rows = (int) QMUL[0].size();
	int cols = (int) QMUL[0][0].size();

	int count = 0;
	output = vector<ImageSample>(rows*cols);
	if(subject.find("NoGlasses") != string::npos) subject = subject.substr(0, subject.find("NoGlasses"));
	ImageSample image;
	int tilt, pan;

	for (int i = 0; i < rows; i++)
	{
		tilt = indexToAngle(i, maxTilt, deltaQMUL);
		for (int j = 0; j < cols; j++)
		{
			pan = indexToAngle(i, maxPan, deltaQMUL);
			output[count] = ImageSample(QMUL[indexSubject][i][j], subject, tilt, pan);
			count++;
		}			
	}
}
	
void Face_Dataset::getImagePoseQMUL(int tilt, int pan, vector<ImageSample> &output)
{
	int indexTilt, indexPan;
	poseToIndex(tilt, pan, deltaQMUL, indexTilt, indexPan);
		
	int numPers = (int) QMUL.size();
	output = vector<ImageSample>(numPers);
	string subject;	

	for (int per = 0; per < numPers; per++)
	{
		subject = subjectsQMUL[per];
		if(subject.find("NoGlasses") != string::npos) subject = subject.substr(0, subject.find("NoGlasses"));
		output[per] = ImageSample(QMUL[per][indexTilt][indexPan], subjectsQMUL[per], tilt, pan);
	}
}

void Face_Dataset::getImagePose(int person, int series, int tilt, int pan, ImageSample &output)
{
	if (person < 1 || person > numPerPose || 
		series < 1 || series > 2)
	{
		cout << "Series " << series <<" Images of Person" << setfill('0') << setw(2) << person << "'s faces not found" << endl;
		return;
	}
	int per = person - 1;
	int ser = series - 1;

	int indexTilt, indexPan;
	poseToIndex(tilt, pan, deltaPose, indexTilt, indexPan);

	output = ImageSample(Pose[per][ser][indexTilt][indexPan].cropImage(), indexToPersonPose(per), tilt, pan);
}

void Face_Dataset::getImagePosePose(int tilt, int pan, vector<ImageSample> &output)
{
	int indexTilt, indexPan;
	poseToIndex(tilt, pan, deltaPose, indexTilt, indexPan);
	
	int numPers = (int) Pose.size();
	int numSer = (int) Pose[0].size();

	int count = 0;
	output = vector<ImageSample>(numPers*numSer);
	string person;

	for (int per = 0; per < numPers; per++)
	{
		person = indexToPersonPose(per);
		for (int ser = 0; ser < numSer; ser++)
		{
			output[count] = ImageSample(Pose[per][ser][indexTilt][indexPan].cropImage(), person, tilt, pan);
			count++;
		}
	}
}

void Face_Dataset::getSettingsQMUL(vector<string> &subjects, vector<int> &tilt, vector<int> &pan)
{
	getSubjectsQMUL(subjects);
	getPoseQMUL(tilt, pan);
}

void Face_Dataset::getSubjectsQMUL(vector<string> &subjects)
{
	subjects = subjectsQMUL;
}

void Face_Dataset::getPoseQMUL(vector<int> &tilt, vector<int> &pan)
{
	getTiltQMUL(tilt);
	getPanQMUL(pan);
}

void Face_Dataset::getTiltQMUL(vector<int> &tilt)
{
	getAngles(maxTilt, deltaQMUL, tilt);
}

void Face_Dataset::getPanQMUL(vector<int> &pan)
{
	getAngles(maxPan, deltaQMUL, pan);
}

void Face_Dataset::printSettingsQMUL()
{
	printSubjectsQMUL();
	printPoseQMUL();
}

void Face_Dataset::printSubjectsQMUL()
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

void Face_Dataset::getSettingsPose(vector<string> &person,  vector<int> &series, vector<int> &tilt, vector<int> &pan)
{
	getPersonPose(person);
	getSeriesPose(series);
	getPosePose(tilt, pan);
}

void Face_Dataset::getPersonPose(vector<string> &person)
{
	person = vector<string>(numPerPose);

	for(int i = 0; i < numPerPose; i++)
	{
		person.push_back(indexToPersonPose(i));
	}
}
void Face_Dataset::getSeriesPose(vector<int> &ser)
{
	ser = vector<int>(numSerPose);

	for(int i = 0; i < numSerPose; i++)
	{
		ser.push_back(i+1);
	}
}

void Face_Dataset::getPosePose(vector<int> &tilt, vector<int> &pan)
{
	getTiltPose(tilt);
	getPanPose(pan);
}

void Face_Dataset::getTiltPose(vector<int> &tilt)
{getAngles(maxTilt, deltaPose, tilt);
}

void Face_Dataset::getPanPose(vector<int> &pan)
{
	getAngles(maxPan, deltaPose, pan);
}

void Face_Dataset::printSettingsPose()
{
	printPersonPose();
	printSeriesPose();
	printPosePose();
}

void Face_Dataset::printPersonPose()
{
	cout << "List of Subjects in QMUL Dataset" << endl;
	
	for (int i = 0; i < numPerPose; i++)
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
	K_Fold_Cross_Set kset(7);
	K_Fold_Cross_Set ksetTemp(7);

	vector<ImageSample> sampleSet;
	int numPer = (int) subjectsQMUL.size();

	for (int per = 0; per < numPer; per++)
	{
		getImageSubjectQMUL(subjectsQMUL[per], sampleSet);

		if(!ksetTemp.addSet(sampleSet))
		{
			cout << "Unable to add images from QMUL to 7-fold cross set" << endl;
			return NULL;
		}

		if (!ksetTemp.create())
		{
			cout << "Unable to create 7-fold cross set for "  << subjectsQMUL[per] << " for QMUL" << endl;
			return NULL;
		}
		kset.addFoldSet(ksetTemp);

	}

	return kset;
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
	string subject, path, fileName, filePath;

	for (int i = 0; i < perSize; i++)
	{
		subject = subjectsQMUL[i];
		path = pathQMUL + '/' + subject + "Grey/";
		if (subject == "JeffN") subject = "JeffNG";
		if (subject == "Jon") subject = "OngEJ";

		for (int tilt = 60; tilt <= 120; tilt+=deltaQMUL)
		{
			for (int pan = 0; pan <= 180; pan+=deltaQMUL)
			{
				fileName = subject + "_" + formatIntStr(tilt, 3) + "_" + formatIntStr(pan, 3) + ".ras"; 
				filePath = path + fileName;
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
	Rect annot;
	vector<ImageAnnot> tempPan;
	vector<vector<ImageAnnot>> tempTilt;
	vector<vector<vector<ImageAnnot>>> tempSer;

	int count;
	string tiltPlus, panPlus;
	string line, path, fileName, filePath, imagePath, annotPath;
		
	int x;
	int y;
	int width = 100;
	int height = 100;

	for (int per = 1; per <= numPerPose; per++)
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
					annot = Rect(x - width/2, y - height/2, width, height);
					
					tempPan.push_back(ImageAnnot(image, annot));
					count++;
				}

				tempTilt.push_back(tempPan);
				tempPan.clear();
			}

			tempSer.push_back(tempTilt);
			tempTilt.clear();
		}

		Pose.push_back(tempSer);
		tempSer.clear();
	}

	return true;
}

string Face_Dataset::formatIntStr(int i, int width)
{
	stringstream output;
	output << setfill('0') << setw(width) << i;
	return output.str();
}

string Face_Dataset::indexToPersonPose(int index)
{
	return "Person" + formatIntStr(index - 1, 2);
}

int Face_Dataset::subjectToIndexQMUL(string subject)
{
	vector<string>::iterator iter = find(subjectsQMUL.begin(), subjectsQMUL.end(), subject);
	if (iter == subjectsQMUL.end())
	{
		cout << "Images of " << subject << "'s faces not found" << endl;
		return -1;
	}
	int index = distance(subjectsQMUL.begin(), iter);
}

void Face_Dataset::poseToIndex(int tilt, int pan, int delta, int &indexTilt, int &indexPan)
{
	if (tilt < -maxTilt || tilt > maxTilt || tilt % deltaPose != 0 || 
		pan < -maxPan || pan > maxPan || pan % deltaPose != 0)
	{
		cout << "Images of faces with tilt angle of " << tilt << " degrees and pan angle of " << " degrees not found" << endl;
		return;
	}

	indexTilt = angleToIndex(tilt, maxTilt, deltaQMUL);
	indexPan = angleToIndex(pan, maxPan, deltaQMUL);
}

int Face_Dataset::angleToIndex(int angle, int max, int delta)
{
	return (angle + max) / delta;
}

void Face_Dataset::indexToPose(int indexTilt, int indexPan, int delta, int &tilt, int &pan)
{
	if (tilt < -maxTilt || tilt > maxTilt || tilt % deltaPose != 0 || 
		pan < -maxPan || pan > maxPan || pan % deltaPose != 0)
	{
		cout << "Images of faces with tilt angle of " << tilt << " degrees and pan angle of " << " degrees not found" << endl;
		return;
	}

	indexTilt = angleToIndex(tilt, maxTilt, deltaPose);
	indexPan = angleToIndex(pan, maxPan, deltaPose);
}

int Face_Dataset::indexToAngle(int index, int max, int delta)
{
	return -max + (index*delta);
}

void Face_Dataset::getAngles(int max, int delta, vector<int> &output)
{
	int x = -max;

	for(int i = x; i <= max; x+=delta)
	{
		output.push_back(x);
	}
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