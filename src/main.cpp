#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "Face_Dataset.h"
#include "K_Fold_Cross_Set.h"
#include "EigenfacesTest.h"
//#include "../test/test.h"

using namespace cv;
using namespace std;

void main(void)
{
	string pathQMUL = "../QMUL";
	string pathPose = "../HeadPoseImageDatabase";
	Face_Dataset fset(pathQMUL, pathPose);
	if(!fset.isSuccessfullyLoaded())
	{
		cout << "An error has occurred in loading Face Dataset" << endl;
		return;
	}

	K_Fold_Cross_Set kset = fset.get7FoldCrossSetQMUL();

	string pathResource = " ";
	fset.dispImageSetQMUL("YongminY", pathResource);
	fset.dispImageSetPose(15, 2, pathResource);
}