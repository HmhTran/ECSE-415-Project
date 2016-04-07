#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "Face_Dataset.h"

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

	fset.get7FoldCrossSetQMUL();
}