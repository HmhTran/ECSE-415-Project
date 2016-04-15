#ifndef EigenfacesTest_H
#define EigenfacesTest_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "Eigenfaces.h"
#include "K_Fold_Cross_Set.h"

using namespace cv;
using namespace std;

void kFoldEigenfacesTest1(K_Fold_Cross_Set kset);
void kFoldEigenfacesTest1extra(K_Fold_Cross_Set kset, int x);
void kFoldEigenfacesTest2(K_Fold_Cross_Set kset);
void kFoldEigenfacesTest2extra(K_Fold_Cross_Set kset, int x);
void kFoldEigenfacesTest3(K_Fold_Cross_Set kset, int x);
void displayImage(Mat image, string imageName, bool x);
void rand2Num(int &x1, int &x2, int range);

#endif