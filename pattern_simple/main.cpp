#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "ReadFile.h"
#include "GetFrame.h"
#include "MyPCA.h"
#include "FaceRecognizer.h"
#include "FaceDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    string trainListFilePath = "D:\\pattern_SUBMIT\\list\\train_list.txt";
    vector<string> trainFacesPath;
    vector<string> trainFacesID;
    vector<string> loadedFacesID;
    //read training list and ID from txt file
    readList(trainListFilePath, trainFacesPath, trainFacesID);
    //read training data(faces, eigenvector, average face) from txt file
    
    Mat frame, processed,testImg;
    namedWindow("Face Recognisation", CV_WINDOW_NORMAL);
        
        cout << "Traning Start......" << endl;
        //do PCA analysis for training faces
        MyPCA myPCA = MyPCA(trainFacesPath);
		Mat avgVec, eigenVec, facesInEigen;
		loadedFacesID = trainFacesID; //要提供一个共显示的人名
		facesInEigen = myPCA.getFacesInEigen();
		avgVec = myPCA.getAverage();
		eigenVec = myPCA.getEigenvectors();
		
        //training finsih.
        cout << "Training finsih." << endl;
        cout << "Recognise Start......" << endl;
        //Initialize capture
        GetFrame getFrame(1);
        
        vector<string> staticsID;
        string showID;
        int idCounter = 0;
        int noFace = 0;
        
        while (getFrame.getNextFrame(frame)) {
            //TO DO FACE DETECTION
            FaceDetector faceDetector;
            faceDetector.findFacesInImage(frame, processed);
            //Only recoginise faces that can be detected
            if (faceDetector.goodFace()) {
                testImg = faceDetector.getFaceToTest();
                //final step: recognize new face from training faces
		        FaceRecognizer faceRecognizer = FaceRecognizer(testImg, avgVec, eigenVec.t(), facesInEigen, loadedFacesID);
                // Show Result
                string faceID = faceRecognizer.getClosetFaceID();
                ////////////////////ID Probalilty Start////////////////
                //To reject sudden wrong recognise
                if(noFace > 8) {
                    showID = faceID;
                }
                string calID;
                int max, cnt;
                idCounter++;
                staticsID.push_back(faceID);
                if (idCounter == 8) {
                    idCounter = 0;
                    max = 0;
                    cnt = 0;
                    calID = staticsID[0];
                    for (int i = 0; i < staticsID.size(); i++) {
                        for (int j = 0; j < staticsID.size(); j++) {
                            if (staticsID[i] == staticsID[j]) {
                                cnt++;
                            }
                        }
                        if (cnt > max) {
                            max = cnt;
                            calID = staticsID[i];
                        }
                        cnt = 0;
                    }
                    staticsID.clear();
                    //4 out of 8 faces is same => update showing ID
                    if(max > 4) showID = calID;
                }
                noFace = 0;
                ////////////////////ID Probalilty END//////////////
                putText(processed, faceID, Point(10,40), 4, 1, Scalar(0,0,255));
            }else{
                noFace++;
            }
            resize(processed, processed, Size(480, 480));
            imshow("Face Recognisation", processed);
            if ( (waitKey(20) & 255) == 27 ) break;
        }
    
    cout << "Program End." << endl;
    return 0;
}
