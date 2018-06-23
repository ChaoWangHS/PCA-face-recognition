#include "WriteTrainData.h"

WriteTrainData::WriteTrainData(MyPCA _trainPCA, vector<string>& _trainFacesID)
{
	cout << "WriteTrainData......:" << _trainPCA.getEigenvectors().size() << endl;
    numberOfFaces = _trainPCA.getFacesMatrix().cols;
    trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
    project(_trainPCA);
    writeTrainFacesData(_trainFacesID);
    writeMean(_trainPCA.getAverage());
    writeEigen(_trainPCA.getEigenvectors());
}

void WriteTrainData::project(MyPCA _trainPCA)
{
    //cout << "Write Class"<<_trainPCA.getFacesMatrix().size() << endl;
    Mat facesMatrix = _trainPCA.getFacesMatrix();
    Mat avg = _trainPCA.getAverage();
    Mat eigenVec = _trainPCA.getEigenvectors();
	cout << "eigenVec size: " << eigenVec .size()<< endl;
    for (int i = 0; i < numberOfFaces; i++) {
        Mat temp;
        Mat projectFace = trainFacesInEigen.col(i);
        subtract(facesMatrix.col(i), avg, temp);
		cout << "WriteTrainData.cpp==> projectFace = eigenVec.t() * temp;" << endl;
        projectFace = eigenVec.t() * temp;
		cout << projectFace.rows << "*" << projectFace.cols << " = " << eigenVec.cols << "*" << eigenVec.rows << "  " << temp.rows << "*" << temp.cols << endl;

    }
    //cout << trainFacesInEigen.col(0).size() <<endl;
}

void WriteTrainData::writeTrainFacesData(vector<string>& _trainFacesID)
{
    string facesDataPath = "D:\\pattern\\data\\facesdata.txt";
    ofstream writeFaceFile(facesDataPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeFaceFile) {
        cout << "Fail to open file: " << facesDataPath << endl;
    }
    
    for (int i = 0; i < _trainFacesID.size(); i++) {
        //writeFaceFile << i + 1 << "#";
        writeFaceFile << _trainFacesID[i] << ":";
        for (int j = 0; j < trainFacesInEigen.rows; j++) {
            writeFaceFile << trainFacesInEigen.col(i).at<float>(j);
            writeFaceFile << " ";
        }
        writeFaceFile << "\n";
    }
    
    writeFaceFile.close();
}

void WriteTrainData::writeMean(Mat avg)
{
    string meanPath = "D:\\pattern\\data\\mean.txt";
    ofstream writeMeanFile(meanPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeMeanFile) {
        cout << "Fail to open file: " << meanPath << endl;
    }
    
    for (int i = 0; i < avg.rows; i++) {
        writeMeanFile << avg.at<float>(i);
        writeMeanFile << " ";
    }
    
    writeMeanFile.close();
}

void WriteTrainData::writeEigen(Mat eigen)
{
    string eigenPath = "D:\\pattern\\data\\eigen.txt";
    ofstream writeEigenFile(eigenPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeEigenFile) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
	cout << "存进去的特征向量：WriteTrainData::writeEigen: eigen shape:  " << eigen.rows << "*" << eigen.cols << endl;
	Mat eigen_deep_copy = eigen.clone(); //我怀疑是存储格式错了，于是按照原先的格式进行存储
	                                     //思路：先进行一个深拷贝，因为我怕改变了原先的eigen，再转置下
	                                     //这个主要是因为我在获取特征向量时和原来的代码有个转置的关系
	eigen_deep_copy = eigen_deep_copy.t();
	for (int i = 0; i < eigen_deep_copy.rows; i++) {
		for (int j = 0; j < eigen_deep_copy.cols; j++) {
			writeEigenFile << eigen_deep_copy.row(i).at<float>(j);
            writeEigenFile << " ";
        }
        writeEigenFile << "\n";
    }
    
    writeEigenFile.close();
}

Mat WriteTrainData::getFacesInEigen()
{
    return trainFacesInEigen;
}

WriteTrainData::~WriteTrainData() {}
