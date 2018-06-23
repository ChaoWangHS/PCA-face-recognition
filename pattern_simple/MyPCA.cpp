#include "MyPCA.h"

#define SHOW_IMAGE 0

MyPCA::MyPCA(vector<string>& _facesPath) //构造函数里调用init
{
	init(_facesPath); //但调用的函数返回值都是void，值是怎么传递的？？
}

void MyPCA::init(vector<string>& _facesPath)
{
	cout << "MyPCA::init=>" << _facesPath[0] << endl;
    getImgSize(_facesPath);
    imgRows = imread(_facesPath[0],0).rows;
	

    mergeMatrix(_facesPath);
    getAverageVector();
    subtractMatrix();
    Mat _covarMatrix = (subFacesMatrix.t()) * subFacesMatrix; //针对这种样本图片数量严重少于图片维数的情况，有一种方法可以解决这类问题。
	                                                          //查看知乎链接：https://zhuanlan.zhihu.com/p/26652435
    getBestEigenVectors(_covarMatrix);
}

void MyPCA:: getImgSize(vector<string>& _facesPath)
{
    Mat sampleImg = imread(_facesPath[0], 0); //_facesPath[0] 形如 D:\\pattern_SUBMIT\\faces\\s\\14.bmp
    if (sampleImg.empty()) {
		cout << "Fail to Load Image in PCA" << "_facesPath:" << _facesPath[0] << endl;
    }
    //Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;  //类的私有变量
    //cout << "Per Image Size is: " << size << endl;
}
//put all face images to one matrix, order in column
void MyPCA::mergeMatrix(vector<string>& _facesPath) //相当于把所有的图片都放到 allFacesMatrix 了
{
    int col = int(_facesPath.size()); //一条数据一列
    allFacesMatrix.create(imgSize, col, CV_32FC1);
    
    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = allFacesMatrix.col(i); //浅拷贝。。。呃呃呃
//注意：浅拷贝 -  不复制数据只创建矩阵头，数据共享（更改a,b,c的任意一个都会对另外2个产生同样的作用）  
//		Mat a;
//		Mat b = a; //a "copy" to b  
//		Mat c(a); //a "copy" to c  
        //Load grayscale image 0
        Mat tmpImg;
        imread(_facesPath[i], 0).convertTo(tmpImg, CV_32FC1);
        //convert to 1D matrix
        tmpImg.reshape(1, imgSize).copyTo(tmpMatrix);
    }
    //cout << "Merged Matix(Width, Height): " << mergedMatrix.size() << endl;
}

//compute average face
void MyPCA::getAverageVector()
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    Mat face;
    reduce(allFacesMatrix, avgVector, 1, CV_REDUCE_AVG);//avgVector是类的私有变量
    
    if (SHOW_IMAGE) {
        avgVector.reshape(0, imgRows).copyTo(face);
        //Just for display face
        normalize(face, face, 0, 1, cv::NORM_MINMAX);
        namedWindow("AverageFace", CV_WINDOW_NORMAL);
		cout << "AverageFace Done! " << endl;
        imshow("AverageFace", face);
		
    }
}

void MyPCA::subtractMatrix() //https://www.cnblogs.com/hadoop2015/p/7419087.html 讲的比较详细
{
    allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < subFacesMatrix.cols; i++) {
        subtract(subFacesMatrix.col(i), avgVector, subFacesMatrix.col(i));
    }
}

void MyPCA::getBestEigenVectors(Mat _covarMatrix)
{
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenValues, allEigenVectors;
    eigen(_covarMatrix, allEigenValues, allEigenVectors);
    
	//eigenVector = allEigenVectors * (subFacesMatrix.t());
	eigenVector =  subFacesMatrix * allEigenVectors ; //根据知乎上说，https://zhuanlan.zhihu.com/p/26652435 ，它应该才是特征向量
	                                                  //在这里降维的
	cout << "_covarMatrix size :" << _covarMatrix.size() << endl;
	cout << "allEigenVectors size :" << allEigenVectors.size() << endl;
	cout << "eigenVector = allEigenVectors * (subFacesMatrix):" << endl;
	cout << "真正的特征向量：" << eigenVector.rows << "*" << eigenVector.cols << " = " << (subFacesMatrix).rows << "*" << (subFacesMatrix).cols << "  " << allEigenVectors.rows << "*" << allEigenVectors.cols << endl;
 
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.cols; i++ ) 
	{
		
        Mat tempVec = eigenVector.col(i); //浅拷贝，改变tempVec就相当于改变eigenVector
        normalize(tempVec, tempVec);
    }
	
    if (SHOW_IMAGE) { //看特征脸的时候要把上边的 //Normalize eigenvectors 这个循环注释掉，不然归一化后就是一团黑
        //Display eigen face
        Mat eigenFaces, allEigenFaces;
		Mat copy;
		eigenVector.copyTo(copy);//先做个深拷贝，防止修改原先的
		//因为如果对cols进行循环，会出现 OpenCV Error : Image step is wrong(The matrix is not continuous, thus its number of rows can not be changed) un cv::Mat::reshape
		copy=copy.t();
		for (int i = 0; i < copy.rows; i++) {
            copy.row(i).reshape(0, imgRows).copyTo(eigenFaces);
			imwrite("eigenFaces" + to_string(i) + ".bmp", eigenFaces);
            normalize(eigenFaces, eigenFaces, 0, 1, cv::NORM_MINMAX);
            if(i == 0) //只有一张图
                allEigenFaces = eigenFaces;
            else				
                hconcat(allEigenFaces, eigenFaces, allEigenFaces);
			
        }
        
        namedWindow("EigenFaces", CV_WINDOW_NORMAL);
        imshow("EigenFaces", allEigenFaces);
		
    }
}

Mat MyPCA::getFacesMatrix()
{
    return allFacesMatrix;
}

Mat MyPCA::getAverage()
{
    return avgVector;
}

Mat MyPCA::getEigenvectors()
{
	return eigenVector;
}

Mat MyPCA::getFacesInEigen()
{
	return eigenVector.t()*subFacesMatrix;
}

MyPCA::~MyPCA() {}

/*去除平均值
计算协方差矩阵
计算协方差矩阵的特征值和特征向量
将特征值排序
保留前N个最大的特征值对应的特征向量
将数据转换到上面得到的N个特征向量构建的新空间中（实现了特征压缩）*/

