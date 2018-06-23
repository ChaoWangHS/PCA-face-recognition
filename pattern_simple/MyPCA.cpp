#include "MyPCA.h"

#define SHOW_IMAGE 0

MyPCA::MyPCA(vector<string>& _facesPath) //���캯�������init
{
	init(_facesPath); //�����õĺ�������ֵ����void��ֵ����ô���ݵģ���
}

void MyPCA::init(vector<string>& _facesPath)
{
	cout << "MyPCA::init=>" << _facesPath[0] << endl;
    getImgSize(_facesPath);
    imgRows = imread(_facesPath[0],0).rows;
	

    mergeMatrix(_facesPath);
    getAverageVector();
    subtractMatrix();
    Mat _covarMatrix = (subFacesMatrix.t()) * subFacesMatrix; //�����������ͼƬ������������ͼƬά�����������һ�ַ������Խ���������⡣
	                                                          //�鿴֪�����ӣ�https://zhuanlan.zhihu.com/p/26652435
    getBestEigenVectors(_covarMatrix);
}

void MyPCA:: getImgSize(vector<string>& _facesPath)
{
    Mat sampleImg = imread(_facesPath[0], 0); //_facesPath[0] ���� D:\\pattern_SUBMIT\\faces\\s\\14.bmp
    if (sampleImg.empty()) {
		cout << "Fail to Load Image in PCA" << "_facesPath:" << _facesPath[0] << endl;
    }
    //Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;  //���˽�б���
    //cout << "Per Image Size is: " << size << endl;
}
//put all face images to one matrix, order in column
void MyPCA::mergeMatrix(vector<string>& _facesPath) //�൱�ڰ����е�ͼƬ���ŵ� allFacesMatrix ��
{
    int col = int(_facesPath.size()); //һ������һ��
    allFacesMatrix.create(imgSize, col, CV_32FC1);
    
    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = allFacesMatrix.col(i); //ǳ����������������
//ע�⣺ǳ���� -  ����������ֻ��������ͷ�����ݹ�������a,b,c������һ�����������2������ͬ�������ã�  
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
    reduce(allFacesMatrix, avgVector, 1, CV_REDUCE_AVG);//avgVector�����˽�б���
    
    if (SHOW_IMAGE) {
        avgVector.reshape(0, imgRows).copyTo(face);
        //Just for display face
        normalize(face, face, 0, 1, cv::NORM_MINMAX);
        namedWindow("AverageFace", CV_WINDOW_NORMAL);
		cout << "AverageFace Done! " << endl;
        imshow("AverageFace", face);
		
    }
}

void MyPCA::subtractMatrix() //https://www.cnblogs.com/hadoop2015/p/7419087.html ���ıȽ���ϸ
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
	eigenVector =  subFacesMatrix * allEigenVectors ; //����֪����˵��https://zhuanlan.zhihu.com/p/26652435 ����Ӧ�ò�����������
	                                                  //�����ｵά��
	cout << "_covarMatrix size :" << _covarMatrix.size() << endl;
	cout << "allEigenVectors size :" << allEigenVectors.size() << endl;
	cout << "eigenVector = allEigenVectors * (subFacesMatrix):" << endl;
	cout << "����������������" << eigenVector.rows << "*" << eigenVector.cols << " = " << (subFacesMatrix).rows << "*" << (subFacesMatrix).cols << "  " << allEigenVectors.rows << "*" << allEigenVectors.cols << endl;
 
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.cols; i++ ) 
	{
		
        Mat tempVec = eigenVector.col(i); //ǳ�������ı�tempVec���൱�ڸı�eigenVector
        normalize(tempVec, tempVec);
    }
	
    if (SHOW_IMAGE) { //����������ʱ��Ҫ���ϱߵ� //Normalize eigenvectors ���ѭ��ע�͵�����Ȼ��һ�������һ�ź�
        //Display eigen face
        Mat eigenFaces, allEigenFaces;
		Mat copy;
		eigenVector.copyTo(copy);//�������������ֹ�޸�ԭ�ȵ�
		//��Ϊ�����cols����ѭ��������� OpenCV Error : Image step is wrong(The matrix is not continuous, thus its number of rows can not be changed) un cv::Mat::reshape
		copy=copy.t();
		for (int i = 0; i < copy.rows; i++) {
            copy.row(i).reshape(0, imgRows).copyTo(eigenFaces);
			imwrite("eigenFaces" + to_string(i) + ".bmp", eigenFaces);
            normalize(eigenFaces, eigenFaces, 0, 1, cv::NORM_MINMAX);
            if(i == 0) //ֻ��һ��ͼ
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

/*ȥ��ƽ��ֵ
����Э�������
����Э������������ֵ����������
������ֵ����
����ǰN����������ֵ��Ӧ����������
������ת��������õ���N�����������������¿ռ��У�ʵ��������ѹ����*/

