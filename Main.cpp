#include "auxiliary.h" 
#include "bow.h" 
#include <opencv2/nonfree/nonfree.hpp> 
#include <iostream> 

using namespace cv;
using namespace std;

enum
{
	RANDOM_FOREST = 1,
	DECISION_TREE = 2
};

void main()
{
	/*
	������� ���������, ���������� ��������� ����� ��������� ������:

	string folder1 � ���� � �����, ���������� ������� ������ ���������;
	string folder2 � ���� � �����, ���������� ������� ������ ���������;
	string detectorType � ��� ��������� �������� �����;
	string descriptorType � ��� ������������ �������� �����;
	int vocSize � ������ �������;
	double trainProportion � ���� ��������, ������������ ��� ���������� ������� (� �������� �������������� "��������� ���").
	*/

	//string folder1 = "D:\\Projects\\OpenCV\\101_ObjectCategories\\dollar_bill";
	//string folder2  = "D:\\Projects\\OpenCV\\101_ObjectCategories\\dollar_bill";
	string folder1 = "D:\\Projects\\OpenCV\\Coins\\coins-5";
	string folder2 = "D:\\Projects\\OpenCV\\Coins\\coins-10";

	//������� ������, ��������������� ��� �������� ������, ��������� ����� JPEG ������ �� folder1 � folder2.
	std::vector<std::string> filesList;
	//��������� ��������� ������ ������� ������ �� ����� ���������, ��������� ����� �����������, ����������� � ������ � ������ ���������, � ��������� ����� �����������.
	int count1 = GetFilesInFolder(folder1, filesList);
	int count2 = GetFilesInFolder(folder2, filesList);
	int totalCount = count1 + count2;

	double **mistakes = new double*[5];
	for (int i = 0; i < 5; i++)
	{
		mistakes[i] = new double[3];
	}
	int vocSizes[] = { 50, 60, 70, 80, 90 };
	double trainProportions[] = { 0.5, 0.6, 0.7 };

	string detectorType = "SIFT";
	string descriptorType = "SIFT";
	//int vocSize = 50;
	//double trainProportion = 0.5;

	for (int i = 0; i < 5; i ++)
	{
		for (int j = 0; j < 3; j ++)
		{
			int vocSize = vocSizes[i];
			double trainProportion = trainProportions[j];

			ParseInputFile("input.txt", detectorType, descriptorType);

			//���������������� ������ nonfree, �������������� ������ � SIFT � SURF ����������� � �������������.
			initModule_nonfree();

			//������� ������ ������, �������������� �������� ����� (���� detectorType).
			Ptr<FeatureDetector> featureDetector = FeatureDetector::create(detectorType);

			//������� ������ ������ ����, ������������ ����������� �������� ����� (���� descriptorType).
			Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(descriptorType);

			//������� ������ ������, ���������������� ��� ���������� ���������� � ����������� "�����" 
			//�� ������� ������������ �������� ����� (���� "BruteForce").
			Ptr<DescriptorMatcher> descriptorsMatcher = DescriptorMatcher::create("BruteForce");

			//������� ������ ������, ���������������� ��� ���������� ������������ �������� �����������.
			Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(descExtractor, descriptorsMatcher);

			//������� ������ ��� �������� �����, ����������� ��������� ����������� �� ������������� � �������� �������. 
			//���������������� ��� ���������� ���������� ����� �������, ����� ���� ��������, 
			//����������� � ������������� ������� ���� ����� trainProportion.
			std::vector<bool> mask(totalCount, false);
			//InitRandomBoolVector(mask, trainProportion);
			InitBoolVector(mask, trainProportion, count1, count2);

			//������� �������, ���������� ��������� �����������: ����� ����� ����� ���������� ����� �����������, ����� �������� ����� 1, 
			//���������� ������� �������� 32-������ ����� ����� �� ������. 
			//��������� ������� ��������� �������: ��������, ����������� � ������ ���������, ������������ �������� 1, 
			//��������, ����������� �� ������ ��������� � -1.
			Mat categories(totalCount, 1, CV_32S);
			for (int k = 0; k < totalCount; k++)
			{
				if (k < count1)
				{
					categories.at<int>(k, 0) = 1;
				}
				else
				{
					categories.at<int>(k, 0) = -1;
				}
			}

			//������� ������� ������������ �������� ����� �� ������������, ����������� � ������������� �������.
			Mat voc = TrainVocabulary(filesList, mask, featureDetector, descExtractor, vocSize);

			//���������� �������.
			//cv::Mat uDictionary;
			//voc.convertTo(uDictionary, CV_8U);
			//bowExtractor->setVocabulary(uDictionary);
			bowExtractor->setVocabulary(voc);

			//������������ ������������� ������� ��� �������������� "��������� ���".
			Mat trainData;
			Mat trainResponses;
			ExtractTrainData(filesList, mask, categories, featureDetector, bowExtractor, trainData, trainResponses);

			//������� ������������� "��������� ���" �� �������������� �������.
			Ptr<CvRTrees> classifier = TrainClassifier<CvRTrees, CvRTParams>(trainData, trainResponses);

			//����������� ��������� �����������, ����������� � �������� �������.
			Mat predictions = PredictOnTestData<CvRTrees>(filesList, mask, featureDetector, bowExtractor, classifier);

			//������������ �������, ���������� ���������� ��������� ����������� �� �������� �������.
			Mat rightAnswers = GetTestResponses(categories, mask);

			//��������� � ������� ������ ������������� �� �������� �������.
			float mistake = CalculateMisclassificationError(rightAnswers, predictions);

			mistakes[i][j] = mistake;
			//cout << mistake << endl;
		}
	}

	WriteOutputFile(mistakes, 5, 3, detectorType, descriptorType);
}