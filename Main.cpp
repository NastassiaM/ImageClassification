#include "auxiliary.h" 
#include "bow.h" 
#include <opencv2/nonfree/nonfree.hpp> 
#include <iostream> 

using namespace cv;
using namespace std;


// detectors types: "FAST", "STAR", "SIFT", "SURF", "ORB", "MSER", "GFTT" (GoodFeaturesToTrack), "HARRIS", "Dense", "SimpleBlob"
// descriptors types: "SIFT", "SURF", "BRIEF", "ORB"

enum
{
	RANDOM_FOREST = 1,
	DECISION_TREE = 2
};

int main(int argc, char* argv[])
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
	int classifierType = 0;

	ParseInputFile(argv[1], detectorType, descriptorType, classifierType);

	for (int i = 0; i < 5; i ++)
	{
		for (int j = 0; j < 3; j ++)
		{
			int vocSize = vocSizes[i];
			double trainProportion = trainProportions[j];

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
			if (descriptorType == "BRIEF" || descriptorType == "ORB")
			{
				cv::Mat uDictionary;
				voc.convertTo(uDictionary, CV_8U);
				bowExtractor->setVocabulary(uDictionary);
			}
			else
			{
				bowExtractor->setVocabulary(voc);
			}
			

			//������������ ������������� ������� ��� �������������� "��������� ���".
			Mat trainData;
			Mat trainResponses;
			ExtractTrainData(filesList, mask, categories, featureDetector, bowExtractor, trainData, trainResponses);

			Mat predictions;
			switch (classifierType)
			{
				case RANDOM_FOREST:
				{
					//������� ������������� "��������� ���" �� �������������� �������.
					Ptr<CvRTrees> classifier = TrainRFClassifier(trainData, trainResponses);

					//����������� ��������� �����������, ����������� � �������� �������.
					predictions = PredictOnTestData<CvRTrees>(filesList, mask, featureDetector, bowExtractor, classifier);
					break;
				}
				case DECISION_TREE:
				{
					//������� ������������� "������ �������" �� �������������� �������.
					Ptr<CvDTree> classifier = TrainDTClassifier(trainData, trainResponses);

					//����������� ��������� �����������, ����������� � �������� �������.
					predictions = PredictOnTestData<CvDTree>(filesList, mask, featureDetector, bowExtractor, classifier);
					break;
				}
				default:
				{
					cout << "Incorrect classifier type. 1 or 2 required." << endl;
					return -1;
				}
			}

			//������������ �������, ���������� ���������� ��������� ����������� �� �������� �������.
			Mat rightAnswers = GetTestResponses(categories, mask);

			//��������� � ������� ������ ������������� �� �������� �������.
			float mistake = CalculateMisclassificationError(rightAnswers, predictions);

			mistakes[i][j] = mistake;
			//cout << mistake << endl;
		}
	}

	WriteOutputFile(mistakes, 5, 3, detectorType, descriptorType, classifierType);

	return 0;
}