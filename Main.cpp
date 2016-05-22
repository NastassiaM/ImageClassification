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
	Считать аргументы, переданные программе через командную строку:

	string folder1 – путь к папке, содержащей объекты первой категории;
	string folder2 – путь к папке, содержащей объекты второй категории;
	string detectorType – тип детектора ключевых точек;
	string descriptorType – тип дескрипторов ключевых точек;
	int vocSize – размер словаря;
	double trainProportion – доля объектов, используемых для построения словаря (и обучения классификатора "случайный лес").
	*/

	//string folder1 = "D:\\Projects\\OpenCV\\101_ObjectCategories\\dollar_bill";
	//string folder2  = "D:\\Projects\\OpenCV\\101_ObjectCategories\\dollar_bill";
	string folder1 = "D:\\Projects\\OpenCV\\Coins\\coins-5";
	string folder2 = "D:\\Projects\\OpenCV\\Coins\\coins-10";

	//Создать массив, предназначенный для хранения списка, хранящего имена JPEG файлов из folder1 и folder2.
	std::vector<std::string> filesList;
	//Заполнить созданный список именами файлов из обеих категорий, вычислить число изображений, относящихся к первой и второй категории, и суммарное число изображений.
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

			//Инициализировать модуль nonfree, обеспечивающий работу с SIFT и SURF детекторами и дескрипторами.
			initModule_nonfree();

			//Создать объект класса, детектирующего ключевые точки (типа detectorType).
			Ptr<FeatureDetector> featureDetector = FeatureDetector::create(detectorType);

			//Создать объект класса типа, вычисляющего дескрипторы ключевых точек (типа descriptorType).
			Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create(descriptorType);

			//Создать объект класса, предназначенного для нахождения ближайшего к дескриптору "слова" 
			//из словаря дескрипторов ключевых точек (типа "BruteForce").
			Ptr<DescriptorMatcher> descriptorsMatcher = DescriptorMatcher::create("BruteForce");

			//Создать объект класса, предназначенного для вычисления признакового описания изображений.
			Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(descExtractor, descriptorsMatcher);

			//Создать массив для хранения маски, описывающей разбиение изображений на тренировочную и тестовую выборки. 
			//Инициализировать его случайными значениями таким образом, чтобы доля объектов, 
			//относящихся к тренировочной выборке была равна trainProportion.
			std::vector<bool> mask(totalCount, false);
			//InitRandomBoolVector(mask, trainProportion);
			InitBoolVector(mask, trainProportion, count1, count2);

			//Создать матрицу, содержащую категории изображений: число строк равно суммарному числу изображений, число столбцов равно 1, 
			//элементами матрицы являются 32-битные целые числа со знаком. 
			//Заполнить матрицу следующим образом: объектам, относящимся к первой категории, соотвествует значение 1, 
			//объектам, относящимся ко второй категории – -1.
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

			//Обучить словарь дескрипторов ключевых точек на изображениях, относящихся к тренировочной выборке.
			Mat voc = TrainVocabulary(filesList, mask, featureDetector, descExtractor, vocSize);

			//Установить словарь.
			//cv::Mat uDictionary;
			//voc.convertTo(uDictionary, CV_8U);
			//bowExtractor->setVocabulary(uDictionary);
			bowExtractor->setVocabulary(voc);

			//Сформировать тренировочную выборку для классификатора "случайный лес".
			Mat trainData;
			Mat trainResponses;
			ExtractTrainData(filesList, mask, categories, featureDetector, bowExtractor, trainData, trainResponses);

			//Обучить классификатор "случайный лес" на сформированной выборке.
			Ptr<CvRTrees> classifier = TrainClassifier<CvRTrees, CvRTParams>(trainData, trainResponses);

			//Предсказать категории изображений, относящихся к тестовой выборке.
			Mat predictions = PredictOnTestData<CvRTrees>(filesList, mask, featureDetector, bowExtractor, classifier);

			//Сформировать матрицу, содержащую правильные категории изображений из тестовой выборки.
			Mat rightAnswers = GetTestResponses(categories, mask);

			//Вычислить и вывести ошибку классификации на тестовой выборке.
			float mistake = CalculateMisclassificationError(rightAnswers, predictions);

			mistakes[i][j] = mistake;
			//cout << mistake << endl;
		}
	}

	WriteOutputFile(mistakes, 5, 3, detectorType, descriptorType);
}