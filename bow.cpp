#include "bow.h" 
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;
using namespace std;

Mat TrainVocabulary(const vector<string>& filesList,
					const vector<bool>& is_voc,
					const Ptr<FeatureDetector>& keypointsDetector,
					const Ptr<DescriptorExtractor>& descriptorsExtractor,
					int	vocSize)
{
	BOWKMeansTrainer bowTrainer(vocSize);

	for (int i = 0; i < filesList.size(); ++i)
	{
		if (is_voc[i])
		{
			Mat img = imread(filesList[i]);
			vector<KeyPoint> keypoints;
			keypointsDetector->detect(img, keypoints);
			Mat descriptors;
			descriptorsExtractor->compute(img, keypoints, descriptors);
			
			if (descriptors.type() != CV_32F)
			{
				Mat desc;
				descriptors.convertTo(desc, CV_32F);
				bowTrainer.add(desc);
			}
			else
			{
				bowTrainer.add(descriptors);
			}
		}
	}

	Mat voc = bowTrainer.cluster();
	return voc;
}

Mat ExtractFeaturesFromImage(Ptr<FeatureDetector> keypointsDetector,
							 Ptr<BOWImgDescriptorExtractor> bowExtractor,
							 const string& fileName)
{
	Mat img = imread(fileName);
	vector<KeyPoint> keypoints;
	keypointsDetector->detect(img, keypoints);
	Mat imgDesc;
	bowExtractor->compute(img, keypoints, imgDesc);
	return imgDesc;
}

void ExtractTrainData(const vector<string>& filesList,
					  const vector<bool>& isTrain,
					  const Mat& responses,
	                  const Ptr<FeatureDetector>& keypointsDetector,
					  const Ptr<BOWImgDescriptorExtractor>& bowExtractor,
					  Mat& trainData,
					  Mat& trainResponses)
{
	int trainCount = 0;
	for (bool var : isTrain)
	{
		if (var)
		{
			trainCount++;
		}
	}

	trainData = Mat(0, bowExtractor->descriptorSize(), CV_32F);
	trainResponses = Mat(0, 1, CV_32S);

	for (int i = 0; i < filesList.size(); ++i)
	{
		if (isTrain[i])
		{
			Mat featureDesc = ExtractFeaturesFromImage(keypointsDetector, bowExtractor, filesList[i]);
			trainData.push_back(featureDesc);
			trainResponses.push_back(responses.at<int>(i, 0));
		}
	}
}

Ptr<CvRTrees> TrainRFClassifier(const Mat& trainData,
							  const Mat& trainResponses)
{
	CvRTParams params;
	params.term_crit.type = CV_TERMCRIT_ITER;
	params.term_crit.max_iter = 200;
	//params.max_num_of_trees_in_the_forest = 200;

	Mat varTypes(1, trainData.cols + 1, CV_8U, Scalar(CV_VAR_ORDERED));
	varTypes.at<uchar>(trainData.cols) = CV_VAR_CATEGORICAL;

	Ptr<CvRTrees> rf = new CvRTrees();
	rf->train(trainData, CV_ROW_SAMPLE,
		trainResponses, Mat(),
		Mat(), varTypes,
		Mat(), params);
	rf->save("model-rf.yml", "simpleRTreesModel");

	return rf;
}


Ptr<CvDTree> TrainDTClassifier(const Mat& trainData,
	const Mat& trainResponses)
{
	CvDTreeParams params;
	params.max_depth = 10;
	params.min_sample_count = 1;
	params.cv_folds = 5;

	Mat varTypes(1, trainData.cols + 1, CV_8U, Scalar(CV_VAR_ORDERED));
	varTypes.at<uchar>(trainData.cols) = CV_VAR_CATEGORICAL;

	Ptr<CvDTree> dt = new CvDTree();
	dt->train(trainData, CV_ROW_SAMPLE,
		trainResponses, Mat(),
		Mat(), varTypes,
		Mat(), params);
	dt->save("model-dt.yml", "simpleDTreeModel");

	return dt;
}


//template <typename T>
//int Predict(const Ptr<FeatureDetector> keypointsDetector,
//			const Ptr<BOWImgDescriptorExtractor> bowExtractor,
//			const Ptr<T> classifier,
//			const string& fileName)
//{
//	Mat featureDesc = ExtractFeaturesFromImage(keypointsDetector, bowExtractor, fileName);
//	return (int)(classifier->predict(featureDesc));
//}

//template <typename T>
//Mat PredictOnTestData(const vector<string>& filesList,
//					  const vector<bool>& isTrain,
//				      const Ptr<FeatureDetector> keypointsDetector,
//					  const Ptr<BOWImgDescriptorExtractor> bowExtractor,
//					  const Ptr<T> classifier)
//{
//	Mat predictions(0, 1, CV_32S);
//	for (int i = 0; i < filesList.size(); ++i)
//	{
//		if (!isTrain[i])
//		{
//			int cat = Predict<T>(keypointsDetector, bowExtractor, classifier, filesList[i]);
//			predictions.push_back(cat);
//		}
//	}
//
//	return predictions;
//}

Mat GetTestResponses(const Mat& responses,
					 const vector<bool>& isTrain)
{
	int testCount = 0;
	for (bool var : isTrain)
	{
		if (!var)
		{
			testCount++;
		}
	}

	Mat testResponses(0, 1, CV_32S);
	for (int i = 0; i < responses.rows; ++i)
	{
		if (!isTrain[i])
		{
			testResponses.push_back(responses.at<int>(i));
		}
	}

	return testResponses;
}

float CalculateMisclassificationError(Mat& responses,
									  Mat& predictions)
{
	int mistakeCount = 0;
	for (int i = 0; i < responses.rows; i++)
	{
		if (responses.at<int>(i, 0) != predictions.at<int>(i, 0))
		{
			mistakeCount++;
		}
	}

	return ((float)mistakeCount / (float)responses.rows);
}