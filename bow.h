#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <vector> 
#include <string> 


cv::Mat TrainVocabulary(const std::vector<std::string>&	filesList, 
	const std::vector<bool>& is_voc, 
	const cv::Ptr<cv::FeatureDetector>& keypointsDetector, 
	const cv::Ptr<cv::DescriptorExtractor>& descriptorsExtractor, 
	int	vocSize);

cv::Mat ExtractFeaturesFromImage(
	cv::Ptr<cv::FeatureDetector> keypointsDetector,
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor, const
	std::string& fileName);

void ExtractTrainData(const std::vector<std::string>& filesList,
	const std::vector<bool>& isTrain,
	const cv::Mat& responses,
	const cv::Ptr<cv::FeatureDetector>&	keypointsDetector,
	const cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor,
	cv::Mat& trainData,
	cv::Mat& trainResponses);

cv::Ptr<CvRTrees> TrainRFClassifier(const cv::Mat& trainData,
	const cv::Mat& trainResponses);

cv::Ptr<CvDTree> TrainDTClassifier(const cv::Mat& trainData,
	const cv::Mat& trainResponses);


template <typename ClassifierType, typename ParamsType>
cv::Ptr<ClassifierType> TrainClassifier(const cv::Mat& trainData,
										const cv::Mat& trainResponses,
										ParamsType params)
{
	cv::Mat varTypes(1, trainData.cols + 1, CV_8U, Scalar(CV_VAR_ORDERED));
	varTypes.at<uchar>(trainData.cols) = CV_VAR_CATEGORICAL;

	cv::Ptr<ClassifierType> classifierPtr = new ClassifierType();
	classifierPtr->train(trainData, CV_ROW_SAMPLE,
					     trainResponses, Mat(),
					     Mat(), varTypes,
					     Mat(), params);

	return classifierPtr;
}

template <typename T>
int Predict(const cv::Ptr<cv::FeatureDetector> keypointsDetector,
	const cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor,
	const cv::Ptr<T> classifier, 
	const std::string& fileName)
{
	Mat featureDesc = ExtractFeaturesFromImage(keypointsDetector, bowExtractor, fileName);
	return (int)(classifier->predict(featureDesc));
}

template <typename T>
cv::Mat PredictOnTestData(const std::vector<std::string>& filesList,
	const std::vector<bool>& isTrain,
	const cv::Ptr<cv::FeatureDetector> keypointsDetector,
	const cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor,
	const cv::Ptr<T> classifier)
{
	Mat predictions(0, 1, CV_32S);
	for (int i = 0; i < filesList.size(); ++i)
	{
		if (!isTrain[i])
		{
			int cat = Predict<T>(keypointsDetector, bowExtractor, classifier, filesList[i]);
			predictions.push_back(cat);
		}
	}

	return predictions;
}

cv::Mat GetTestResponses(const cv::Mat& responses, 
	const std::vector<bool>& isTrain);

float CalculateMisclassificationError(cv::Mat& responses,
	cv::Mat& predictions);