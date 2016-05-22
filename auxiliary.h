#include <string> 
#include <vector> 


/*
Заполняет массив filesList списком всех файлов с расширением *.jpg из директории dirPath.
dirPath – путь к директории, содержащей изображения
filesList – список всех файлов с расширением *.jpg, содержащихся в данной директории
*/
int GetFilesInFolder(const std::string& dirPath, std::vector<std::string> &filesList);

/*
Заполняет булевский вектор mask случайными значениями (true с вероятностью prob).
mask – булевский вектор, который должен быть заполнен случайными значениями;
prob – вероятность того, что элементу булевского массива будет присвоено значение true.
*/
void InitRandomBoolVector(std::vector<bool>& mask, double prob);


void InitBoolVector(std::vector<bool>& mask, double prob, int count1, int count2);


void ParseInputFile(const std::string& fileName, 
					std::string& detectorType, 
					std::string& descriptorType,
					int& classifierType);

void WriteOutputFile(double **mistakes,
					 int n,
					 int m,
					 std::string& detectorType,
					 std::string& descriptorType,
					 int classifierType);