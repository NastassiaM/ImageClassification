#include <string> 
#include <vector> 


/*
��������� ������ filesList ������� ���� ������ � ����������� *.jpg �� ���������� dirPath.
dirPath � ���� � ����������, ���������� �����������
filesList � ������ ���� ������ � ����������� *.jpg, ������������ � ������ ����������
*/
int GetFilesInFolder(const std::string& dirPath, std::vector<std::string> &filesList);

/*
��������� ��������� ������ mask ���������� ���������� (true � ������������ prob).
mask � ��������� ������, ������� ������ ���� �������� ���������� ����������;
prob � ����������� ����, ��� �������� ���������� ������� ����� ��������� �������� true.
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