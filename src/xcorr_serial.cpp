#include <opencv2/opencv.hpp>
using namespace cv;

Mat crossCorrelationSerial(const Mat &image, const Mat &templ)
{
    int outRows = image.rows - templ.rows + 1;
    int outCols = image.cols - templ.cols + 1;
    Mat result(outRows, outCols, CV_32F, Scalar(0));

    for (int y = 0; y < outRows; y++)
    {
        for (int x = 0; x < outCols; x++)
        {
            float sum = 0.0f;
            for (int j = 0; j < templ.rows; j++)
            {
                for (int i = 0; i < templ.cols; i++)
                {
                    sum += image.at<uchar>(y + j, x + i) * templ.at<uchar>(j, i);
                }
            }
            result.at<float>(y, x) = sum;
        }
    }
    return result;
}
