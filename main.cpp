#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
     std::cout << "Started" << std::endl;
    // Load the image
    cv::Mat image = cv::imread("/Users/jd/Desktop/Labrador_Retriever_portrait.jpg");
    if (image.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }else{
             std::cout << "Started ok" << std::endl;

    }

    // Convert the image from BGR to Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Convert to floating point for k-means
    labImage.convertTo(labImage, CV_32F);

    // Reshape and convert to a batch of 3D color vectors
    cv::Mat data = labImage.reshape(1, labImage.total());
    
    // Number of clusters
    int K = 8;
    cv::Mat labels;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
    cv::Mat centers;

    // Apply k-means
    cv::kmeans(data, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);

    // Map each pixel to its center
    cv::Mat newImage(labImage.size(), labImage.type());
    for (int i = 0; i < labImage.rows; i++) {
        for (int j = 0; j < labImage.cols; j++) {
            int clusterIdx = labels.at<int>(i * labImage.cols + j);
            newImage.at<cv::Vec3f>(i, j)[0] = centers.at<float>(clusterIdx, 0);
            newImage.at<cv::Vec3f>(i, j)[1] = centers.at<float>(clusterIdx, 1);
            newImage.at<cv::Vec3f>(i, j)[2] = centers.at<float>(clusterIdx, 2);
        }
    }

    // Convert the clustered Lab image back to BGR color space
    cv::Mat finalImage;
    newImage.convertTo(newImage, CV_8UC3);
    cv::cvtColor(newImage, finalImage, cv::COLOR_Lab2BGR);

    // Display the original and the processed image
    cv::imshow("Original Image", image);
    cv::imshow("Processed Image", finalImage);
    cv::waitKey(0);

    return 0;
}