#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descSource.type() != CV_32F) { 
            // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout<<"FLANN matching"<<endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) { 
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0) { 
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &descriptorTime) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0){
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    else {
        throw invalid_argument(descriptorType + "is not a valid type!!");
    }

    // perform feature description
    descriptorTime = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    descriptorTime = ((double)cv::getTickCount() - descriptorTime) / cv::getTickFrequency();
    descriptorTime = 1000 * descriptorTime / 1.0;
    cout << descriptorType << " descriptor extraction in " << descriptorTime << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &detectorTime, int &detectorKeypoints, bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    detectorTime = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
    detectorTime = 1000 * detectorTime / 1.0;
    detectorKeypoints = keypoints.size();
    cout << "Shi-Tomasi detection with n=" << detectorKeypoints << " keypoints in " << detectorTime << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using Harris Corner Detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &detectorTime, int &detectorKeypoints, bool bVis) {

    // detector parameters
    int blockSize = 2; // for every pixel, a blockSize x blockSize neighborhood is considered
    int apertureSize = 3; // aperture parametes for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in thr 8bit scaled response matrix
    double k = 0.04; // Harris parameter

    // detect Harris corners and normalize output
    detectorTime = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // maximum permissible overlap between two features in %, used during non-max suppression
    for (size_t j = 0; j <dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse){
                // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppresion(NMS) in local neighborhood around new keypoint
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);

                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            // if overlap is > t and response is higher for a new kpt
                            // replace old keypoint with new one
                            *it = newKeyPoint;
                            break; // quit loop over keypoint
                        }
                    }
                }

                if (!bOverlap) {
                    // only add new keypoint if no overlap has been found
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        }
    }

    detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
    detectorTime = 1000 * detectorTime / 1.0;
    detectorKeypoints = keypoints.size();
    cout << "Harris detection with n=" << detectorKeypoints << " keypoints in " << detectorTime << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the modern techniques - FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, double &detectorTime, int &detectorKeypoints, bool bVis) {

    string windowName;
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0) {
        int threshold = 30; // difference between intensity of the central pixel and pixel of a circle around this pixel
        bool bNMS = true; // perform non-max suppresion on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);

        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime / 1.0;
        detectorKeypoints = keypoints.size();

        cout<<"FAST Detector with n= " <<detectorKeypoints<<" keypoints in "<<detectorTime<<" ms." <<endl;
        windowName = "FAST Results";
    }

    else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);

        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime / 1.0;
        detectorKeypoints = keypoints.size();
        cout<<"BRISK Detector with n= " <<detectorKeypoints<<" keypoints in "<<detectorTime<<" ms."<<endl;
        windowName = "BRISK Results";
    }

    else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);

        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime / 1.0;
        detectorKeypoints = keypoints.size();
        cout<<"ORB Detector with n= " <<detectorKeypoints<<" keypoints in "<<detectorTime<<" ms."<<endl;
        windowName = "ORB Results";
    }

    else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);

        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime / 1.0;
        detectorKeypoints = keypoints.size();
        cout<<"AKAZE Detector with n= " <<detectorKeypoints<<" keypoints in "<<detectorTime<<" ms."<<endl;
        windowName = "AKAZE Results";
    }

    else if (detectorType.compare("SIFT") == 0) {
        detector = cv::xfeatures2d::SIFT::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);

        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime / 1.0;
        detectorKeypoints = keypoints.size();
        cout<<"SIFT Detector with n= " << detectorKeypoints <<" keypoints in "<<detectorTime<<" ms."<<endl;
        windowName = "SIFT Results";
    }

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}