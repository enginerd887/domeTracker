// Pylon_with_OpenCV.cpp

/*
    Note: Before getting started, Basler recommends reading the Programmer's Guide topic
    in the pylon C++ API documentation that gets installed with pylon.
    If you are upgrading to a higher major version of pylon, Basler also
    strongly recommends reading the Migration topic in the pylon C++ API documentation.
    This sample illustrates how to grab and process images using the CInstantCamera class and OpenCV.
    The images are grabbed and processed asynchronously, i.e.,
    while the application is processing a buffer, the acquisition of the next buffer is done
    in parallel.

	OpenCV is used to demonstrate an image display, an image saving and a video recording.
    The CInstantCamera class uses a pool of buffers to retrieve image data
    from the camera device. Once a buffer is filled and ready,
    the buffer can be retrieved from the camera object for processing. The buffer
    and additional image data are collected in a grab result. The grab result is
    held by a smart pointer after retrieval. The buffer is automatically reused
    when explicitly released or when the smart pointer object is destroyed.
*/

//#include "stdafx.h"

// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iterator>


// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using OpenCV objects.
using namespace cv;

// Namespace for using cout.
using namespace std;

////////////////////////// Program Setup //////////////////////////////////////

// Function Declarations
void on_low_r_thresh_trackbar(int, void *);
void on_high_r_thresh_trackbar(int, void *);
void on_low_g_thresh_trackbar(int, void *);
void on_high_g_thresh_trackbar(int, void *);
void on_low_b_thresh_trackbar(int, void *);
void on_high_b_thresh_trackbar(int, void *);
string type2str(int type);


// Globals (May need to scan through this and delete stuff)
int low_r=30, low_g=30, low_b=30;
int high_r=100, high_g=100, high_b=100;

const int minTmax = 50;
const int maxTmax = 1000;
const int areaMax = 2000;
const int convexMax = 100;
const int circleMax = 100;
const int inertiaMax = 100;

int minTslider = 20;
int maxTslider = 1000;
int areaSlider = 100;
int convexSlider = 1;
int circleSlider = 3;
int inertiaSlider = 1;

int Imultiply = 60;
int threshVal = 100;
double multiplier = 60;
double threshValue = 40;
int counter2 = 0;
int oldSize = 0;
int mouseClicked = 0;
RNG rng(12345);


/////////////////////// Blob Detector Setup ///////////////////////////////////

void display_vector(const vector<int> &v) //note the const
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<int>(std::cout, " "));
}
/////////////////////////// Main Program ///////////////////////////////////////
int main(int argc, char* argv[])
{

    std::vector<KeyPoint> keypointList;

    // Parameters for blob detector
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 100;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 1000;
    params.maxArea = 2000;

    // Filter by Circularity
    params.filterByCircularity = false;
    params.minCircularity = 0.2;

    // Filter by Convexity
    params.filterByConvexity = false;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // The exit code of the sample application.
    int exitCode = 0;

    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    try
    {
        // Create an instant camera object with the camera device found first.
        CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice());

		// Get a camera nodemap in order to access camera parameters.
		GenApi::INodeMap& nodemap= camera.GetNodeMap();

		// Open the camera before accessing any parameters.
		camera.Open();
		// Create pointers to access the camera Width and Height parameters.
		GenApi::CIntegerPtr width= nodemap.GetNode("Width");
		GenApi::CIntegerPtr height= nodemap.GetNode("Height");

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        //camera.MaxNumBuffer = 5;

		// Create a pylon ImageFormatConverter object.
		CImageFormatConverter formatConverter;
		// Specify the output pixel format.
		formatConverter.OutputPixelFormat= PixelType_BGR8packed;
		// Create a PylonImage that will be used to create OpenCV images later.
		CPylonImage pylonImage;

		// Create an OpenCV image.
		Mat openCvImage;

        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
		camera.StartGrabbing(GrabStrategy_LatestImageOnly);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

		// tracking parameter
		int frame = 1;
		vector<Point2f> corners, nextcorners;
		Mat prevgrayimage;
    Mat im;
    Mat imAverage;
    Mat imCurrent;
    Mat filtered;
    Mat filtered2;
    Mat ref;
    Mat imInv;
    Mat edges;
    Mat findYellow;

    int AvgCounter = 0;
        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {

            double min,max;
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
                //cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                //cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;

				// Convert the grabbed buffer to a pylon image.
				formatConverter.Convert(pylonImage, ptrGrabResult);

				// Create an OpenCV image from a pylon image.
				openCvImage= cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *) pylonImage.GetBuffer());

				// Create an OpenCV display window.
				namedWindow( "OpenCV Display Window", CV_WINDOW_NORMAL); // other options: CV_AUTOSIZE, CV_FREERATIO
        namedWindow( "Processed Value", CV_WINDOW_NORMAL);
        //Create TrackBars for keypoints parameters


        // Will be used later to hold starting centroid of blobs
        Point2f b(0.0,0.0);
        Point startPoint = b;
        int counter = 0;



        im = openCvImage;
        imAverage = openCvImage;
        ref = Mat::zeros(im.size(),CV_32FC3);


        int countCheck = 0;
        Point2f firstCenter;

        Scalar color = Scalar(255,255,255);
        Scalar color2 = Scalar(0,255,0);
        Scalar color3 = Scalar(0,0,255);

        int refCounter = 0;

				// Display the current image in the OpenCV display window.
				imshow( "OpenCV Display Window", openCvImage);
				// Define a timeout for customer's input in ms.
				// '0' means indefinite, i.e. the next image will be displayed after closing the window.
				// '1' means live stream

				// optical flow tracking algorithm

				if (frame == 1)
				{
          //Stuff yu want to happen on the first frame only
					//feature extraction
					Mat image, grayimage;
					im = openCvImage;
          imAverage = openCvImage;
					cvtColor(im, grayimage, CV_BGR2GRAY); //transform to grayimage

					//corner detection
					int maxCorners = 10;
					double qualityLevel = 0.01;
					double minDistance = 10;
					int blockSize = 3;
					bool useHarrisDetector = false;
					double k = 0.04;
					goodFeaturesToTrack(grayimage, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

					////refine corner detection
					//cornerSubPix(grayimage, corners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.1));

					//update
					prevgrayimage = grayimage;
					//plot feature point
					for (size_t i = 0; i < corners.size(); i++)
					{
						circle(image, corners[i], 5, cv::Scalar(255.), -1);
					}


					//if (waitKey(1) == 'q') break;
				}

				else
          // Stuff you want to happen every frame
				{

					Mat nextimage, nextgrayimage;
          Mat mask;
					nextimage = openCvImage;
					cvtColor(nextimage, nextgrayimage, CV_BGR2GRAY); //transform to grayimage

          ////////////// Prepare Image for Computer Vision /////////////////////
          AvgCounter++;
          add(imAverage,nextimage,imAverage);
          divide(imAverage,2,imAverage);

          //////////////////////CV for red fiducials //////////////////////////
          cvtColor(imAverage,mask,CV_BGR2HSV);

          // Threshold the HSV image, keep only the red pixels
         	Mat lower_red_hue_range;
         	Mat upper_red_hue_range;

          //Finding Red
         	inRange(mask, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
         	inRange(mask, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
          addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, mask);

          GaussianBlur(mask,mask,Size(7,7),0,0);
          medianBlur(mask,mask,5);

          // Create a structuring element (SE)

          int dilation_size = 4;

          Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
          /// Apply the dilation operation
          dilate( mask, mask, element );
          //bitwise_not(mask,mask);
          // Find the contours
          vector<vector<Point> > rContours;
          vector<Vec4i> hierarchy;

          findContours(mask,rContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
          int morph_size = 3;

          element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

          morphologyEx(mask,mask,MORPH_CLOSE,element,Point(-1,-1),1);
          /// Draw contours
          Mat drawingR = Mat::zeros( mask.size(), CV_8UC3 );

          double areaR[rContours.size()];
          double perimR[rContours.size()];
          double circularity;

          vector<Moments> muR(rContours.size() );

          for( int i = 0; i< rContours.size(); i++ )
          {

            approxPolyDP(rContours[i],rContours[i],1,true);
            muR[i] = moments(rContours[i],false);
            areaR[i] = contourArea(rContours[i],true);
            perimR[i] = arcLength(rContours[i],true);
          }
          vector<Point2f> ContCenterR(rContours.size());

          for( int i = 0; i< rContours.size(); i++ )
          {

            circularity = -4*M_PI*(areaR[i]/(perimR[i]*perimR[i]));

            if (circularity > .7 && circularity < 1.2 && -areaR[i] > 2000)
            {
              Scalar color = Scalar( 255, 0,255 );
              Scalar color2 = Scalar(255,0,10);

              ContCenterR[i] = Point2f(muR[i].m10/muR[i].m00, muR[i].m01/muR[i].m00);

              drawContours( drawingR, rContours, i, color, 2, 8, hierarchy, 0, Point() );
              circle( drawingR, ContCenterR[i], 15, color2, -1, 8, 0 );

            }


          }

          dilation_size = 4;

          element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
          /// Apply the dilation operation
          dilate( drawingR, drawingR, element );

          //////////////////// CV for central portion //////////////////////////
          bitwise_not(imAverage,imInv);

          cvtColor(imInv,imInv,CV_BGR2GRAY);
          GaussianBlur(imInv,filtered,Size(7,7),0,0);
          threshold(filtered,filtered,150,255,0);
          Canny(filtered,filtered,0,255,3,true);
          //GaussianBlur(filtered,filtered,Size(7,7),0,0);

          vector<vector<Point> > contours;
          vector<vector<Point> > outerContours;



          // Create a structuring element (SE)
          morph_size = 3;
          element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

          morphologyEx(filtered,filtered,MORPH_CLOSE,element,Point(-1,-1),1);

          dilation_size = 4;

          element = getStructuringElement( MORPH_RECT,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
          /// Apply the dilation operation
          dilate( filtered, filtered, element );

          floodFill(filtered,cv::Point(0,0),Scalar(255));


          findContours(filtered,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

          /// Draw contours
          Mat drawing = Mat::zeros( filtered.size(), CV_8UC3 );
          Mat drawing2 = Mat::zeros( filtered.size(), CV_8UC3 );

          double area[contours.size()];
          double perims[contours.size()];
          double tol = .7;

          vector<Moments> mu(contours.size() );

          for( int i = 0; i< contours.size(); i++ )
          {

            approxPolyDP(contours[i],contours[i],1,true);
            mu[i] = moments(contours[i],false);
            area[i] = contourArea(contours[i],true);
            perims[i] = arcLength(contours[i],true);
          }

          Point2f ContCenter;

          for( int i = 0; i< contours.size(); i++ )
          {

            circularity = 4*M_PI*(area[i]/(perims[i]*perims[i]));

            if (circularity > .6 && circularity < 1.2 && area[i] > 100000)
            {
              Scalar color = Scalar( 255, 255,255 );
              Scalar color2 = Scalar(0,0,0);

              ContCenter = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);

              drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );


            }
          }

          bitwise_not(drawing,drawing);
          floodFill(drawing,cv::Point(10,10),Scalar(0));


          openCvImage.copyTo(drawing2,drawing);


          if (counter2 < 1)
          {
            ref = drawing2.clone();
            counter2++;

          }
          //bitwise_not(drawing,drawing);


          //////////////// Detect and Display contacts ////////////////////////////////

          //Start from the masked image provided by "drawing2"
          //cvtColor(drawing2,drawing2,CV_BGR2GRAY);
          //threshold(drawing2,drawing2,230,255,0);
          circle( drawing2, ContCenter, 15, color2, -1, 8, 0 );
          //detector->detect( filtered2, keypointList2);
          // Draw detected blobs as red circles.
          // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
          // the size of the circle corresponds to the size of blob

          //Mat im_with_keypoints;
          //Mat im_with_refpoints;
          //drawKeypoints( imCurrent, keypointList, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          //drawKeypoints( im, keypointList2,im_with_refpoints, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
          //addWeighted(im_with_keypoints,.5,im_with_refpoints,.5,0.0,im_with_keypoints);
          // Show blobs
          //imshow("keypoints", im_with_keypoints );
          //addWeighted(drawing2,.5,findYellow,.5,0.0,drawing2);


          //imshow("threshold2",drawing);
					//opticalflow
					vector<uchar> status;
					vector<float> err;
					calcOpticalFlowPyrLK(prevgrayimage, nextgrayimage, corners, nextcorners, status, err);

					//update
					prevgrayimage = nextgrayimage;
					corners = nextcorners;

					//plot feature point
					for (size_t i = 0; i < nextcorners.size(); i++)
					{
						circle(nextimage, nextcorners[i], 5, cv::Scalar(255.), -1);
					}
          add(drawing2,drawingR,drawing2);
              // Show our image inside it.
          imshow("Processed Value",drawing2);
					waitKey(1);
					// if (waitKey(1) == 'q') break;

				}
				// waitKey(1);
				frame++;


#ifdef PYLON_WIN_BUILD
#endif
            }
            else
            {
                cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }
        }

    }
    catch (GenICam::GenericException &e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
        << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Comment the following two lines to disable waiting on exit.
    cerr << endl << "Press Enter to exit." << endl;
    while( cin.get() != '\n');

    return exitCode;
}
