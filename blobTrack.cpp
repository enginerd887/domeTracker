// Pylon_with_OpenCV.cpp

/*
Tito Fernandez
Lynch Lab, Northwestern University
Fingertip Force Sensor Project

This code accomplishes the following tasks:

* Reads a Pylon-based camera (Basler Dart daA1600-60uc in this case)
* Converts Pylon-based images to OpenCV Mats
* Reads the input image and tracks any circular red regions found (fiducials for position tracking)
* Determines what portion of the image is the dome region, and masks out everything else
* Keeps track of the centroid of each red fiducial and of the main dome regions

This script is based on the Pylon_with_OpenCV example available online.
*/


// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iterator>


// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#    include <pylon/PylonGUI.h>
#endif

// Settings for using Basler USB cameras.
#include <pylon/usb/BaslerUsbInstantCamera.h>
typedef Pylon::CBaslerUsbInstantCamera Camera_t;
typedef Pylon::CBaslerUsbCameraEventHandler CameraEventHandler_t; // Or use Camera_t::CameraEventHandler_t
typedef Pylon::CBaslerUsbImageEventHandler ImageEventHandler_t; // Or use Camera_t::ImageEventHandler_t
typedef Pylon::CBaslerUsbGrabResultPtr GrabResultPtr_t; // Or use Camera_t::GrabResultPtr_t
using namespace Basler_UsbCameraParams;

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using OpenCV objects.
using namespace cv;

// Namespace for using cout.
using namespace std;


//////////////////// Variables for Red tracking sliders ////////////////////////

// Slider maxima
const int redMaxHue = 255;
const int threshMax = 255;

// Slider default values
int minHue = 10;
int minHueHigh = 160;
int maxHueHigh = 180;
int redThresh = 110;

int rHue_slider = 10;
int rHueHighMin_slider = minHueHigh;
int rHueHighMax_slider = maxHueHigh;
int rThresh_slider = redThresh;

// Text holders for area next to sliders
char RedHue[50];
char rMinHigh[50];
char rMaxHigh[50];
char rThresh[50];


////////////////////////// All different frame holders /////////////////////////
Mat mask;
Mat filtered;
Mat imInv;
Mat drawingR;
Mat drawing;
Mat drawing2;
Mat undistorted;
Mat finalMat;
Mat drawContact;

// Variables and arrays for holding information from previous (old) frame
vector<Point> OldCentroids;
vector<int> oldIDs;
vector<int> lifeSpans;
vector<int> taken;

// Mats to hold the rotation and translation information to
cv::Mat rotation_vector; // Rotation in axis-angle form
cv::Mat translation_vector;
Mat rotationMatrix(3,3,cv::DataType<double>::type);

//Colors used in the program
Scalar color = Scalar( 255, 255,255 );
Scalar color2 = Scalar(100,0,255);
Scalar color3 = Scalar(0,255,0);

// Variables for Camera setting sliders

char ExpString[50];
char BlackString[50];
char ContactString[50];

const int maxExposure = 80000;
const int maxBlack = 20;
const int maxContact = 255;

double targetExposure = 20000.0;
double targetBlack = -10;
double targetContact = 220;

int exp_slider = 23000;
int black_slider = 2;
int contact_slider = 220;


Mat cameraMatrix = Mat::eye(3,3, CV_64F);
Mat distanceCoefficients;
///////////////////// Function Declarations ///////////////////////////////////

static void on_trackbarRed( int, void* );
static void on_trackbarExp( int, void* );
static void on_trackbarBlack( int, void* );
static void on_trackbarContact( int, void*);

// Function for loading camera intrinsic parameters from text file
bool loadCameraCalibration(Mat& cameraMatrix, Mat& distanceCoefficients)
{
  ifstream inStream;
  inStream.open("CameraParameters.txt");

  if(inStream)
  {
    uint16_t rows;
    uint16_t columns;

    inStream >> rows;
    inStream >> columns;

    cameraMatrix = Mat(Size(columns,rows), CV_64F);

    // Camera Matrix
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < columns; c++)
      {
        double read = 0.0f;
        inStream >> read;
        cameraMatrix.at<double>(r,c) = read;
      }
    }

    // Distance Coefficients
    inStream >> rows;
    inStream >> columns;

    distanceCoefficients = Mat::zeros(rows,columns, CV_64F);
    for (int r = 0; r < rows; r++)
    {
      for (int c = 0; c < columns; c++)
      {
        double read = 0.0f;
        inStream >> read;
        distanceCoefficients.at<double>(r,c) = read;
      }
    }
    inStream.close();
    return true;
  }
  return false;
}

// Function for closing detected contours to make them easier to track
void closeContours(int morph_size, Mat& frame)
{
  Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  morphologyEx(frame,frame,MORPH_CLOSE,element,Point(-1,-1),1);
}

// Function for dilating/blurring an image to facilitate detection
void dilateMat(int dilation_size,Mat& frame)
{
  Mat element = getStructuringElement( MORPH_RECT,
                               Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                               Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( frame,frame, element );
}
////////////////////////// Program Setup //////////////////////////////////////

void display_vector(const vector<int> &v) //note the const
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<int>(std::cout, " "));
}


/////////////////////////// Main Program ///////////////////////////////////////
int main(int argc, char* argv[])
{

    //////////Everything here is to get the camera up and running//////////////

    // The exit code of the sample application.
    int exitCode = 0;

    // Load the camera calibration parameters from a txt file
    loadCameraCalibration(cameraMatrix,distanceCoefficients);


    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    try
    {
        CDeviceInfo info;
        info.SetDeviceClass(Camera_t::DeviceClass());
        // Create an instant camera object with the camera device found first.
        Camera_t camera(CTlFactory::GetInstance().CreateFirstDevice(info));

		// Get a camera nodemap in order to access camera parameters.
		GenApi::INodeMap& nodemap= camera.GetNodeMap();

		// Open the camera before accessing any parameters.
		camera.Open();

    // Disable Auto functions here
    camera.ExposureAuto.SetValue(ExposureAuto_Off);
    camera.GainAuto.SetValue(GainAuto_Off);
    camera.BalanceWhiteAuto.SetValue(BalanceWhiteAuto_Off);

    // Set exposure and black level to default
    camera.ExposureTime.SetValue(targetExposure);
    camera.BlackLevel.SetValue(-targetBlack);
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

    int AvgCounter = 0;
        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {
            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Could access the image data.
                //cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                //cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;

				// Convert the grabbed buffer to a pylon image.
				formatConverter.Convert(pylonImage, ptrGrabResult);

				// Create an OpenCV image from a pylon image.
				openCvImage= cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *) pylonImage.GetBuffer());

				// Create OpenCV display windows for original image, red threshold, and final processed value.
				namedWindow( "OpenCV Display Window", CV_WINDOW_NORMAL); // other options: CV_AUTOSIZE, CV_FREERATIO
        namedWindow( "Red Detection", CV_WINDOW_NORMAL);
        namedWindow( "Processed Value", CV_WINDOW_NORMAL);

				// optical flow tracking algorithm

        // use the intrinsic camera parameters to correct image
        undistorted = openCvImage;
        //undistort(openCvImage,undistorted, cameraMatrix,distanceCoefficients);

        imshow( "OpenCV Display Window", undistorted);

        //////////////////////CV for red fiducials //////////////////////////


        // Create the Trackbars for the Red Values
        sprintf( RedHue, "Max of Low Range");
        sprintf( rMinHigh, "Min of High Range");
        sprintf( rMaxHigh, "Max of High Range");
        sprintf( rThresh, "Threshold");

        createTrackbar( RedHue, "Red Detection", &rHue_slider, redMaxHue, on_trackbarRed );
        createTrackbar( rMinHigh, "Red Detection", &rHueHighMin_slider, maxHueHigh, on_trackbarRed );
        createTrackbar( rMaxHigh, "Red Detection", &rHueHighMax_slider, redMaxHue, on_trackbarRed );
        createTrackbar( rThresh, "Red Detection", &rThresh_slider, threshMax, on_trackbarRed );
        on_trackbarRed(rHue_slider,0);

        // Create Trackbars for Exposure and Black Level
        sprintf( ExpString, "Exposure");
        sprintf( BlackString, "Black Level");

        createTrackbar(ExpString,"OpenCV Display Window", &exp_slider, maxExposure, on_trackbarExp);
        createTrackbar(BlackString,"OpenCV Display Window", &black_slider, maxBlack, on_trackbarBlack);

        sprintf( ContactString, "Contact Threshold");

        createTrackbar(ContactString, "Processed Value", &contact_slider, maxContact, on_trackbarContact);


        // Find the contours of the red markers
        vector<vector<Point> > rContours;
        vector<Vec4i> hierarchy;

        findContours(mask,rContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
        closeContours(5,mask);

        // Draw contours. Only marking the center points
        drawingR = Mat::zeros( mask.size(), CV_8UC3 );

        double areaR[rContours.size()];
        double perimR[rContours.size()];
        double circularity;
        vector<Moments> muR(rContours.size() );

        //Calculate parameters for each red blob recognized
        for( int i = 0; i< rContours.size(); i++ )
        {
          approxPolyDP(rContours[i],rContours[i],3,true);
          muR[i] = moments(rContours[i],false);
          areaR[i] = contourArea(rContours[i],true);
          perimR[i] = arcLength(rContours[i],true);
        }

        vector<Point2f> ContCenterR(rContours.size());
        vector<cv::Point> pointVals;

        int contCount = 0;
        int lineThickness = 2;
        int lineType = LINE_8;

        for( int i = 0; i< rContours.size(); i++ )
        {

          circularity = -4*M_PI*(areaR[i]/(perimR[i]*perimR[i]));

          // if detected blobs are sufficiently circular and large, keep them
          if ( circularity > .5 && circularity < 1.3 && -areaR[i] > 5000)
          {
            ContCenterR[i] = Point2f(muR[i].m10/muR[i].m00, muR[i].m01/muR[i].m00);
            pointVals.push_back(Point(ContCenterR[i]));

            //Draw centroids
            circle( drawingR, ContCenterR[i], 15, color2, -1, 8, .1);
            contCount++;
          }
        }

        dilateMat(4,drawingR);


        //////////////////// Assign IDs to found red points //////////////////////////
        vector<int> newIDs(contCount);
        vector<int> changeList(contCount,0);
        vector<int> taken(OldCentroids.size(),0);

        int blobCounter = 0;

        // If there is no history, just assign numbers
        if (OldCentroids.size()==0)
        {
          for (int ii = 0; ii < newIDs.size(); ii++)
          {
            newIDs[ii] = blobCounter;
            blobCounter++;
          }
        }

        // If there are fewer history points than current points,
        // Assign them, then add new IDs
        else if (pointVals.size() >= OldCentroids.size())
        {
          vector<int> changed(pointVals.size(),0);

          for (int ii = 0; ii < OldCentroids.size(); ii++)
          {
            double refDistance = 100000;
            int closest;

            for (int kk = 0; kk < pointVals.size(); kk++)
            {
              double distance = cv::norm(pointVals.at(kk)-OldCentroids.at(ii));
              if (distance < refDistance && changed[kk] == 0)
              {
                refDistance = distance;
                closest = kk;
              }
            }

            newIDs[closest] = oldIDs[ii];
            changed[closest] = 1;
          }

          for (int ii = 0; ii < newIDs.size(); ii++)
          {

            if (changed[ii]==0)
            {
              blobCounter = 0;
              bool exists = true;
              bool foundIt = false;

              while (exists)
              {
                for (int jj = 0; jj < oldIDs.size(); jj++)
                {
                  if (blobCounter == oldIDs[jj])
                  {
                    foundIt = true;
                  }
                }
                if (foundIt == true)
                {
                  blobCounter++;
                  foundIt = false;
                }
                else
                {
                  exists = false;
                }
              }
              newIDs[ii] = blobCounter;
            }
          }
        }

        //If there are more history then current points, assign all, keep unused in memory
        else if (pointVals.size() < OldCentroids.size())
        {
          double distances[pointVals.size()][OldCentroids.size()];

          for (int z = 0; z < pointVals.size(); z++)
          {
            for (int y = 0; y < OldCentroids.size(); y++)
            {
              distances[z][y] = cv::norm(pointVals.at(z)-OldCentroids.at(y));
            }
          }

          vector<double> minVals(pointVals.size(),10000);
          vector<int> minAt(pointVals.size(),500);

          for (int z = 0; z < pointVals.size(); z++)
          {
            for (int y = 0; y < OldCentroids.size(); y++)
            {
              if (distances[z][y] < minVals[z])
              {
                minVals[z] = distances[z][y];
                minAt[z] = y;
              }
            }
          }

          for (int z = 0; z < pointVals.size(); z++)
          {

            newIDs[z] = oldIDs[minAt[z]];
          }
        }




        // Check conditions, do update accordingly

        // If there are more new points than old, replace history
        if (pointVals.size() >= OldCentroids.size())
        {
          OldCentroids = pointVals;
          oldIDs = newIDs;
          lifeSpans.clear();

          for (int ll = 0; ll < oldIDs.size(); ll++)
          {
             lifeSpans.push_back(100);
          }
        }
        else
        // Remember old history
        {
          vector<int> tempIDs;
          vector<Point> tempCentroids;

          // iterate through oldIDs, decreasing life of any that haven't been assigned
          // and removing any whose life has gone to zero

          for (int kk = 0; kk < oldIDs.size(); kk++)
          {
             if (!taken[kk] && lifeSpans[kk] > 0)
             {
               lifeSpans[kk] -= 1;
             }
             if (lifeSpans[kk]!=0)
             {
               tempIDs.push_back(oldIDs[kk]);
             }
             if (taken[kk])
             {
               tempCentroids.push_back(pointVals[kk]);
             }
             else
             {
               tempCentroids.push_back(OldCentroids[kk]);
             }

          }
          oldIDs = tempIDs;
          OldCentroids = tempCentroids;
        }



        if (newIDs.size() > 0)
        {
          for (int nn = 0; nn < newIDs.size(); nn++)
          {
            char idVal [2];
            sprintf(idVal,"%d",newIDs[nn]);
            putText(drawingR,idVal,pointVals.at(nn),FONT_HERSHEY_DUPLEX,1,color3,2);
          }
        }

        //////////// Pose Estimation when 4 points detected///////////////////
        if (newIDs.size() == 4)
        {
          vector<int> imageIDs;
          std::vector<cv::Point2d> image_points;
          for (int nn = 0; nn < newIDs.size(); nn++)
          {
            for (int mm=0; mm < newIDs.size(); mm++)
            {
              if (newIDs[mm]==nn)
              {
                image_points.push_back(pointVals.at(mm));
              }
            }
          }

          // 3D model points.
          std::vector<cv::Point3d> model_points;
          float ledDistance = 25.00f; // in mm

          //As written, these points set the origin of the dome frame at the center of its base
          model_points.push_back(cv::Point3d(-ledDistance/2, -ledDistance/2, 0.0f));               // Nose tip
          model_points.push_back(cv::Point3d(ledDistance/2,-ledDistance/2,0.0f));          // Chin
          model_points.push_back(cv::Point3d(-ledDistance/2, ledDistance/2, 0.0f));       // Left eye left corner
          model_points.push_back(cv::Point3d(ledDistance/2,ledDistance/2, 0.0f));        // Right eye right corner



          // Solve for pose, returns a rotation vector and translation vector
          cv::solvePnP(model_points, image_points, cameraMatrix, distanceCoefficients, rotation_vector, translation_vector);

          // Convert the rotation vector to a rotation matrix for transformation
          Rodrigues(rotation_vector,rotationMatrix);

          // Project a 3D point onto the image plane, 1 in each direction
          // We use this to draw the frame
          vector<Point3d> z_end_point3D;
          vector<Point3d> dome_center;
          vector<Point3d> x_end_point3D, y_end_point3D;
          vector<Point2d> z_end_point2D;
          vector<Point2d> x_end_point2D;
          vector<Point2d> y_end_point2D;
          vector<Point2d> dome_center_2D;

          // THINK I CAN GET RID OF THESE 2 LINES, CHECK
          Point2d midX = Point2d((image_points[0].x + image_points[1].x)/2,(image_points[0].y+image_points[1].y)/2);
          Point2d midY = Point2d((image_points[0].x+image_points[3].x)/2, (image_points[0].y+image_points[3].y)/2);

          double xZero = 0;
          double yZero = 0;
          double axisLength = 2.0;
          z_end_point3D.push_back(Point3d(xZero,yZero,axisLength));
          x_end_point3D.push_back(Point3d(xZero+axisLength,yZero,0));
          y_end_point3D.push_back(Point3d(xZero,yZero+axisLength,0));
          dome_center.push_back(Point3d(xZero,yZero,0));

          projectPoints(z_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, z_end_point2D);
          projectPoints(x_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, x_end_point2D);
          projectPoints(y_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, y_end_point2D);
          projectPoints(dome_center,rotation_vector,translation_vector,cameraMatrix,distanceCoefficients,dome_center_2D);
          cv::line(drawingR,dome_center_2D[0], z_end_point2D[0], cv::Scalar(0,255,0), 3);
          cv::line(drawingR,dome_center_2D[0], x_end_point2D[0], cv::Scalar(255,0,0),3);
          cv::line(drawingR,dome_center_2D[0], y_end_point2D[0], cv::Scalar(100,0,255),3);

        }

        if (contCount > 0)
        {
          imshow( "Red Detection",drawingR);
        }

        //////////////////// CV for central portion //////////////////////////

        //Isolate bright spots in the central regions
        bitwise_not(undistorted,imInv);
        cvtColor(imInv,imInv,CV_BGR2GRAY);
        GaussianBlur(imInv,filtered,Size(7,7),0,0);
        threshold(filtered,filtered,170,255,0);
        GaussianBlur(filtered,filtered,Size(7,7),0,0);
        Canny(filtered,filtered,0,255,3,true);

        vector<vector<Point> > contours;
        vector<vector<Point> > outerContours;

        closeContours(3,filtered);
        dilateMat(2,filtered);
        bitwise_not(filtered,filtered);
        floodFill(filtered,cv::Point(0,0),Scalar(0));

        findContours(filtered,contours,hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

        /// Draw contours
        drawing = Mat::zeros( filtered.size(), CV_8UC3 );
        drawing2 = Mat::zeros( filtered.size(), CV_8UC3 );

        double area[contours.size()];
        double perims[contours.size()];
        double tol = .7;

        vector<Moments> mu(contours.size() );
        vector< vector<Point> > hull(contours.size());

        for( int i = 0; i< contours.size(); i++ )
        {
          approxPolyDP(contours[i],contours[i],1,true);
          convexHull(Mat(contours[i]), hull[i], false);
          mu[i] = moments(contours[i],false);
          area[i] = contourArea(contours[i],true);
          perims[i] = arcLength(contours[i],true);
        }

        Point2f ContCenter;

        for( int i = 0; i< contours.size(); i++ )
        {
          circularity = 4*M_PI*(area[i]/(perims[i]*perims[i]));

          if (-area[i] > 10000)
          {
            ContCenter = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);
            drawContours(drawing, hull, i, color, 2, 8, vector<Vec4i>(), 0, Point());
          }
        }

        bitwise_not(drawing,drawing);
        floodFill(drawing,cv::Point(10,10),Scalar(0));

        circle( drawing2, ContCenter, 15, color2, -1, 8, 0 );

				//opticalflow
				vector<uchar> status;
				vector<float> err;
        // Display the current image in the OpenCV display window.


        // Overlay the mask on the original image to focus only on the center
        undistorted.copyTo(drawing2,drawing);

        ///////////////// Now try to find contacts from this image //////////
        cvtColor(drawing2,drawing2,CV_BGR2GRAY);
        threshold(drawing2,drawing2,targetContact,255,0);
        GaussianBlur(drawing2,drawing2,Size(7,7),0,0);
        GaussianBlur(drawing2,drawing2,Size(7,7),0,0);
        Canny(drawing2,drawing2,0,255,3,true);

        closeContours(3,drawing2);
        dilateMat(3,drawing2);

        vector<vector<Point> > contactContours;

        findContours(drawing2,contactContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

        cvtColor(drawing2,drawing2,CV_GRAY2BGR);

        vector<Moments> muC(contactContours.size() );
        vector<Point2f> ContCenterC(contactContours.size());
        vector<Point2f>centers( contactContours.size() );
        vector<float>radius( contactContours.size() );
        double areaC[contactContours.size()];
        double perimC[contactContours.size()];
        double circularityC;

        Mat drawingC = Mat::zeros( mask.size(), CV_8UC3 );

        for( int i = 0; i< contactContours.size(); i++ )
        {
          approxPolyDP(contactContours[i],contactContours[i],1,true);
          muC[i] = moments(contactContours[i],false);
          areaC[i] = contourArea(contactContours[i],true);
          perimC[i] = arcLength(contactContours[i],true);
          ContCenterC[i] = Point2f(muC[i].m10/muC[i].m00, muC[i].m01/muC[i].m00);
          minEnclosingCircle( contactContours[i], centers[i], radius[i] );

          double circularityC = -4*M_PI*(areaC[i]/(perimC[i]*perimC[i]));

          if (-areaC[i] > 500 && circularityC > .5 && circularityC < 1.3)
          {
            //drawContours( drawingC, contactContours, i, color2, 3, 8, hierarchy, 0, Point() );
            circle( drawingC, ContCenterC[i], 8, color3, -1, 8, 0);
            circle( drawingC, centers[i], (int)radius[i], color2, 3 );

            cout << endl << endl;
          }
        }

        /// Determine 3D position of contacts from 2D image contacts detected ///
        vector<Point3d> trueContacts;
        for (int i=0; i < ContCenterC.size(); i++)
        {
          // Define some transformation matrices and points
          Mat Tsr;
          Mat Trd;
          Mat Pc;

          double rd = 38.1; // mm radius of dome
          double h = 9.5; // mm distance from reference plane to dome frame
          Mat Prd = (cv::Mat_<double>(3,1) << 0, 0, h);
          Mat Prd2 = (cv::Mat_<double>(4,1)<<0,0,h,1);
          Mat Identity = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

          // Define transformation matrix from camera frame to reference frame
          Mat TransformBottom = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);


          hconcat(rotationMatrix,translation_vector,Tsr);
          //vconcat(Tsr,TransformBottom, Tsr);

          // Define transformation matrix from reference frame to dome frame
          hconcat(Identity,Prd,Trd);
          //vconcat(Trd,TransformBottom, Trd);


          Mat Psd = Tsr*Prd2;

          // The point in 2D space
          Mat uvPoint = (cv::Mat_<double>(3,1) << ContCenterC[i].x, ContCenterC[i].y, 1);
          Mat invCMatrix = cameraMatrix.inv();

          // The undistorted 2D space
          Mat c2Prime = invCMatrix*uvPoint;

          double theta = acos(Psd.dot(c2Prime)/(norm(Psd)*norm(c2Prime)));
          double firstPart = norm(Psd)*cos(theta);
          double insideSqrt = pow(rd,2)-pow(norm(Psd),2)*pow(sin(theta),2);

          // Distance from camera to contact
          double MagSC = firstPart+sqrt(insideSqrt);

          //Scaling factor to project the undistorted 2D point into 3D space
          double s = MagSC/norm(c2Prime);

          // True 3D location of the point detected
          Mat Ctrue = invCMatrix*uvPoint*s;
          trueContacts.push_back(Point3d(Ctrue.at<double>(0,0),Ctrue.at<double>(1,0),Ctrue.at<double>(2,0)));


        }

        // If there are contacts, output the true points
        if (trueContacts.size() > 0)
          cout << trueContacts << endl << endl;

          
        // Combine the Red fiducial detection results with the contact results
        Mat totalDrawing;
        add(drawingC,drawingR,totalDrawing);
        addWeighted(totalDrawing,.7,undistorted,.3,0.0,finalMat);

        // Show the result
        imshow("Processed Value",finalMat);

				waitKey(1);



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

////////////////////////// Trackbar Function////////////////////////////////////

static void on_trackbarRed( int, void* )
{
   minHue = rHue_slider;
   minHueHigh = rHueHighMin_slider;
   maxHueHigh = rHueHighMax_slider;
   redThresh = rThresh_slider;

   cvtColor(undistorted,mask,CV_BGR2HSV);

   // Threshold the HSV image, keep only the red pixels
   Mat lower_red_hue_range;
   Mat upper_red_hue_range;

   //Finding Red
   inRange(mask, cv::Scalar(0, 100,100), cv::Scalar(minHue, 255,255), lower_red_hue_range);
   inRange(mask, cv::Scalar(minHueHigh, 100,100), cv::Scalar(maxHueHigh, 255,255), upper_red_hue_range);
   addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, mask);

   GaussianBlur(mask,mask,Size(7,7),0,0);
   medianBlur(mask,mask,5);
   medianBlur(mask,mask,5);

   dilateMat(20,mask);
   threshold(mask,mask,redThresh,255,0);

   createTrackbar( rMinHigh, "Red Detection", &rHueHighMin_slider, maxHueHigh, on_trackbarRed );


}

static void on_trackbarExp( int, void* )
{
   targetExposure = (double) exp_slider;

}

static void on_trackbarBlack( int, void* )
{
   targetBlack = (double) black_slider;

}

static void on_trackbarContact( int, void* )
{
   targetContact = (double) contact_slider;

}
