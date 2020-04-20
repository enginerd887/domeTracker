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
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
#include <tbb/tbb.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iterator>
#include <time.h>
#include <stdio.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <sys/time.h>

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
using namespace std::chrono;
// Namespace for parallel computing.
using namespace tbb;

//cvUseOptimized(true);
// Start time
volatile bool done = false;

////////////////////////// All different frame holders /////////////////////////

struct timeval tv, tstart;
double startTime, currentTime, diffTime;

Mat undistorted;
Mat red;
Mat blue;
Mat contact;
Mat backgroundImg;
Mat targetColor(1,1,CV_8UC3,Scalar(250,160,25));
// Mats to hold the rotation and translation information to
cv::Mat rotation_vector; // Rotation in axis-angle form
cv::Mat translation_vector;
Mat rotationMatrix(3,3,cv::DataType<double>::type);

//Colors used in the program
Scalar color = Scalar( 255, 255,255 );
Scalar color2 = Scalar(100,0,255);
Scalar color3 = Scalar(0,255,0);
Scalar color4 = Scalar(0,0,255);
Scalar color5 = Scalar(0,255,100);
// Variables for Camera setting sliders

char ExpString[50];
char BlackString[50];
char ContactString[50];

const int maxExposure = 7000;
const int maxBlack = 20;
const int maxContact = 255;

double targetExposure = 7000.0;
double targetBlack = 0;
double targetContact = 115;

Mat cameraMatrix = Mat::eye(3,3, CV_64F);
Mat distanceCoefficients;
Mat CtrueDome;

Mat blueArea;
static bool useExtrinsicGuess = false;
int guessCounter = 0;
int contactCounter = 0;
int touchCounter = 4;

// Variables and arrays for holding information from previous (old) frame
vector<Point> OldCentroids;
vector<int> oldIDs;
vector<int> lifeSpans;
vector<int> taken;

clock_t Start = clock();

Point2f AverageCenter;
float AvgRadius;
int centerCounter = 0;
///////////////////// Function Declarations ///////////////////////////////////

// Function for writing to a file
void write2File(Mat& CtrueDome, bool writing, double elapse) {
  char fileName[50];
  sprintf(fileName,"point%d.txt",touchCounter);
  ofstream myfile;
  myfile.open(fileName, ofstream::app);

  myfile << elapse << '\n';
  myfile.close();
}

// Function for writing to a file
void writeTsd(Mat& Tsr) {

  char fileName[50];
  sprintf(fileName,"Tiltpositive_newZ.txt");
  ofstream myfile;
  myfile.open(fileName, ofstream::app);
  myfile << Tsr << "\n\n" << (float)(clock()-Start)/CLOCKS_PER_SEC << "\n\n";

  myfile.close();
}

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

// This is the code that finds the red fiducials and performs the necessary math
// for displacement detection
class find_fiducials
    {

    private:
        cv::Mat img;
        cv::Mat& retVal;


    public:
        find_fiducials(cv::Mat inputImage, cv::Mat& outImage)
            : img(inputImage), retVal(outImage){}

        virtual void operator()() const
        {


            cvtColor(img,retVal,CV_BGR2HSV);

            // Threshold the HSV image, keep only the red pixels
            Mat lower_red_hue_range;
            Mat upper_red_hue_range;
            // find red in the input image
            inRange(retVal, cv::Scalar(0, 160,160), cv::Scalar(30, 255,255), lower_red_hue_range);
            inRange(retVal, cv::Scalar(150, 160,160), cv::Scalar(185, 255,255), upper_red_hue_range);
            addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, retVal);

            gettimeofday(&tv, NULL);
            currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
            diffTime = currentTime-startTime;
            //cout << "Found Red Fiducials: " << diffTime << endl;
            GaussianBlur(retVal,retVal,Size(7,7),0,0);
            dilateMat(3,retVal);

            // Now find the red marker contours
            vector<vector<Point> > rContours;
            vector<Vec4i> hierarchy;
            dilateMat(4,retVal);

            findContours(retVal,rContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

            // Draw contours
            Mat drawingR = Mat::zeros( retVal.size(), CV_8UC3 );

            double areaR[rContours.size()];
            vector<Moments> muR(rContours.size() );

            //Calculate parameters for each red blob recognized
            for( int i = 0; i< rContours.size(); i++ )
            {
              muR[i] = moments(rContours[i],false);
              areaR[i] = contourArea(rContours[i],true);
            }

            vector<Point2f> ContCenterR(rContours.size());
            vector<cv::Point> pointVals;

            int contCount = 0;
            int lineThickness = 2;
            int lineType = LINE_8;

            // Perform the actual drawing on drawingR
            for( int i = 0; i< rContours.size(); i++ )
            {

              // if detected blobs are sufficiently large, keep them
              if ( -areaR[i] > 2000)
              {
                ContCenterR[i] = Point2f(muR[i].m10/muR[i].m00, muR[i].m01/muR[i].m00);
                pointVals.push_back(Point(ContCenterR[i]));

                //Draw centroids
                circle( drawingR, ContCenterR[i], 15, color2, -1, 8, .1);
                drawContours(drawingR, rContours, i, color, 2, 8, vector<Vec4i>(), 0, Point());

                contCount++;
              }
            }

            gettimeofday(&tv, NULL);
            currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
            diffTime = currentTime-startTime;
            //cout << "Identified Red Contours: " << diffTime << endl;
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

            gettimeofday(&tv, NULL);
            currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
            diffTime = currentTime-startTime;
            //cout << "Assigned Fiducial IDs: " << diffTime << endl;
            //////////// Pose Estimation when 4 or more points detected///////////////

            if (newIDs.size() >= 4)
            {
              vector<int> imageIDs;
              std::vector<cv::Point2d> image_points;
              for (int nn = 0; nn < newIDs.size(); nn++)
              {
                for (int mm=0; mm < newIDs.size(); mm++)
                {
                  if (newIDs[mm]==nn && pointVals.at(mm).x > 1 && pointVals.at(mm).y > 1)
                  {
                    image_points.push_back(pointVals.at(mm));
                  }
                }
              }

              // 3D model points.
              std::vector<cv::Point3d> model_points;
              float ledDistance = 16.00f; // in mm
              float ledR = 11.314f; // radius from center
              float dFrameOffset = 0.00f; // in mm. Offset from reference plane to dome center of curvature
              //As written, these points set the origin of the dome frame at the center of its base
              model_points.push_back(cv::Point3d(-ledR,0,dFrameOffset));
              model_points.push_back(cv::Point3d(-ledDistance/2, -ledDistance/2, dFrameOffset));
              model_points.push_back(cv::Point3d(-ledDistance/2,ledDistance/2, dFrameOffset));
              model_points.push_back(cv::Point3d(0,-ledR,dFrameOffset));
              model_points.push_back(cv::Point3d(0,ledR,dFrameOffset));
              model_points.push_back(cv::Point3d(ledDistance/2, -ledDistance/2, dFrameOffset));
              model_points.push_back(cv::Point3d(ledDistance/2,ledDistance/2,dFrameOffset));
              model_points.push_back(cv::Point3d(ledR,0,dFrameOffset));


              sort(newIDs.begin(),newIDs.end());
              std::vector<cv::Point3d> seen_model_points;
              for (int ll=0;ll<image_points.size();ll++){
                seen_model_points.push_back(model_points[newIDs[ll]]);
              }

              // Solve for pose, returns a rotation vector and translation vector
              cv::solvePnP(seen_model_points, image_points, cameraMatrix, distanceCoefficients, rotation_vector, translation_vector,useExtrinsicGuess,CV_ITERATIVE);
              guessCounter++;

              if(!useExtrinsicGuess && guessCounter > 20){
                useExtrinsicGuess=true;
              }

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

            gettimeofday(&tv, NULL);
            currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
            diffTime = currentTime-startTime;
            //cout << "Pose Estimation Complete: " << diffTime << endl;
            retVal = drawingR;


        }
    };

// This code looks for the contact points on the dome
    class find_contacts
        {

        private:
            cv::Mat img;
            cv::Mat& retVal;

        public:
            find_contacts(cv::Mat inputImage, cv::Mat& outImage)
                : img(inputImage), retVal(outImage){}

            virtual void operator()() const
            {
              //retVal = img;
              auto start = std::chrono::steady_clock::now();
              retVal = img + img*.01;

              cvtColor(retVal,retVal,CV_BGR2HSV);

              blur(retVal,retVal,Size(5,5));
              blur(retVal,retVal,Size(3,3));

              inRange(retVal, cv::Scalar(90,205,255), cv::Scalar(102,255,255),retVal);


              gettimeofday(&tv, NULL);
              currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
              diffTime = currentTime-startTime;
              //cout << "Isolated Contacts: " << diffTime << endl;
              dilateMat(3,retVal);

              blueArea = img+img*.1;
              cvtColor(blueArea,blueArea,CV_BGR2HSV);

              inRange(blueArea, cv::Scalar(70,40,40), cv::Scalar(120, 255,255),contact);

              gettimeofday(&tv, NULL);
              currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
              diffTime = currentTime-startTime;
              //cout << "Isolated Blue Center Region: " << diffTime << endl;
              GaussianBlur(contact,contact,Size(3,3),0,0);

              vector<vector<Point> > contactContours;
              vector<vector<Point> > bigContour;
              vector<Vec4i> hierarchy;

              findContours(retVal,contactContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

              findContours(contact,bigContour,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE,Point(0,0) );
              cvtColor(retVal,retVal,CV_GRAY2BGR);

              vector<Moments> muC(contactContours.size() );
              vector<Moments> muB(bigContour.size() );
              vector<Point2f> ContCenterC(contactContours.size());
              vector<Point2f>centers( contactContours.size() );
              vector<Point2f>bigCenter(bigContour.size());
              vector<float>radius( contactContours.size() );
              vector<float>bigRadius(bigContour.size());
              double areaC[contactContours.size()];
              double areaB[bigContour.size()];
              Point2f ContCenter;

              Mat drawingC = Mat::zeros( img.size(), CV_8UC3 );
              int contactCount = 0;
              vector<Point2f> ContCenterReal;

              // Find the blue region center
              for( int i = 0; i< bigContour.size(); i++ )
              {
                areaB[i] = contourArea(bigContour[i],true);
                muB[i] = moments(bigContour[i],false);
                if (-areaB[i]>2000)
                {
                  minEnclosingCircle(bigContour[i],bigCenter[i],bigRadius[i]);
                  //drawContours(drawingC, bigContour, i, color5, 2, 8, vector<Vec4i>(), 0, Point());

                  ContCenter = Point2f(muB[i].m10/muB[i].m00, muB[i].m01/muB[i].m00);
                  if (centerCounter ==0)
                  {
                    AverageCenter = ContCenter;
                    AvgRadius = bigRadius[i];
                  }
                  else{
                    AverageCenter = Point2f((AverageCenter.x+ContCenter.x)/2,(AverageCenter.y+ContCenter.y)/2);
                    AvgRadius = (AvgRadius+bigRadius[i])/2;

                  }
                  circle(drawingC,AverageCenter,(int)AvgRadius,color5,2,8,0);
                }
              }

              gettimeofday(&tv, NULL);
              currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
              diffTime = currentTime-startTime;
              //cout << "Identified central contours: " << diffTime << endl;


              for( int i = 0; i< contactContours.size(); i++ )
              {
                muC[i] = moments(contactContours[i],false);
                areaC[i] = contourArea(contactContours[i],true);
                ContCenterC[i] = Point2f(muC[i].m10/muC[i].m00, muC[i].m01/muC[i].m00);
                minEnclosingCircle( contactContours[i], centers[i], radius[i] );
                //approxPolyDP(Mat(contactContours[i]),contactContours[i],5,true);
                if (-areaC[i] > 400 && abs(ContCenterC[i].x - AverageCenter.x)>10)//&& circularityC > .5)
                {
                  ContCenterReal.push_back(Point2f(ContCenterC[i]));
                  //drawContours(drawingC, contactContours, i, color2, 2, 8, vector<Vec4i>(), 0, Point());
                  drawMarker(drawingC, centers[i],color3,MARKER_STAR,30,2);
                  circle(drawingC,centers[i],(int)radius[i],color2,2,8,0);

                }
              }

              gettimeofday(&tv, NULL);
              currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
              diffTime = currentTime-startTime;
              //cout << "Contacts drawn on image: " << diffTime << endl;


                /// Determine 3D position of contacts from 2D image contacts detected ///
                vector<Point3d> trueContacts;
                vector<Point2f> undistortedConts;
                //cout << ContCenterReal << endl;
                if (ContCenterReal.size() > 0)
                {
                  undistortPoints(ContCenterReal,undistortedConts,cameraMatrix,distanceCoefficients);

                }


                Mat contacts;

                Mat Tsr;
                // Define transformation matrix from camera frame to reference frame
                Mat TransformBottom = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);

                hconcat(rotationMatrix,translation_vector,Tsr);
                vconcat(Tsr,TransformBottom, Tsr);

                //writeTsd(Tsr);


                for (int i=0; i < ContCenterReal.size(); i++)
                {
                  Mat contactPixels;
                  Mat contactPixels2D;
                  Mat tempContact = (cv::Mat_<double>(1,3) << undistortedConts[i].x,undistortedConts[i].y,1);
                  transpose(tempContact,tempContact);
                  contactPixels = cameraMatrix*tempContact;
                  transpose(contactPixels,contactPixels);

                  contacts.push_back(contactPixels);
                  hconcat(contactPixels.col(0),contactPixels.col(1),contactPixels2D);
                  //circle( drawingC, Point2f(contactPixels2D), 4, color2, -1, 8, 0);

                  // Define some transformation matrices and points

                  Mat Trd;
                  Mat Pc;

                  double rd = 25.4/2.0; // mm radius of dome
                  double h = 0; // mm distance from reference plane to dome frame
                  Mat Prd = (cv::Mat_<double>(3,1) << 0, 0, h);
                  Mat Prd2 = (cv::Mat_<double>(4,1)<<0,0,h,1);
                  Mat Identity = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

                  // Define transformation matrix from camera frame to reference frame
                  Mat TransformBottom = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);


                  hconcat(rotationMatrix,translation_vector,Tsr);
                  vconcat(Tsr,TransformBottom, Tsr);

                  // Define transformation matrix from reference frame to dome frame
                  hconcat(Identity,Prd,Trd);
                  vconcat(Trd,TransformBottom, Trd);


                  Mat Psd = Tsr*Prd2;
                  Psd = (cv::Mat_<double>(3,1) << Psd.at<double>(0), Psd.at<double>(1), Psd.at<double>(2));

                  // The point in 2D space
                  Mat uvPoint = contacts.row(i);
                  Mat invCMatrix = cameraMatrix.inv();


                  // The undistorted 2D space
                  transpose(uvPoint,uvPoint);
                  Mat c2Prime = invCMatrix*uvPoint;
                  double theta = acos(Psd.dot(c2Prime)/(norm(Psd)*norm(c2Prime)));

                  double firstPart = norm(Psd)*cos(theta);
                  double insideSqrt = pow(rd,2)-pow(norm(Psd),2)*pow(sin(theta),2);

                  // Distance from camera to contact
                  double MagSC = firstPart+sqrt(insideSqrt);


                  //Scaling factor to project the undistorted 2D point into 3D space
                  double s = MagSC/norm(c2Prime);

                  // True 3D location of the point detected
                  Mat Ctrue = c2Prime*s;
                  Mat Trs;
                  //transpose(Tsr,Trs);
                  Trs = Tsr.inv();
                  Mat Ctrue2 = (cv::Mat_<double>(4,1) << Ctrue.at<double>(0), Ctrue.at<double>(1), Ctrue.at<double>(2),1);

                  trueContacts.push_back(Point3d(Ctrue));

                  CtrueDome = Trs*Ctrue2;
                  CtrueDome = (cv::Mat_<double>(3,1) << CtrueDome.at<double>(0), CtrueDome.at<double>(1), CtrueDome.at<double>(2));
                  transpose(CtrueDome,CtrueDome);
                  //writing = true;
                  //write2File(CtrueDome, true);
                  //cout << "Dome Contact: " << CtrueDome << endl;


                }

                gettimeofday(&tv, NULL);
                currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
                diffTime = currentTime-startTime;
                //
                //cout << "True contact location calculated: " << diffTime << endl;
                retVal = drawingC;

            }
        };

////////////////////////// Program Setup //////////////////////////////////////

void display_vector(const vector<int> &v) //note the const
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<int>(std::cout, " "));
}


/////////////////////////// Main Program ///////////////////////////////////////
double totalTime;
double averageTime;
int timeCount = 0;

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
    camera.AcquisitionFrameRate.SetValue(60.0);
    // Set exposure and black level to default
    camera.ExposureTime.SetValue(targetExposure);
    camera.BlackLevel.SetValue(-targetBlack);
    camera.Gain.SetValue(10);
    camera.SensorShutterMode.SetValue(SensorShutterMode_Global);
    camera.PixelFormat.SetValue(PixelFormat_BayerRG8);
    //camera.OverlapMode.SetValue(OverlapMode_On);
    // Set the Camera ROI
    int64_t maxHeight = camera.Height.GetMax();
    int64_t maxWidth = camera.Width.GetMax();
    camera.Height.SetValue(maxHeight);
    camera.Width.SetValue(1100);
    camera.OffsetX.SetValue(250);
    camera.OffsetY.SetValue(0);

    // Set Binning (not used for Visiflex sensor)
    /*
    camera.BinningHorizontal.SetValue(2);
    camera.BinningVertical.SetValue(2);
    camera.BinningHorizontalMode.SetValue(BinningHorizontalMode_Average);
    camera.BinningVerticalMode.SetValue(BinningVerticalMode_Average);
    */
		// Create pointers to access the camera Width and Height parameters.
		GenApi::CIntegerPtr width= nodemap.GetNode("Width");
		GenApi::CIntegerPtr height= nodemap.GetNode("Height");

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 20;

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
        gettimeofday(&tstart, NULL);
        startTime = (tstart.tv_sec)*1000+(tstart.tv_usec)/1000;
		camera.StartGrabbing(GrabStrategy_LatestImages);

        // This smart pointer will receive the grab result data.
        CGrabResultPtr ptrGrabResult;

    int AvgCounter = 0;
        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {
          auto start = std::chrono::steady_clock::now();
          gettimeofday(&tv, NULL);
          currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
          diffTime = currentTime-startTime;
          //
          //cout << "Captured a frame: " << diffTime << endl;

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

        namedWindow( "Processed Value", CV_WINDOW_NORMAL);;
        //namedWindow( "Red Value", CV_WINDOW_NORMAL);;
        //namedWindow( "Blue Value", CV_WINDOW_NORMAL);;
        namedWindow( "Total Results", CV_WINDOW_NORMAL);;

				// optical flow tracking algorithm

        // use the intrinsic camera parameters to correct image
        undistorted = openCvImage;


        imshow("Processed Value",undistorted);

        task_group tg;
        tg.run(find_fiducials(undistorted,red)); // spawn 1st task and return
        tg.run(find_contacts(undistorted,blue)); // spawn 2nd task and return
        tg.wait( );             // wait for tasks to complete

        //imshow("Red Value",blue);
        //imshow("Blue Value",contact);

        Mat totalDrawing;
        Mat finalMat;
        add(red,blue,totalDrawing);

        addWeighted(totalDrawing,.6,undistorted,.4,0.0,finalMat);
        //imshow("Red Value",red);
        //imshow("Blue Value",blue);
        ///imshow("Red Value",contact);

        imshow("Total Results",finalMat);

        //imshow("Blue Value",blue);

        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        gettimeofday(&tv, NULL);
        currentTime = (tv.tv_sec)*1000+(tv.tv_usec)/1000;
        diffTime = currentTime-startTime;
        //cout << "Program Complete: " << diffTime << endl;
        //cout << "It took me " << elapsed.count()/1000 << " milliseconds." << endl;
        int64_t lastBlockId = camera.GetStreamGrabberParams().Statistic_Missed_Frame_Count.GetValue();
        cout << "Last Block ID: " << lastBlockId << endl;
        write2File(CtrueDome,true,elapsed.count()/1000);
        gettimeofday(&tstart, NULL);
        startTime = (tstart.tv_sec)*1000+(tstart.tv_usec)/1000;
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
