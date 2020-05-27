////
// Compare with smiledetect.cpp sample provided with opencv
////
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <sys/time.h>

#define SEC2MILLISEC 1000
#define MILLISEC2NANOSEC 1000000

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

using namespace std;
using namespace cv;

/////////////////////// Variable Definitions/////////////////////////////
Mat cameraMatrix = Mat::eye(3,3, CV_64F);
Mat distanceCoefficients;

double targetExposure = 700.0;
double targetBlack = 0;

double startTime;
struct timeval tv, tstart;

Mat frame, image;

//Timing Variables
struct timespec resStart, resEnd, ts;
double resTime;

//Colors used in the program
Scalar color = Scalar( 255, 255,255 );
Scalar color2 = Scalar(100,0,255);
Scalar color3 = Scalar(0,255,0);
Scalar color4 = Scalar(0,0,255);
Scalar color5 = Scalar(0,255,100);

// Variables and arrays for holding information from previous (old) frame
vector<Point> OldCentroids;
vector<int> oldIDs;
vector<int> lifeSpans;
vector<int> taken;

cv::Mat rotation_vector; // Rotation in axis-angle form
cv::Mat translation_vector;
Mat rotationMatrix(3,3,cv::DataType<double>::type);
static bool useExtrinsicGuess = false;
int guessCounter = 0;

Point2f AverageCenter;
float AvgRadius;
int centerCounter = 0;

Mat CtrueDome;

int touchCounter = 0;
////////////////////// Function Definitions /////////////////////////////

// Function for writing to a file
void write2File(Mat& CtrueDome, bool writing) {
  char fileName[50];
  sprintf(fileName,"point%d.txt",touchCounter);
  ofstream myfile;
  myfile.open(fileName, ofstream::app);

  myfile << CtrueDome << '\n';
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

// TBB NOTE: we need these headers
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
volatile bool done = false; // volatile is enough here. We don't need a mutex for this simple flag.
struct ProcessingChainData
{
    Mat img;
    vector<Rect> faces, faces2;
    Mat red,blue,blueMask, blueIso;
};
void detectAndDrawTBB( Camera_t &camera,CGrabResultPtr ptrGrabResult,
                       tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue);

int main( int argc, const char** argv )
{
    //namedWindow( "Starting Value", CV_WINDOW_NORMAL);
    //namedWindow( "Red", CV_WINDOW_NORMAL);
    //namedWindow( "Blue", CV_WINDOW_NORMAL);
    namedWindow("Initial", CV_WINDOW_NORMAL);
    namedWindow( "Total Results", CV_WINDOW_NORMAL);
    string inputName;
    bool tryflip;

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

    // Create pointers to access the camera Width and Height parameters.
    GenApi::CIntegerPtr width= nodemap.GetNode("Width");
    GenApi::CIntegerPtr height= nodemap.GetNode("Height");

        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
    camera.MaxNumBuffer = 20;



    // Start the grabbing of c_countOfImagesToGrab images.
    // The camera device is parameterized with a default configuration which
    // sets up free-running continuous acquisition.
    gettimeofday(&tstart, NULL);
    startTime = (tstart.tv_sec)*1000+(tstart.tv_usec)/1000;
    camera.StartGrabbing(GrabStrategy_LatestImages);
    double totalTime = 0;
    int cycleCount = 1;
    // This smart pointer will receive the grab result data.
    CGrabResultPtr ptrGrabResult;

    int64 startTime;
    clock_gettime(CLOCK_MONOTONIC, &resStart);
    cout << "Image grabbing has begun..." << endl;
    while ( camera.IsGrabbing())
    {

      tbb::concurrent_bounded_queue<ProcessingChainData *> guiQueue;
      guiQueue.set_capacity(2); // TBB NOTE: flow control so the pipeline won't fill too much RAM
      auto pipelineRunner = thread( detectAndDrawTBB, ref(camera), ptrGrabResult,ref(guiQueue));

      startTime = getTickCount();
      double frameCount = 0;
      // TBB NOTE: GUI is executed in main thread
      ProcessingChainData *pData=0;
      for(;! done;)
      {

          if (guiQueue.try_pop(pData))
          {
              char c = (char)waitKey(1);
              if( c == 27 || c == 'q' || c == 'Q' )
              {
                  done = true;
              }

              Mat totalDrawing;
              Mat finalMat;

              add(pData->red,pData->blue,totalDrawing);

              addWeighted(totalDrawing,.6,pData->img,.4,0.0,finalMat);

              imshow("Initial",pData->img);
              imshow("Total Results",finalMat);
              clock_gettime(CLOCK_MONOTONIC, &resEnd);        // end timer
              resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);
              totalTime = totalTime + resTime;
              double avgTime = totalTime/cycleCount;
              cycleCount++;
              //cout << "FrameRate: " << 1/(avgTime/1000) << endl;
              clock_gettime(CLOCK_MONOTONIC, &resStart);

              //imshow("Red Value",red);
              //imshow("Blue Value",blue);
              ///imshow("Red Value",contact);

     // end timer
            //  resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);

              //cout << "Program Complete: " << resTime << endl;

              //clock_gettime(CLOCK_MONOTONIC, &resStart);
              delete pData;
              pData = 0;
          }
      }

      // TBB NOTE: flush the queue after marking 'done'
      do
      {
          delete pData;
      } while (guiQueue.try_pop(pData));
      pipelineRunner.join(); // TBB NOTE: wait for the pipeline to finish

    }

    }
    catch (GenICam::GenericException &e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
        << e.GetDescription() << endl;
        exitCode = 1;
    }


    return 0;
}

// TBB NOTE: This usage below is just for the tbb demonstration.
//           It is not an example for good OO code. The lambda
//           expressions are used to easily show the correleation
//           between the original code and the tbb code.
void detectAndDrawTBB( Camera_t &camera,CGrabResultPtr ptrGrabResult,
                       tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue)
{
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    tbb::parallel_pipeline(4, // TBB NOTE: (recommendation) NumberOfFilters
                           // 1st filter
                           tbb::make_filter<void,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](tbb::flow_control& fc)->ProcessingChainData*
                          {   // TBB NOTE: this filter feeds input into the pipeline
                            // Create a pylon ImageFormatConverter object.
                            CImageFormatConverter formatConverter;
                            // Specify the output pixel format.
                            formatConverter.OutputPixelFormat= PixelType_BGR8packed;
                            // Create a PylonImage that will be used to create OpenCV images later.
                            CPylonImage pylonImage;

                            // Create an OpenCV image.
                            Mat openCvImage;

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

                                // optical flow tracking algorithm

                                // use the intrinsic camera parameters to correct image
                                frame = openCvImage;

                                //clock_gettime(CLOCK_MONOTONIC, &resEnd);        // end timer
                                //resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);

                                //cout << "Frame captured: " << resTime << endl;
                                //clock_gettime(CLOCK_MONOTONIC, &resStart);




                                #ifdef PYLON_WIN_BUILD
                                #endif
                            }
                            else
                            {
                                //cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
                            }

                            if( done || frame.empty() )
                            {
                                // 'done' is our own exit flag
                                // being set and checked in and out
                                // of the pipeline
                                done = true;
                                // These 2 lines are how to tell TBB to stop the pipeline
                                fc.stop();
                                return 0;
                            }
                            auto pData = new ProcessingChainData;
                            pData->img = frame.clone();
                            return pData;
                          }
                          )&
                           // 2nd filter (Finds the red fiducials and calculates pose)
                           tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)->ProcessingChainData*
                          {
                              cvtColor(pData->img,pData->red,CV_BGR2HSV);

                              // Threshold the HSV image, keep only the red pixels
                              Mat lower_red_hue_range;
                              Mat upper_red_hue_range;
                              // find red in the input image
                              inRange(pData->red, cv::Scalar(0, 100,100), cv::Scalar(40, 255,255), lower_red_hue_range);
                              inRange(pData->red, cv::Scalar(140, 100,100), cv::Scalar(185, 255,255), upper_red_hue_range);
                              addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, pData->red);

                              GaussianBlur(pData->red,pData->red,Size(7,7),0,0);
                              dilateMat(3,pData->red);

                              // Now find the red marker contours
                              vector<vector<Point> > rContours;
                              vector<Vec4i> hierarchy;
                              dilateMat(4,pData->red);

                              findContours(pData->red,rContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );



                              // Draw contours
                              Mat drawingR = Mat::zeros( pData->red.size(), CV_8UC3 );

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


                              //cout << "Red Contours Drawn: " << resTime << endl;

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


                              //cout << "Red IDs Assigned: " << resTime << endl;

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

                            //  cout << "Pose Estimation Complete : " << resTime << endl;

                              pData->red = drawingR;

                              return pData;
                          }
                          )&
                          // 3rd filter (Contact Detection)
                          tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                [&](ProcessingChainData *pData)->ProcessingChainData*
                          {

                            cvtColor(pData->img,pData->blue,CV_BGR2HSV);


                            blur(pData->blue,pData->blue,Size(5,5));

                            inRange(pData->blue, cv::Scalar(75,100,100), cv::Scalar(90,255,255),pData->blueMask);
                            cvtColor(pData->blue,pData->blue,CV_HSV2BGR);
                            pData->img.copyTo(pData->blueIso, pData->blueMask);

                            blur(pData->blueIso,pData->blueIso,Size(3,3));

                            vector<vector<Point> > contactContours;
                            vector<Vec4i> hierarchy;
                            cvtColor(pData->blueIso,pData->blueIso,CV_BGR2GRAY);
                            closeContours(8,pData->blueIso);
                            blur(pData->blueIso,pData->blueIso,Size(3,3));
                            findContours(pData->blueIso,contactContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
                            cvtColor(pData->blueIso,pData->blueIso,CV_GRAY2BGR);
                            //cout << "Blue Contours Detected: " << resTime << endl;

                            vector<Moments> muC(contactContours.size() );
                            vector<Point2f> ContCenterC(contactContours.size());
                            vector<Point2f>centers( contactContours.size() );
                            vector<float>radius( contactContours.size() );
                            double areaC[contactContours.size()];
                            Point2f ContCenter;

                            Mat drawingC = Mat::zeros( pData->img.size(), CV_8UC3 );
                            int contactCount = 0;
                            vector<Point2f> ContCenterReal;
                            // Find the blue region center
                            for( int i = 0; i< contactContours.size(); i++ )
                              {
                                muC[i] = moments(contactContours[i],false);
                                areaC[i] = contourArea(contactContours[i],true);
                                ContCenterC[i] = Point2f(muC[i].m10/muC[i].m00, muC[i].m01/muC[i].m00);
                                minEnclosingCircle( contactContours[i], centers[i], radius[i] );
                                //approxPolyDP(Mat(contactContours[i]),contactContours[i],5,true);
                                if (-areaC[i] > 250)//&& circularityC > .5)
                                {
                                  ContCenterReal.push_back(Point2f(ContCenterC[i]));
                                  drawContours(drawingC, contactContours, i, color2, 2, 8, vector<Vec4i>(), 0, Point());
                                  drawMarker(drawingC, centers[i],color3,MARKER_CROSS,10,2);
                                  //circle(drawingC,centers[i],(int)radius[i],color4,2,8,0);

                                }
                              }
                              pData->blueIso = drawingC;

                              //cout << "Blue Contours Drawn: " << resTime << endl;

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
                                cout << CtrueDome << endl;
                                write2File(CtrueDome,true);
                                //writing = true;
                                //write2File(CtrueDome, true);
                                //cout << "Dome Contact: " << CtrueDome << endl;


                              }

                              //cout << "True Contacts Detected: " << resTime << endl;

                            pData->blue = drawingC;

                            return pData;
                          }

                          )&
                           // 4th filter
                           tbb::make_filter<ProcessingChainData*,void>(tbb::filter::serial_in_order,
                                                                  [&](ProcessingChainData *pData)
                          {   // TBB NOTE: pipeline end point. dispatch to GUI
                              if (! done)
                              {
                                  try
                                  {
                                      guiQueue.push(pData);
                                  }
                                  catch (...)
                                  {
                                      cout << "Pipeline caught an exception on the queue" << endl;
                                      done = true;
                                  }
                              }
                          }
                          )
                          );

}
