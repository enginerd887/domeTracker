////
// Compare with smiledetect.cpp sample provided with opencv
////
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <ctime>
#include <sys/time.h>

#define SEC2MILLISEC 1000
#define MILLISEC2NANOSEC 1000000

//// Include files to use the PYLON API.
//#include <pylon/PylonIncludes.h>
//#ifdef PYLON_WIN_BUILD
//#    include <pylon/PylonGUI.h>
//#endif


//// Settings for using Basler USB cameras.
//#include <pylon/usb/BaslerUsbInstantCamera.h>
//typedef Pylon::CBaslerUsbInstantCamera Camera_t;
//typedef Pylon::CBaslerUsbCameraEventHandler CameraEventHandler_t; // Or use Camera_t::CameraEventHandler_t
//typedef Pylon::CBaslerUsbImageEventHandler ImageEventHandler_t; // Or use Camera_t::ImageEventHandler_t
//typedef Pylon::CBaslerUsbGrabResultPtr GrabResultPtr_t; // Or use Camera_t::GrabResultPtr_t
//using namespace Basler_UsbCameraParams;
//
//// Namespace for using pylon objects.
//using namespace Pylon;

using namespace std;

/////////////////////// Variable Definitions/////////////////////////////
cv::Mat cameraMatrix = cv::Mat::eye(3,3, CV_64F);
cv::Mat distanceCoefficients;

double targetExposure = 700.0;
double targetBlack = 0;

double startTime;
struct timeval tv, tstart;

cv::Mat frame, image;

//Timing Variables
struct timespec resStart, resEnd, ts;
double resTime;

//Colors used in the program
cv::Scalar color = cv::Scalar( 255, 255,255 );
cv::Scalar color2 = cv::Scalar(100,0,255);
cv::Scalar color3 = cv::Scalar(0,255,0);

//Colors used in the program
const static cv::Scalar colors[] =
        {
                cv::Scalar(255,0,0),
                cv::Scalar(255,128,0),
                cv::Scalar(255,255,0),
                cv::Scalar(0,255,0),
                cv::Scalar(0,128,255),
                cv::Scalar(0,255,255),
                cv::Scalar(0,0,255),
                cv::Scalar(255,0,255),
                cv::Scalar(255,255,255)
        };

// Variables and arrays for holding information from previous (old) frame
vector<cv::Point> OldCentroids;
vector<int> oldIDs;
vector<int> lifeSpans;

cv::Mat rotation_vector; // Rotation in axis-angle form
cv::Mat translation_vector;
cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
static bool useExtrinsicGuess = false;
int guessCounter = 0;

cv::Mat CtrueDome;

int touchCounter = 2;
////////////////////// Function Definitions /////////////////////////////

// Function for writing to a file
void write2File(cv::Mat& CtrueDome, bool writing) {
    char fileName[50];
    sprintf(fileName,"point%d.txt",touchCounter);
    ofstream myfile;
    myfile.open(fileName, ofstream::app);

    myfile << CtrueDome << '\n';
    myfile.close();
}

// Function for loading camera intrinsic parameters from text file
bool loadCameraCalibration(cv::Mat& cameraMatrix, cv::Mat& distanceCoefficients)
{
    ifstream inStream;
    inStream.open("../CameraParameters.txt");

    if(inStream)
    {
        uint16_t rows;
        uint16_t columns;

        inStream >> rows;
        inStream >> columns;

        cameraMatrix = cv::Mat(cv::Size(columns,rows), CV_64F);

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

        distanceCoefficients = cv::Mat::zeros(rows,columns, CV_64F);
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
void closeContours(int morph_size, cv::Mat& frame)
{
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

    morphologyEx(frame,frame,cv::MORPH_CLOSE,element,cv::Point(-1,-1),1);
}

// Function for dilating/blurring an image to facilitate detection
void dilateMat(int dilation_size,cv::Mat& frame)
{
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         cv::Point( dilation_size, dilation_size ) );
    /// Apply the dilation operation
    cv::dilate( frame,frame, element );
}

// TBB NOTE: we need these headers
#include <thread>
#include <tbb/concurrent_queue.h>
#include <tbb/pipeline.h>
volatile bool done = false; // volatile is enough here. We don't need a mutex for this simple flag.
struct ProcessingChainData
{
    cv::Mat img;
    vector<cv::Rect> faces, faces2;
    cv::Mat red,blue,blueMask, blueIso;
};
void detectAndDrawTBB( cv::VideoCapture capture, tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue);

int main( int argc, const char** argv )
{
    //namedWindow( "Starting Value", CV_WINDOW_NORMAL);
    //namedWindow( "Red", CV_WINDOW_NORMAL);
    //namedWindow( "Blue", CV_WINDOW_NORMAL);
    cv::namedWindow("Initial", CV_WINDOW_NORMAL);
    cv::namedWindow( "Total Results", CV_WINDOW_NORMAL);
    string inputName;
    bool tryflip;

    //////////Everything here is to get the camera up and running//////////////

    // The exit code of the sample application.
    int exitCode = 0;

    // Load the camera calibration parameters from a txt file
    loadCameraCalibration(cameraMatrix,distanceCoefficients);

    cv::VideoCapture capture;
//    Mat frame1;
    frame = capture.open("../video/Motion.avi");
    if (!capture.isOpened()) {
        printf("can not open ...\n");
        return -1;
    }
    //namedWindow("output", CV_WINDOW_AUTOSIZE);

    // Create an OpenCV image.
    cv::Mat openCvImage;

    // Start the grabbing of c_countOfImagesToGrab images.
    // The camera device is parameterized with a default configuration which
    // sets up free-running continuous acquisition.
    gettimeofday(&tstart, NULL);
    startTime = (tstart.tv_sec) * 1000 + (tstart.tv_usec) / 1000;
    double totalTime = 0;
    int cycleCount = 1;

    int64 startTime;
    clock_gettime(CLOCK_MONOTONIC, &resStart);
    cout << "Image grabbing has begun..." << endl;

        while (true)
        {
//            // Create an OpenCV image from a pylon image.
//            openCvImage = frame;
//
//            cout<<(int)(*(openCvImage.data+openCvImage.step[0]*800+openCvImage.step[1]*600))<<endl;

            tbb::concurrent_bounded_queue<ProcessingChainData *> guiQueue;
            guiQueue.set_capacity(2); // TBB NOTE: flow control so the pipeline won't fill too much RAM
            auto pipelineRunner = thread( detectAndDrawTBB, ref(capture),ref(guiQueue));

            startTime = cv::getTickCount();
            double frameCount = 0;
            // TBB NOTE: GUI is executed in main thread
            ProcessingChainData *pData=0;
            for(;! done;)
            {

                if (guiQueue.try_pop(pData))
                {

                    char c = (char)cv::waitKey(1);
                    if( c == 27 || c == 'q' || c == 'Q' )
                    {
                        done = true;
                    }

                    cv::Mat totalDrawing;
                    cv::Mat finalMat;

                    add(pData->red,pData->blue,totalDrawing);

                    addWeighted(totalDrawing,.6,pData->img,.4,0.0,finalMat);
                    //cout<<(int)(*(openCvImage.data+openCvImage.step[0]*800+openCvImage.step[1]*600))<<endl;
                    imshow("Initial",pData->img);
                    //imshow("Initial",openCvImage);
                    imshow("Total Results",finalMat);
                    clock_gettime(CLOCK_MONOTONIC, &resEnd);        // end timer
                    resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);
                    totalTime = totalTime + resTime;
                    double avgTime = totalTime/cycleCount;
                    cycleCount++;
                    cout << "FrameRate: " << 1/(avgTime/1000) << endl;
                    clock_gettime(CLOCK_MONOTONIC, &resStart);

                    //imshow("Red Value",red);
                    //imshow("Blue Value",blue);
                    ///imshow("Red Value",contact);

                    // end timer
                    //  resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);

                    //cout << "Program Complete: " << resTime << endl;
                    //write2File(CtrueDome,true,resTime);
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
//            waitKey(1);
        }

    return 0;
}


// TBB NOTE: This usage below is just for the tbb demonstration.
//           It is not an example for good OO code. The lambda
//           expressions are used to easily show the correleation
//           between the original code and the tbb code.
void detectAndDrawTBB(cv::VideoCapture capture,tbb::concurrent_bounded_queue<ProcessingChainData *> &guiQueue)
{

    tbb::parallel_pipeline(4, // TBB NOTE: (recommendation) NumberOfFilters
            // 1st filter
                           tbb::make_filter<void,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                       [&](tbb::flow_control& fc)->ProcessingChainData*
                                                                       {   // TBB NOTE: this filter feeds input into the pipeline

                                                                           // Image grabbed successfully?
                                                                           if (capture.read(frame))
                                                                           {
                                                                               // use the intrinsic camera parameters to correct image
                                                                               //frame = openCvImage;

                                                                               //clock_gettime(CLOCK_MONOTONIC, &resEnd);        // end timer
                                                                               //resTime = ((double)resEnd.tv_sec*SEC2MILLISEC + (double)resEnd.tv_nsec/MILLISEC2NANOSEC) - ((double)resStart.tv_sec*SEC2MILLISEC + (double)resStart.tv_nsec/MILLISEC2NANOSEC);

                                                                               //cout << "Frame captured: " << resTime << endl;
                                                                               //clock_gettime(CLOCK_MONOTONIC, &resStart);
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
                                                                           //cout<<
                                                                           return pData;
                                                                       }
                           )&
                                   // 2nd filter (Finds the red fiducials and calculates pose)
                                   tbb::make_filter<ProcessingChainData*,ProcessingChainData*>(tbb::filter::serial_in_order,
                                                                                               [&](ProcessingChainData *pData)->ProcessingChainData*
                                                                                               {
                                                                                                   cv::cvtColor(pData->img,pData->red,CV_BGR2HSV);

                                                                                                   // Threshold the HSV image, keep only the red pixels
                                                                                                   cv::Mat lower_red_hue_range;
                                                                                                   cv::Mat upper_red_hue_range;
                                                                                                   // find red in the input image
                                                                                                   cv::inRange(pData->red, cv::Scalar(0, 100,100), cv::Scalar(40, 255,255), lower_red_hue_range);
                                                                                                   cv::inRange(pData->red, cv::Scalar(140, 100,100), cv::Scalar(185, 255,255), upper_red_hue_range);
                                                                                                   cv::addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, pData->red);

                                                                                                   cv::GaussianBlur(pData->red,pData->red,cv::Size(7,7),0,0);
                                                                                                   dilateMat(3,pData->red);

                                                                                                   // Now find the red marker contours
                                                                                                   vector<vector<cv::Point> > rContours;
                                                                                                   vector<cv::Vec4i> hierarchy;
                                                                                                   dilateMat(4,pData->red);

                                                                                                   cv::findContours(pData->red,rContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0) );



                                                                                                   // Draw contours
                                                                                                   cv::Mat drawingR = cv::Mat::zeros( pData->red.size(), CV_8UC3 );

                                                                                                   double areaR[rContours.size()];
                                                                                                   vector<cv::Moments> muR(rContours.size() );

                                                                                                   //Calculate parameters for each red blob recognized
                                                                                                   for( int i = 0; i< rContours.size(); i++ )
                                                                                                   {
                                                                                                       muR[i] = moments(rContours[i],false);
                                                                                                       areaR[i] = contourArea(rContours[i],true);
                                                                                                   }

                                                                                                   vector<cv::Point2f> ContCenterR(rContours.size());
                                                                                                   vector<cv::Point> pointVals;

                                                                                                   int contCount = 0;
                                                                                                   int lineThickness = 2;
                                                                                                   int lineType = cv::LINE_8;

                                                                                                   // Perform the actual drawing on drawingR
                                                                                                   for( int i = 0; i< rContours.size(); i++ )
                                                                                                   {

                                                                                                       // if detected blobs are sufficiently large, keep them
                                                                                                       if ( -areaR[i] > 2000)
                                                                                                       {
                                                                                                           ContCenterR[i] = cv::Point2f(muR[i].m10/muR[i].m00, muR[i].m01/muR[i].m00);
                                                                                                           pointVals.push_back(cv::Point(ContCenterR[i]));

                                                                                                           //Draw centroids
                                                                                                           circle( drawingR, ContCenterR[i], 15, color2, -1, 8, .1);
                                                                                                           drawContours(drawingR, rContours, i, color, 2, 8, vector<cv::Vec4i>(), 0, cv::Point());

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
                                                                                                       vector<cv::Point> tempCentroids;

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
                                                                                                           //cout<<newIDs[nn]<<" ";
                                                                                                           putText(drawingR,idVal,pointVals.at(nn),cv::FONT_HERSHEY_DUPLEX,1,color3,2);
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
                                                                                                                   //cout<<pointVals.at(mm)<<endl;
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
                                                                                                       model_points.push_back(cv::Point3d(-ledDistance/2,ledDistance/2, dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(0,ledR,dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(-ledR,0,dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(ledDistance/2,ledDistance/2,dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(-ledDistance/2, -ledDistance/2, dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(ledR,0,dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(0,-ledR,dFrameOffset));
                                                                                                       model_points.push_back(cv::Point3d(ledDistance/2, -ledDistance/2, dFrameOffset));

                                                                                                       sort(newIDs.begin(),newIDs.end());
                                                                                                       std::vector<cv::Point3d> seen_model_points;
                                                                                                       for (int ll=0;ll<image_points.size();ll++){
                                                                                                           seen_model_points.push_back(model_points[newIDs[ll]]);
                                                                                                       }

//                                                                                                       for(int i=0;i<seen_model_points.size();i++){
//                                                                                                           //seen_model_points[i]=seen_model_points[i]*35.354428+Point3_<double>(550,600,0);
//                                                                                                           cout<<seen_model_points[i]<<" ";
//                                                                                                       }
//                                                                                                       cout<<endl;
//                                                                                                       for(int i=0;i<image_points.size();i++){
//                                                                                                           image_points[i]=image_points[i]-cv::Point_<double>(21,0);
//                                                                                                           //cout<<image_points[i]<<" ";
//                                                                                                       }
                                                                                                       //cout<<endl;


                                                                                                       // Solve for pose, returns a rotation vector and translation vector
                                                                                                       cv::solvePnP(seen_model_points, image_points, cameraMatrix, distanceCoefficients, rotation_vector, translation_vector,useExtrinsicGuess,CV_ITERATIVE);
                                                                                                       guessCounter++;

                                                                                                       cout<<translation_vector<<endl;

                                                                                                       if(!useExtrinsicGuess && guessCounter > 20){
                                                                                                           useExtrinsicGuess=true;
                                                                                                       }

                                                                                                       // Convert the rotation vector to a rotation matrix for transformation
                                                                                                       Rodrigues(rotation_vector,rotationMatrix);

                                                                                                       // Project a 3D point onto the image plane, 1 in each direction
                                                                                                       // We use this to draw the frame
                                                                                                       vector<cv::Point3d> z_end_point3D;
                                                                                                       vector<cv::Point3d> dome_center;
                                                                                                       vector<cv::Point3d> x_end_point3D, y_end_point3D;
                                                                                                       vector<cv::Point2d> z_end_point2D;
                                                                                                       vector<cv::Point2d> x_end_point2D;
                                                                                                       vector<cv::Point2d> y_end_point2D;
                                                                                                       vector<cv::Point2d> dome_center_2D;

                                                                                                       double xZero = 0;
                                                                                                       double yZero = 0;
                                                                                                       double axisLength = 2.0;
                                                                                                       z_end_point3D.push_back(cv::Point3d(xZero,yZero,axisLength));
                                                                                                       x_end_point3D.push_back(cv::Point3d(xZero+axisLength,yZero,0));
                                                                                                       y_end_point3D.push_back(cv::Point3d(xZero,yZero+axisLength,0));
                                                                                                       dome_center.push_back(cv::Point3d(xZero,yZero,0));

                                                                                                       projectPoints(z_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, z_end_point2D);
                                                                                                       projectPoints(x_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, x_end_point2D);
                                                                                                       projectPoints(y_end_point3D, rotation_vector, translation_vector, cameraMatrix, distanceCoefficients, y_end_point2D);
                                                                                                       projectPoints(dome_center,rotation_vector,translation_vector,cameraMatrix,distanceCoefficients,dome_center_2D);
                                                                                                       cv::line(drawingR,dome_center_2D[0], x_end_point2D[0], cv::Scalar(100,0,255),3);
                                                                                                       cv::line(drawingR,dome_center_2D[0], y_end_point2D[0], cv::Scalar(0,255,0),3);
                                                                                                       cv::line(drawingR,dome_center_2D[0], z_end_point2D[0], cv::Scalar(255,0,0), 3);
                                                                                                       cout<<dome_center_2D[0]<<endl;
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


                                                                                                   blur(pData->blue,pData->blue,cv::Size(5,5));

                                                                                                   inRange(pData->blue, cv::Scalar(75,100,100), cv::Scalar(90,255,255),pData->blueMask);
                                                                                                   cvtColor(pData->blue,pData->blue,CV_HSV2BGR);
                                                                                                   pData->img.copyTo(pData->blueIso, pData->blueMask);

                                                                                                   blur(pData->blueIso,pData->blueIso,cv::Size(3,3));

                                                                                                   vector<vector<cv::Point> > contactContours;
                                                                                                   vector<cv::Vec4i> hierarchy;
                                                                                                   cvtColor(pData->blueIso,pData->blueIso,CV_BGR2GRAY);
                                                                                                   closeContours(8,pData->blueIso);
                                                                                                   blur(pData->blueIso,pData->blueIso,cv::Size(3,3));
                                                                                                   findContours(pData->blueIso,contactContours,hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0) );
                                                                                                   cvtColor(pData->blueIso,pData->blueIso,CV_GRAY2BGR);
                                                                                                   //cout << "Blue Contours Detected: " << resTime << endl;

                                                                                                   vector<cv::Moments> muC(contactContours.size() );
                                                                                                   vector<cv::Point2f> ContCenterC(contactContours.size());
                                                                                                   vector<cv::Point2f>centers( contactContours.size() );
                                                                                                   vector<float>radius( contactContours.size() );
                                                                                                   double areaC[contactContours.size()];
                                                                                                   cv::Point2f ContCenter;

                                                                                                   cv::Mat drawingC = cv::Mat::zeros( pData->img.size(), CV_8UC3 );
                                                                                                   int contactCount = 0;
                                                                                                   vector<cv::Point2f> ContCenterReal;
                                                                                                   // Find the blue region center
                                                                                                   for( int i = 0; i< contactContours.size(); i++ )
                                                                                                   {
                                                                                                       muC[i] = moments(contactContours[i],false);
                                                                                                       areaC[i] = contourArea(contactContours[i],true);
                                                                                                       ContCenterC[i] = cv::Point2f(muC[i].m10/muC[i].m00, muC[i].m01/muC[i].m00);
                                                                                                       minEnclosingCircle( contactContours[i], centers[i], radius[i] );
                                                                                                       //approxPolyDP(Mat(contactContours[i]),contactContours[i],5,true);
                                                                                                       if (-areaC[i] > 250)//&& circularityC > .5)
                                                                                                       {
                                                                                                           ContCenterReal.push_back(cv::Point2f(ContCenterC[i]));
                                                                                                           drawContours(drawingC, contactContours, i, color2, 2, 8, vector<cv::Vec4i>(), 0, cv::Point());
                                                                                                           drawMarker(drawingC, centers[i],color3,cv::MARKER_CROSS,10,2);
                                                                                                           //circle(drawingC,centers[i],(int)radius[i],color4,2,8,0);

                                                                                                       }
                                                                                                   }
                                                                                                   pData->blueIso = drawingC;

                                                                                                   //cout << "Blue Contours Drawn: " << resTime << endl;

                                                                                                   /// Determine 3D position of contacts from 2D image contacts detected ///
                                                                                                   vector<cv::Point3d> trueContacts;
                                                                                                   vector<cv::Point2f> undistortedConts;
                                                                                                   //cout << ContCenterReal << endl;
                                                                                                   if (ContCenterReal.size() > 0)
                                                                                                   {
                                                                                                       undistortPoints(ContCenterReal,undistortedConts,cameraMatrix,distanceCoefficients);

                                                                                                   }


                                                                                                   cv::Mat contacts;

                                                                                                   cv::Mat Tsr;
                                                                                                   // Define transformation matrix from camera frame to reference frame
                                                                                                   cv::Mat TransformBottom = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);

                                                                                                   hconcat(rotationMatrix,translation_vector,Tsr);
                                                                                                   vconcat(Tsr,TransformBottom, Tsr);

                                                                                                   //writeTsd(Tsr);


                                                                                                   for (int i=0; i < ContCenterReal.size(); i++)
                                                                                                   {
                                                                                                       cv::Mat contactPixels;
                                                                                                       cv::Mat contactPixels2D;
                                                                                                       cv::Mat tempContact = (cv::Mat_<double>(1,3) << undistortedConts[i].x,undistortedConts[i].y,1);
                                                                                                       transpose(tempContact,tempContact);
                                                                                                       contactPixels = cameraMatrix*tempContact;
                                                                                                       transpose(contactPixels,contactPixels);

                                                                                                       contacts.push_back(contactPixels);
                                                                                                       hconcat(contactPixels.col(0),contactPixels.col(1),contactPixels2D);
                                                                                                       //circle( drawingC, Point2f(contactPixels2D), 4, color2, -1, 8, 0);

                                                                                                       // Define some transformation matrices and points

                                                                                                       cv::Mat Trd;
                                                                                                       cv::Mat Pc;

                                                                                                       double rd = 25.4/2.0; // mm radius of dome
                                                                                                       double h = 0; // mm distance from reference plane to dome frame
                                                                                                       cv::Mat Prd = (cv::Mat_<double>(3,1) << 0, 0, h);
                                                                                                       cv::Mat Prd2 = (cv::Mat_<double>(4,1)<<0,0,h,1);
                                                                                                       cv::Mat Identity = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

                                                                                                       // Define transformation matrix from camera frame to reference frame
                                                                                                       cv::Mat TransformBottom = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);


                                                                                                       hconcat(rotationMatrix,translation_vector,Tsr);
                                                                                                       vconcat(Tsr,TransformBottom, Tsr);

                                                                                                       // Define transformation matrix from reference frame to dome frame
                                                                                                       hconcat(Identity,Prd,Trd);
                                                                                                       vconcat(Trd,TransformBottom, Trd);


                                                                                                       cv::Mat Psd = Tsr*Prd2;
                                                                                                       Psd = (cv::Mat_<double>(3,1) << Psd.at<double>(0), Psd.at<double>(1), Psd.at<double>(2));

                                                                                                       // The point in 2D space
                                                                                                       cv::Mat uvPoint = contacts.row(i);
                                                                                                       cv::Mat invCMatrix = cameraMatrix.inv();


                                                                                                       // The undistorted 2D space
                                                                                                       transpose(uvPoint,uvPoint);
                                                                                                       cv::Mat c2Prime = invCMatrix*uvPoint;
                                                                                                       double theta = acos(Psd.dot(c2Prime)/(norm(Psd)*norm(c2Prime)));

                                                                                                       double firstPart = norm(Psd)*cos(theta);
                                                                                                       double insideSqrt = pow(rd,2)-pow(norm(Psd),2)*pow(sin(theta),2);

                                                                                                       // Distance from camera to contact
                                                                                                       double MagSC = firstPart+sqrt(insideSqrt);


                                                                                                       //Scaling factor to project the undistorted 2D point into 3D space
                                                                                                       double s = MagSC/norm(c2Prime);

                                                                                                       // True 3D location of the point detected
                                                                                                       cv::Mat Ctrue = c2Prime*s;
                                                                                                       cv::Mat Trs;
                                                                                                       //transpose(Tsr,Trs);
                                                                                                       Trs = Tsr.inv();
                                                                                                       cv::Mat Ctrue2 = (cv::Mat_<double>(4,1) << Ctrue.at<double>(0), Ctrue.at<double>(1), Ctrue.at<double>(2),1);

                                                                                                       trueContacts.push_back(cv::Point3d(Ctrue));

                                                                                                       CtrueDome = Trs*Ctrue2;
                                                                                                       CtrueDome = (cv::Mat_<double>(3,1) << CtrueDome.at<double>(0), CtrueDome.at<double>(1), CtrueDome.at<double>(2));
                                                                                                       transpose(CtrueDome,CtrueDome);
                                                                                                       //cout << CtrueDome << endl;
                                                                                                       //write2File(CtrueDome,true);
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
