# Dome Tracking
Pylon/OpenCV tracking algorithm for master's thesis.

This program is for use with the VisiFlex sensor developed by the Lynch Lab at Northwestern University.

This sensor is capable of contact location detection and force measurement. The code in this repository can accomplish the following tasks:

* Read a Pylon-based camera (Basler Dart daA1600-60uc in this case)
* Convert Pylon-based images to OpenCV Mats
* Reads the input image and tracks any circular red regions found (fiducials for position tracking)
* Use the red fiducials and a PnP solver to estimate the pose of the dome in 3D space
* Determines what portion of the image is the dome region, and masks out everything else (used for contact location detection)
* Keeps track of the centroid of each red fiducial and of the main dome regions

* When a contact is detected, calculate its 3D position on the dome using its 2D position on the image plane, the pose estimation calculated earlier, and basic geometric knowledge of the system (dome diameter, etc.).
