# corridor_navigation
This repository has the OpenCV Python code to detect the center line of the floor.

The following is the method followed:
First, the image is blurred and Canny edge detection is used to detect the edges, followed by Connected component analysis to filter out irrelevant edges.
And finally, Hough line transform is applied to detect both the corner edges of the floor.
Using these 2 edges, the center line is calculated.

This can be used to enable a bot traverse through a corridor, by moving forward while trying to orient the center column of the frame of on-board camera with the center line of the floor calculated above.
