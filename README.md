# corridor_navigation
OpenCV Python code to detect the center line of the floor.

First, the image is blurred and Canny edge detection is used to detect the edges, followed by Connected component analysis to filter out irrelevant edges.
And finally, Hough line transform is applied to detect both the corner edges of the floor.
Using these 2 lines, the center line is calculated.

This can be used to enable a bot traverse through a corridor, by trying to orient the center column of the frame of on-board camera with the center line of the floor calculated above.
