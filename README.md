# Hallway Floor Detection
This repository has the OpenCV Python code to detect the floor of a hallway.

## Method
- First, the image is blurred and Canny edge detection is used to detect the edges in the image.
- Second, Connected component analysis is done to filter out irrelevant edges.
- Then, Hough line transform is applied to detect both the corner edge-lines of the floor (lines of intersection of the floor and the walls of the hallway).
- The area between these two edge lines is marked as the floor of the hallway.
