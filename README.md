# CS-129.18-Project

- This project was created in partial fulfillment for the course requirements of CS 129.18 (Introduction to Pattern Recognition)
- The aim of the project is to utilize OpenCV for vehicle and pedestrian detection, tracking and classification.
- The project utilized two different methods to perform the detection, tracking and classification. These two methods are binary contour detection and the Viola-Jones algorithm.

# Running The Project

- There are two separate scripts which correspond to the two different methods that we have, *contour.py* and *viola-jones_code.py*

## Binary Contour Detection

1. Run the script by running the following on the terminal:

```
  python3 contour.py
```

2. You will then be prompted to select which video you would like to process from among the 3 videos that we have.
3. Then input the number on the terminal that corresponds to the video that you want to process [1 - video1, 2 - video2, 3 - video3]
4. Windows that correspond to the different stages of the processing that the video undergoes should appear.

## Viola-Jones 

1. Run the script by running the following on the terminal:

```
  python3 viola-jones_code.py
```

2. You will then be prompted to select which video you would like to process from among the 3 videos that we have.
3. Then input the number on the terminal that corresponds to the video that you want to process [1 - video1, 2 - video2, 3 - video3]
4. Then select which cascade you would like to use for the processing, the online built cascade or the one we personally built.
5. Afterwards a window should open that will show the processed output of the video.
6. The processed video will also be saved as viola-jones_output.mp4
