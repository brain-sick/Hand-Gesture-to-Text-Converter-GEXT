# Hand-Gesture-to-Text-Converter-GEXT
The project uses CNN based approach to converthand gesture to form paragraphs.
The dataset is available in this link 

The file **NOTrequirement.txt** contains all the libraries I have in my system. So in case of some error appropriate version of required library could be installed. 
**The file NOTrequirement.txt is not required to be installed completely. **

Run the file  **MyApp2.py** to start the program.
Press enter to start bankground determination , make sure that you are to the left of your system, so that you don't interfer with the region of interest.
30 seconds after background determination starts, press **S** to start gesture recognition, a new window containing predicted output will be shown.
Now make valid gestures inside the region highlighted in front of your webcam.
While 'Gesture is expected' - Green Box, make some gesture for some time( about 6 seconds), till the Box turns blue.
Remove the gesture when 'Blank is expected'.
Follow the given gestures to make sentences or train on your own datset using **modelTrainer.py**.
**Note that for training the model you need to download the dataset from the given link and place it along all the files, that is the 'Dataset' folder near ModelTrainer.py**.
Maintain the sturcture same as the the 'Dataset' folder in the given link.
