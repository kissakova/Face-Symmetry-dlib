# Face-Symmetry-dlib
The current project deploys a Python dlib library to detect a face from the uploaded picture and allocate 68 facial landmarks. Enhanced code retrieves (x, y) coordinates from each point to calculate the Euclidean distance between facial feature points pairwise. The facial symmetry index is therefore calculated on the basis of the sum of pairwise deviations between main facial attributes from their midpoint on the face (e.g. the difference between the left-side jaw-center distance and right-side jaw-center distance + the difference between the left-corner nose-center distance and right-corner nose-center distance + etc.). Use the main.py file to obtain the Symmetry Index.

# Face-Ratio-Deviation
For the face ratio we deployed a framework of Farkas et.el.(1985), where they proposed nine neoclassical canons of facial horizontal and vertical proportions. Due to the limitation in terms of facial feature points allocations, we were able to deploy two horizontal proportions: Face Width = 4 * Nose Width; Eye Width = Interocular Distance. We thus calculated facial deviation index for those two proportions to be the sum of two deviations. You can run the ratiodeviation.py file to obtain the Face Ratio Deviation indexes.

If you, however, want to return the picture with with 68 allocated points, run the detect.py file. 


# Important!
Running the code requires the installation of heavy dependencies on your computer locally. Once you put all the files downloaded from the current repository on your Python environment you should be getting warning errors from the output that will require you to install packages. 
Please, install them all. 

If you follow the directions in the next section, you will get the Symmetry Index output in number. 

# How to run this code on PyCharm?
1. Create a new Project (File->New Project)
2. Find VCS on the top panel -> Get from Version Control -> paste current repo's URL -> Clone
(Alternatively, you can just transfer all the files yourself into your PyCharm project directory)
3. Download "shape_predictor_68_face_landmarks.dat" file from https://drive.google.com/file/d/1v0nGz_rvGeWp3eiJg_50-9IyLPTIoWwj/view?usp=share_li
4. {optional} Navigate the .dat file into the current pycharm project directory 
5. Install CMake Library (Python Packages on lower panel -> type CMake)
6. Install dlib library (this step takes a while)
7. Run "-m pip install --upgrade pip" in terminal
8. Install opencv-python package
9. Install matplotlib package
10. Install pandas package
12. Install scikit-learn package

* For path_to_folder (first instance):
Make sure to preliminarily add the images to the folder and only assign integer names to them. The only allowed restrictions are: .jpeg and .jpg. Copy the folder path and insert it into the code. 

* For predictor:
Make sure to copy the right path of the "shape_predictor_68_face_landmarks.dat" file 
