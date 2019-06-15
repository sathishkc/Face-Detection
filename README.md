# openCV-DL (learning repository from Adrian pyimagesearch)

Usage of DNN in OpenCV 

**Detect faces in an image:**
python detect_face_from_images.py -i image.jpg -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel


**Detect faces in a webcam stream:**
python detect_face_from_webcam.py -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel


**Detect faces in a recorded video:**
python detect_face_from_recordedvideo.py -v VIDEO.mp4 -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel

