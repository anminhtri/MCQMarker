# MCQMarker

<h1>Overview</h1>

The Automated Grading System is a powerful tool designed to evaluate student answers based on predefined correct responses. Using advanced image processing and deep learning techniques, the system extracts and compares answers, providing accurate scores efficiently.

<h1>Features </h1>

<b>Automatic Answer Detection:</b> Extracts student responses from scanned answer sheets.

<b>Customizable Answer Key:</b> Allows predefined correct answers for flexible grading.

<b>Multi-Format Support:</b> Works with handwritten, printed, or digital responses.

<b>Efficient Processing:</b> Reduces manual effort and speeds up grading.

<b>Detailed Reports:</b> Generates comprehensive feedback for students.

<h1>Technologies Used</h1>

<b>Python:</b>Python: Core programming language.

<b>OpenCV: </b>Image processing and object detection.

<b>Pandas:</b> Data processing and analysis.

<b>Convolutional Neural Networks (CNNs):</b> Deep learning model for detecting marked answers.

<h1>Installation</h1>

1.Clone the repository:

    git clone https://github.com/anminhtri/MCQMarker


2.Install dependencies:

    pip install -r requirements.txt


<h1>Usage</h1>

Provide scanned images of answer sheets.

Detect empty answer regions using OpenCV.

Use a CNN model to determine whether the bubbles are filled or not.

Define the answer key in a CSV file.

Run the grading script to automatically evaluate responses.

Retrieve the results from the output folder.

<h1>Preview</h1>
Use OpenCV to detect table

![alt text](README-screenshots/image-2.png)

Use CNN to detect colored cells

![alt text](README-screenshots/image-1.png)

Create an answer table

![alt text](README-screenshots/image-3.png)

Autograde

![alt text](README-screenshots/image-4.png)