Copyright 2023 The MediaPipe Authors. All Rights Reserved.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
Language Detector with MediaPipe Tasks
This notebook shows you how to use MediaPipe Tasks Python API to detect the language(s) in text.

Preparation
Let's start with installing MediaPipe.

!pip install -q mediapipe
Then download an off-the-shelf model. Check out the MediaPipe documentation for more language detection models that you can use.

!wget -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite
Running inference
Here are the steps to run lanugage detection using MediaPipe.

Check out the MediaPipe documentation to learn more about configuration options that this solution supports.

You can try the following examples to try the API out or enter your own text in the text bar.

English: To be, or not to be, that is the question

French - Il y a beaucoup de bouches qui parlent et fort peu de têtes qui pensent

Russian - это какой-то английский язык

Mixed - 分久必合合久必分

# Define the input text that you wants the model to classify.
INPUT_TEXT = "\u5206\u4E45\u5FC5\u5408\u5408\u4E45\u5FC5\u5206" #@param {type:"string"}
# STEP 1: Import the necessary modules.
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# STEP 2: Create a LanguageDetector object.
base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = text.LanguageDetectorOptions(base_options=base_options)
detector = text.LanguageDetector.create_from_options(options)

# STEP 3: Get the language detcetion result for the input text.
detection_result = detector.detect(INPUT_TEXT)

# STEP 4: Process the detection result and print the languages detected and
# their scores.

for detection in detection_result.detections:
  print(f'{detection.language_code}: ({detection.probability:.2f})')