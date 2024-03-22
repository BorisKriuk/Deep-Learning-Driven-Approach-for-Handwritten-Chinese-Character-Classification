# Deep Learning Driven Approach for Handwritten Chinese Character Classification
Code accompaniment for the research work submitted by Sparcus Technologies Limited on Deep Learning-Driven Approach for Handwritten Chinese Character Classification. <br>
Preprint can be found here: *not available yet* <br>

# Abstract <br>
Handwritten character recognition (HCR) is a challenging problem for machine learning researchers. Unlike printed text data, handwritten character datasets have more variation due to human-introduced bias. With numerous unique character classes present, some data, such as Logographic Scripts or Sino-Korean character sequences, bring new complications to the HCR problem. The classification task on such datasets requires the model to learn high-complexity details of the images that share similar features. With recent advances in computational resource availability and further computer vision theory development, some research teams have effectively addressed the arising challenges. Although known for achieving high accuracy while keeping the number of parameters small, many common approaches are still not generalizable and use dataset-specific solutions to achieve better results. Due to complex structure, existing methods frequently prevent the solutions from gaining popularity. This paper proposes a highly scalable approach for detailed character image classification by introducing the model architecture, data preprocessing steps, and testing design instructions. We also perform experiments to compare the performance of our method with that of existing ones to show the improvements achieved.

# Data <br>
CASIA-HWDB Dataset, Chinese Academy of Sciences: https://nlpr.ia.ac.cn/databases/handwriting/Home.html <br>
Benchmarks and Papers: https://paperswithcode.com/dataset/casia-hwdb <br>

 # Repository Structure <br>
_trainer1.py: Contains the code for model 1 training. <br>
_trainer2.py: Contains the code for model 2 training. <br>
_trainer3.py: Contains the code for model 3 training. <br>
_predictor1.py: Contains the code for model 1 prediction with multicrop. <br>
_predictor2.py: Contains the code for model 2 prediction with multicrop. <br>
_predictor3.py: Contains the code for model 3 prediction with multicrop. <br>
_predictor_finalizer.py: Contains the code for performing prediction ensembling at test time. <br>
README.md: This file, containing an overview of the repository. <br>
LICENSE: Apache 2.0 License. <br>

# About Authors <br>
Boris Kriuk is the Co-Founder and CEO at Sparcus Technologies Limited, Machine Learning Lead at Antei Limited, LinkedIn Top Machine Learning Voice. 
His research interests lie in the field of deep learning in computer vision. Boris Kriuk is the co-author of two conference papers. 
Boris Kriuk is an undergraduate student at the Hong Kong University of Science and Technology, pursuing a degree in Computer Engineering. <br>

Fedor Kriuk is the Co-Founder and CTO at Sparcus Technologies Limited. His research interests include statistical machine learning and deep learning in computer vision. Fedor Kriuk is the co-author of two conference papers. Fedor Kriuk is an undergraduate student at the Hong Kong University of Science and Technology, pursuing a degree in Electrical Engineering. <br>

