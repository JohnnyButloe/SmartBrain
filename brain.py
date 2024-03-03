# New Version:
# imageai.Prediction no longer exists, replaced by imageai.Classification
from imageai.Classification import ImageClassification
import os

exec_path = os.getcwd()

prediction = ImageClassification()
# SqueezeNet model also no longer exists, now the fastest is MobileNetV2
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(exec_path, 'resnet50-19c8e357.pth'))
prediction.loadModel()

predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'giraffe.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')

