# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import cv2


class NeuralNetwork():

  def __init__(self, modelFile, labelFile):
    self.model_file = modelFile
    self.label_file = labelFile
    
    self.label = self.load_labels(self.label_file)
    self.graph = self.load_graph(self.model_file)
    self.sess = tf.Session(graph=self.graph)


  def load_graph(self, modelFile):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(modelFile, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)
    return graph


  def load_labels(self, labelFile):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label


  def read_tensor_from_image(self, image, imageSizeOuput):
    """
    inputs an image and converts to a tensor
    """
    image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation = cv2.INTER_CUBIC)
    np_image_data = np.asarray(image)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.expand_dims(np_image_data,axis=0)
    return np_final


  def label_image(self, tensor):
    """
    for MobileNet
    """
    input_name = "import/input"
    output_name = "import/final_result"

    input_operation = self.graph.get_operation_by_name(input_name);
    output_operation = self.graph.get_operation_by_name(output_name);

    results = self.sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: tensor})
    results = np.squeeze(results)
    labels = self.label
    top_k = results.argsort()[-1:][::-1]
    return labels[top_k[0]]
    

  def label_image_list(self, listImages, imageSizeOuput):
    plate = ""
    for img in listImages:
      plate = plate + self.label_image(self.read_tensor_from_image(img, imageSizeOuput))
    # print(plate)
    return plate, len(plate)