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
    image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation = cv2.INTER_CUBIC)
    np_image_data = np.asarray(image)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    np_final = np.expand_dims(np_image_data,axis=0)
    return np_final

  def label_image(self, tensor):
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

# def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
#   with tf.Graph().as_default():
#     input_name = "file_reader"
#     output_name = "normalized"
#     file_reader = tf.read_file(file_name, input_name)
#     if file_name.endswith(".png"):
#       image_reader = tf.image.decode_png(file_reader, channels = 3,
#                                         name='png_reader')
#     elif file_name.endswith(".gif"):
#       image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
#                                                   name='gif_reader'))
#     elif file_name.endswith(".bmp"):
#       image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
#     else:
#       image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
#                                         name='jpeg_reader')
#     float_caster = tf.cast(image_reader, dtype=tf.float32) # casts values of image_reader to dtype=tf.float32
#     dims_expander = tf.expand_dims(float_caster, 0);
#     resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
#     normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
#     sess = tf.Session()
#     result = sess.run(normalized)
#     return result

# def label_characters(listImages):
#   plate = ""
#   graph = load_graph("model/128_0.50_ver2.pb")
#   input_name = "import/input"
#   output_name = "import/final_result"
#   input_operation = graph.get_operation_by_name(input_name);
#   output_operation = graph.get_operation_by_name(output_name);    
#   with tf.Session(graph=graph) as sess:
#     for image in listImages:
#       image = cv2.resize(image, dsize=(128, 128), interpolation = cv2.INTER_CUBIC)
#       np_image_data = np.asarray(image)
#       np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
#       np_final = np.expand_dims(np_image_data,axis=0)
#       start = time.time()
#       results = sess.run(output_operation.outputs[0],
#                       {input_operation.outputs[0]: np_final})
#       end=time.time()
#       results = np.squeeze(results)

#       top_k = results.argsort()[-1:][::-1]
#       labels = load_labels("model/128_0.50_labels_ver2.txt")
#       for i in top_k:
#         plate = plate + labels[i]
#   print('Plate: ', plate)

# if __name__ == "__main__":
#   file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
#   model_file = "model/retrained_graph.pb"
#   label_file = "model/retrained_labels.txt"
#   input_height = 128
#   input_width = 128
#   input_mean = 128
#   input_std = 128
#   input_layer = "input"
#   output_layer = "final_result"

#   parser = argparse.ArgumentParser()
#   parser.add_argument("--image", help="image to be processed")
#   parser.add_argument("--graph", help="graph/model to be executed")
#   parser.add_argument("--labels", help="name of file containing labels")
#   parser.add_argument("--input_height", type=int, help="input height")
#   parser.add_argument("--input_width", type=int, help="input width")
#   parser.add_argument("--input_mean", type=int, help="input mean")
#   parser.add_argument("--input_std", type=int, help="input std")
#   parser.add_argument("--input_layer", help="name of input layer")
#   parser.add_argument("--output_layer", help="name of output layer")
#   args = parser.parse_args()

#   if args.graph:
#     model_file = args.graph
#   if args.image:
#     file_name = args.image
#   if args.labels:
#     label_file = args.labels
#   if args.input_height:
#     input_height = args.input_height
#   if args.input_width:
#     input_width = args.input_width
#   if args.input_mean:
#     input_mean = args.input_mean
#   if args.input_std:
#     input_std = args.input_std
#   if args.input_layer:
#     input_layer = args.input_layer
#   if args.output_layer:
#     output_layer = args.output_layer

#   graph = load_graph(model_file)
#   t = read_tensor_from_image_file(file_name,
#                                   input_height=input_height,
#                                   input_width=input_width,
#                                   input_mean=input_mean,
#                                   input_std=input_std)

#   input_name = "import/" + input_layer
#   output_name = "import/" + output_layer
#   input_operation = graph.get_operation_by_name(input_name);
#   output_operation = graph.get_operation_by_name(output_name);

#   #Loading the file
#   img2 = cv2.imread(file_name)
#   #Format for the Mul:0 Tensor
#   img2 = cv2.resize(img2, dsize=(128, 128), interpolation = cv2.INTER_CUBIC)
#   #Numpy array
#   np_image_data = np.asarray(img2)
#   np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
#   #maybe insert float convertion here - see edit remark!
#   np_final = np.expand_dims(np_image_data,axis=0)
  
#   with tf.Session(graph=graph) as sess:
#     start = time.time()
#     results = sess.run(output_operation.outputs[0],
#                       {input_operation.outputs[0]: np_final})
#     end=time.time()
#   results = np.squeeze(results)

#   top_k = results.argsort()[-1:][::-1]
#   labels = load_labels(label_file)

#   print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
#   template = "{} (score={:0.5f})"
#   for i in top_k:
#     print(template.format(labels[i], results[i]))