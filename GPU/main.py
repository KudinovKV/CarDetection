# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
from model import yolov3
from sort import *
import random

LIMIT = 1500
anchor_path = "./param/yolo_anchors.txt"
class_name_path = "./param/coco.names"
restore_path = "./param/yolov3-416.ckpt"
image_size = [416, 416]

# Видос с парковки
# line = [(250, 80), (240, 92)]
# line2 = [(240, 92), (230, 102)]

# SPB - Марсово поле
# line = [(1400, 1250), (200, 2300)]
# line2 = [(2000, 600), (2800, 600)]
# POS_TIME = (1400, 2800)

# Новгород
line = [(1900, 1700), (2100, 2000)]
line2 = [(2100, 2000), (2400, 2300)]
POS_TIME = (1400, 300)

COLOR = (0 , 255 , 128)
COLOR_GREEN = (0, 255, 0)

FONT_SIZE = 10 
FONT_SCALE = 10.0

POS_YELLOW = (100,300)
POS_PINK = (3500,300)


def plot_one_box(img , coord , label = None , color = None , line_thickness = None):
	tl = line_thickness or int(round(0.002 * max(img.shape[0:2]))) 
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
	cv2.rectangle(img, c1, c2, color, thickness = tl)
	if label :
		tf = max(tl - 1, 1)
		t_size = cv2.getTextSize(label, 0, fontScale = float(tl) / 3, thickness = tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1)
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness = tf, lineType = cv2.LINE_AA)


def gpu_nms(boxes, scores, num_classes, max_boxes = 50, score_thresh = 0.5, nms_thresh = 0.5):
	boxes_list, label_list, score_list = [], [], []
	max_boxes = tf.constant(max_boxes, dtype = 'int32')
	boxes = tf.reshape(boxes, [-1, 4]) 
	score = tf.reshape(scores, [-1, num_classes])
	mask = tf.greater_equal(score, tf.constant(score_thresh))
	for i in range(num_classes):
		filter_boxes = tf.boolean_mask(boxes, mask[:,i])
		filter_score = tf.boolean_mask(score[:,i], mask[:,i])
		nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
												   scores=filter_score,
												   max_output_size=max_boxes,
												   iou_threshold=nms_thresh, name='nms_indices')
		label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
		boxes_list.append(tf.gather(filter_boxes, nms_indices))
		score_list.append(tf.gather(filter_score, nms_indices))
	boxes = tf.concat(boxes_list, axis=0)
	score = tf.concat(score_list, axis=0)
	label = tf.concat(label_list, axis=0)
	return boxes, score, label


def parse_anchors(anchor_path) :
	anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
	return anchors


def read_class_names(class_name_path) :
	names = {}
	with open(class_name_path, 'r') as data:
		for ID, name in enumerate(data):
			names[ID] = name.strip('\n')
	return names


def Clear() : 
	files = glob.glob('output/*.png')
	for f in files:
   		os.remove(f)


def Intersect(A,B,C,D) :
	return CCW(A,C,D) != CCW(B,C,D) and CCW(A,B,C) != CCW(A,B,D)


def CCW(A,B,C) :
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def ParseArgs() :
	parser = argparse.ArgumentParser(description = " YOLO-v3 car detection using Tensorflow-GPU. ")
	parser.add_argument("--input" , type = str , help = "The path of the input video.")
	parser.add_argument("--output" , type = str , help = "The path of the output video.")
	return parser.parse_args()


def Init(args) :
	anchors = parse_anchors(anchor_path)
	classes = read_class_names(class_name_path)
	num_class = len(classes)
	vid = cv2.VideoCapture(args.input)
	video_frame_cnt = int(vid.get(7))
	video_width = int(vid.get(3))
	video_height = int(vid.get(4))
	video_fps = int(vid.get(5))
	print("[ FRAME COUNT ] " + str(video_frame_cnt))
	print("[ WIDTH ] " + str(video_width))
	print("[ HEIGHT ] " + str(video_height))
	print("[ FPS ] " + str(video_fps))
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	videoWriter = cv2.VideoWriter(args.output , fourcc, video_fps, (video_width, video_height))
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 1.0
	config.gpu_options.allow_growth = True
	session = tf.Session(config = config)
	return session , videoWriter , vid , video_frame_cnt , video_width , video_height , video_fps , anchors , classes , num_class


def ParseVideo(session , videoWriter , vid , video_frame_cnt , video_width , video_height , video_fps , anchors , classes , num_class , args):
	counter = 0
	counter2 = 0
	tracker = Sort()
	memory = {}
	counter = 0
	counter2  = 0
	sec_x = []
	sec_y_to = []
	sec_y_from = []
	frame_x = []
	frame_y_to = []
	frame_y_from = []
	input_data = tf.placeholder(tf.float32, [1, image_size[1], image_size[0], 3], name='input_data')
	yolo_model = yolov3(num_class, anchors)
	with tf.variable_scope('yolov3'):
		pred_feature_maps = yolo_model.forward(input_data, False)
	pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
	pred_scores = pred_confs * pred_probs
	boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=1000, score_thresh=0.5, nms_thresh=0.5)
	saver = tf.train.Saver()
	saver.restore(session, restore_path)
	for i in range(video_frame_cnt):
		ret, frame = vid.read()
		height_ori, width_ori = frame.shape[:2]
		img = cv2.resize(frame, tuple(image_size))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = np.asarray(img, np.float32)
		img = img[np.newaxis, :] / 255.
		start_time = time.time()
		boxes_, scores_, labels_ = session.run([boxes, scores, labels], feed_dict={input_data: img})
		end_time = time.time()
		objects = []
		boxes_[:, 0] *= (width_ori/float(image_size[0]))
		boxes_[:, 2] *= (width_ori/float(image_size[0]))
		boxes_[:, 1] *= (height_ori/float(image_size[1]))
		boxes_[:, 3] *= (height_ori/float(image_size[1]))
		for j in range(len(boxes_)):	
			(x, y) = (boxes_[j][0], boxes_[j][1])
			(x1 , y1) = (boxes_[j][2], boxes_[j][3])
			(w, h) = (x1 - x, y1 - y)
			objects.append([x, y, x+w, y+h, scores_[j]])
		np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
		objects = np.asarray(objects)
		tracks = tracker.update(objects)
		boundaries = []
		IDs = []
		prev = memory.copy()
		memory = {}
		for track in tracks:
			boundaries.append([track[0], track[1], track[2], track[3]])
			IDs.append(int(track[4]))
			memory[IDs[-1]] = boundaries[-1]
		if len(boundaries) > 0:
			k = int(0)
			for square in boundaries:
				(x, y) = (int(square[0]), int(square[1]))
				(w, h) = (int(square[2]), int(square[3]))
				cv2.rectangle(frame, (x, y), (w, h), COLOR, FONT_SIZE)
				if IDs[k] in prev:
					previous_box = prev[IDs[k]]
					(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
					(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
					p0 = (int(x + (w - x) / 2) , int(y + (h - y) / 2))
					p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
					cv2.line(frame, p0, p1, COLOR, FONT_SIZE)
					if Intersect(p0, p1, line[0], line[1]):
						counter += 1
					if Intersect(p0, p1, line2[0], line2[1]):
						counter2 += 1 
				cv2.putText(frame, "{}".format(classes[labels_[k]]) , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE // 2 , COLOR, FONT_SIZE // 2)
				k += 1
		cv2.line(frame, line[0], line[1], (0, 255, 255), FONT_SIZE)
		cv2.line(frame, line2[0], line2[1], (255, 0, 255), FONT_SIZE)
		cv2.putText(frame, str(counter), POS_YELLOW , cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, (0, 255, 255), FONT_SIZE)
		cv2.putText(frame, str(counter2), POS_PINK, cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, (255, 0, 255), FONT_SIZE)
		cv2.putText(frame, '{:.2f}ms'.format((end_time - start_time) * 1000), POS_TIME , cv2.FONT_HERSHEY_DUPLEX , FONT_SCALE , COLOR_GREEN, FONT_SIZE)
		# Раскомментируйте для получения онлайн-трансляции
		# cv2.imshow(' YOLO-v3 car detection using Tensorflow-GPU. ', frame)
		# Раскомментируйте для получения покадрового вывода видео
		# cv2.imwrite("output/frame-{}.png".format(i), frame)
		videoWriter.write(frame)
		frame_x.append(i)
		frame_y_to.append(counter)
		frame_y_from.append(counter2)
		if i % video_fps == 0 :
			sec_x.append(i // video_fps)
			sec_y_to.append(counter)
			sec_y_from.append(counter2)
		if i >= LIMIT:
			break
	vid.release()
	videoWriter.release()		
	return sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from


def DrawGraf(sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from):
	
	print('[ + ] Draw graf ')
	
	plt.subplot(211)
	plt.plot(sec_x , sec_y_to , 'y--' , sec_x , sec_y_from , 'm--' )
	plt.ylabel('Количество машин')
	plt.xlabel('Количество секунд')
	
	plt.subplot(212)
	plt.plot(frame_x , frame_y_to , 'y--' , frame_x , frame_y_from , 'm--' )
	plt.ylabel('Количество машин')
	plt.xlabel('Количество кадров')
	
	plt.show()


def main() :
	Clear()
	args = ParseArgs()
	session , videoWriter , vid , video_frame_cnt , video_width , video_height , video_fps , anchors , classes , num_class = Init(args)
	sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from = ParseVideo(session , videoWriter , vid , video_frame_cnt , video_width , video_height , video_fps , anchors , classes , num_class , args)
	DrawGraf(sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from)


if __name__ == '__main__':
	main()
