import numpy as np
import argparse
import imutils
import time
import cv2  
import os
import glob
from sort import *
import matplotlib.pyplot as plt

LIMIT = 1500

# Запускать программу следующим образом :
# python main.py --input <input filename> --output <output filename> --yolo <param dir>

# SPB - Марсово поле
# line = [(1400, 1250), (200, 2300)]
# line2 = [(2000, 600), (2800, 600)]

# Нижний Новгород
line = [(1900, 1700), (2100, 2000)]
line2 = [(2100, 2000), (2400, 2300)]

# Тестовый - хороший
# line = [(100, 500), (450, 520)]
# line2 = [(660, 530), (1060, 540)]

# Тестовый - хороший 2
# line = [(200, 600), (640 , 600)]
# line2 = [(640, 600), (1050, 600)]
 
# Очищаем директорию 
def Clear() : 
	files = glob.glob('output/*.png')
	for f in files:
   		os.remove(f)

# Парсим аргументы
def ParseArgs() :
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True, help="path to input video")
	ap.add_argument("-o", "--output", required=True, help="path to output video")
	ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
	args = vars(ap.parse_args())
	return args

# Возвращает true, если линии пересеклись
def Intersect(A,B,C,D):
	return CCW(A,C,D) != CCW(B,C,D) and CCW(A,B,C) != CCW(A,B,D)

def CCW(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Инициализация
def Init(args) : 
	# Загрузка модели
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# Инициализация цветов
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

	# Загрузка весов и конфигурации
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

	print("[ + ] Loading YOLO from disk...")
	
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	vs = cv2.VideoCapture(args["input"])

	return LABELS , COLORS , net, ln , vs

# Парсим видео 
def ParseVideo(args , LABELS , COLORS , net, ln , vs) : 
	tracker = Sort()
	memory = {}

	writer = None
	(W, H) = (None, None)
	
	frameIndex = 0

	# Счетчики машин 
	counter = 0
	counter2  = 0
	# 2 списка для графика
	sec_x = []
	sec_y_to = []
	sec_y_from = []
	frame_x = []
	frame_y_to = []
	frame_y_from = []

	try:
		if imutils.is_cv2() == True :
			prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
		else :
			prop = cv2.CAP_PROP_FRAME_COUNT
		
		total = int(vs.get(prop))
		print("[ + ] {} total frames in video".format(total))
	except:
		print("[ - ] Could not determine # of frames in video")
		total = -1
		return
	
	# Главный цикл обработки видео
	while True:

		(grabbed, frame) = vs.read()

		if not grabbed:
			break
		
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		
		
		start = time.time()
		layers = net.forward(ln)
		end = time.time()

		
		# Границы
		boundaries = []
		confidences = []

		for layer in layers :
			for detection in layer :
				
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > args["confidence"]:
					
					square = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = square.astype("int")

					# Опеределяем левый верхний угол
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boundaries.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))

		idxs = cv2.dnn.NMSBoxes(boundaries, confidences, args["confidence"], args["threshold"])
		objects = []
		
		if len(idxs) > 0 :
			for i in idxs.flatten():
				(x, y) = (boundaries[i][0], boundaries[i][1])
				(w, h) = (boundaries[i][2], boundaries[i][3])
				objects.append([x, y, x+w, y+h, confidences[i]])

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
			i = int(0)
			for square in boundaries:
				# Левый верхний угол
				(x, y) = (int(square[0]), int(square[1]))
				# Ширина + высота
				(w, h) = (int(square[2]), int(square[3]))

				
				color = [int(c) for c in COLORS[IDs[i] % len(COLORS)]]
				# Рисуем коробку вокруг обьекта
				cv2.rectangle(frame, (x, y), (w, h), color, 5)

				if IDs[i] in prev:
					previous_box = prev[IDs[i]]
					
					(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
					(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
					
					p0 = (int(x + (w - x) / 2) , int(y + (h - y) / 2))
					p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
					
					cv2.line(frame, p0, p1, color, 5)
					
					# Проверяем пересечения коробки с линиями
					if Intersect(p0, p1, line[0], line[1]):
						counter += 1
					if Intersect(p0, p1, line2[0], line2[1]):
						counter2 += 1 

				# Подписываем коробку
				cv2.putText(frame, "{}".format(IDs[i]) , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
				i += 1

		# draw line
		cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
		cv2.line(frame, line2[0], line2[1], (255, 0, 255), 5)

		# (x, y) - левый верхний угол начало координат
		# cv2.putText(frame, str(counter), (100,300), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
		# cv2.putText(frame, str(counter2), (3800,300), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
		
		# Тестовый - хороший
		cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
		cv2.putText(frame, str(counter2), (1000,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 0, 255), 10)
		
		
		# Сохраняем изображение - для отладки
		cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

		if writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
			if total > 0:
				elap = (end - start)
				print("[ + ] Single frame took {:.4f} seconds".format(elap))
				print("[ + ] Total time to finish: {:.4f}".format(elap * total))

		writer.write(frame)

		frameIndex += 1
		
		frame_x.append(frameIndex)
		frame_y_to.append(counter)
		frame_y_from.append(counter2)

		if frameIndex % 30 == 0 : # Для заданного значения fps
			sec_x.append(frameIndex // 30)
			sec_y_to.append(counter)
			sec_y_from.append(counter2)

		
		if frameIndex >= LIMIT:
			break

	# Чистим следы
	print("[ + ] Cleaning up...")
	
	writer.release()
	vs.release()
	
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

# Главная функция
def main() :

	# print(cv2.getBuildInformation())
	
	# 1. Чистим следы
	Clear()
	# 2. Парсим аргументы
	args = ParseArgs()
	# 3. Инициализируем параметры
	LABELS , COLORS , net, ln , vs = Init(args)
	# 4. Парсим видео
	sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from = ParseVideo(args , LABELS , COLORS , net, ln , vs)
	# 5. Строим график
	DrawGraf(sec_x , sec_y_to , sec_y_from , frame_x , frame_y_to , frame_y_from)

if __name__ == '__main__':
    main()