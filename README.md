# CarDetection
Определение количества автомобилей на видеозаписи с камеры наблюдения с использованием Python 3.6, OpenCV, YOLO , CUDA и Tensorflow-GPU.

# CPU
![ExampleCPU.gif](Example.gif)

Для запуска: 

1. Скачать код на компьютер и перейти в папку CPU.
2. Создать папку input и поместить в нее видео с камеры наблюдения.
3. Создать папку ouput (там будет результат).
4. Загрузить [веса](https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) и поместить их в папку parameters.
5. Запустить программу : 
``` cmd 
python main.py --input input\Example.mp4 --output output\Example_CarDetection.mp4 --yolo parameters
```

# GPU
![ExampleGPU.gif](ExampleGPU.gif)

Для запуска:

0. Установить все нужные программы по [гайду](https://medium.com/@lmoroney_40129/installing-tensorflow-with-gpu-on-windows-10-3309fec55a00).
1. Скачать код на компьютер и перейти в папку GPU.
2. Создать папку input и поместить в нее видео с камеры наблюдения.
3. Создать папку ouput (там будет результат).
4. Скачать веса и недостающие [параметры](https://dropmefiles.com/WkhpZ).  
6. Запустить программу : 
>python main.py --input input\ExampleGPU.mp4 --output output\ExampleGPU.mp4

# Credits
При реализации использовалась информация из следующих источников :

1. [YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) 
2. [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)
3. [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
4. [pjreddie/darknet](https://github.com/pjreddie/darknet)
5. [dmlc/gluon-cv](https://github.com/dmlc/gluon-cv)
6. [wizyoung/YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)
7. [abewley/sort](https://github.com/abewley/sort)
