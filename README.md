# CarDetection
Определение количества автомобилей на видео записи с камеры наблюдения.

Использовался Python 3.6.

![Example.gif](Example.gif)

Для запуска: 

[ + ] Скачать код на компьютер.
[ + ] Создать папку input и поместить в нее видео с камеры наблюдения.
[ + ] Создать папку ouput (там будет результат).
[ + ] Загрузить веса (https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) и поместить их в папку parameters.
[ + ] Запустить программу : 
python main.py --input input\Example.mp4 --output output\Example_CarDetection.mp4 --yolo parameters