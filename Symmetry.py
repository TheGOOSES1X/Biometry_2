from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from array import *

# Нарисовать изображения для обнаруженных объектов
def draw_image_with_boxes(filename, result_list):
    # Загрузка изображения
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    # Создание прямоугольников
    for result in result_list:
        # Получение координат
        x, y, width, height = result['box']
        # Создание формы
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # Рисование прямоугольника
        ax.add_patch(rect)
        # Массив для расчитывания расстояния между линиями симметрии
        array = []
        # Рисование точек
        for key, value in result['keypoints'].items():
            # Создание точек
            dot = Circle(value, radius=1, color='red')
            ax.add_patch(dot)
            # Добавление x координаты в массив
            a = int(value[0])
            array.append(a)
            if key in ['left_eye', 'right_eye', 'nose']:
               line = Rectangle((value[0], y), 0, height, fill=False, color='red')
               ax.add_patch(line)
        # Вычисление и вывод расстояния между линиями симметрии
        print(f"Расстояние от линии симметрии до линии левого/правого глаза: {array[2]-array[0]}/ {array[1]-array[2]}")
    pyplot.show()


def main():
    filename = 'People.jpg'
    #  Загрузка изображения
    pixels = pyplot.imread(filename)
    # Создание детектора, со стандартными весами
    detector = MTCNN()
    # Детекция лиц на изображении
    faces = detector.detect_faces(pixels)
    # Отображение лиц на оригинальном изображении
    draw_image_with_boxes(filename, faces)


main()


