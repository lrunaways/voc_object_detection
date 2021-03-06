# voc_object_detection

Задачи:
1) Обучить two-stage детектор, который будет использовать RoI Pooling для больших объектов и RoI Align для маленьких
2) Решить, какие объекты считать большими, а какие маленькими
3) Выбрать метрику
4) Сравнить метрики для больших и малых объектов

---
Колаб с базовым EDA: 
</br>https://colab.research.google.com/drive/174LtcsBTerkeja3A0ZjnQhJ-BluVGV4q
</br>
</br>Логи обучения и что всё запускается и обучается можно посмотреть в колабе:
</br>https://colab.research.google.com/drive/1oLNcKmG_5z0mXeYrZlDIWRsTn7DmbN26?usp=sharing
</br>
</br>

1) Взял стандартную имплементацию Faster R-CNN в torchvision с предтренированным VGG16 и заменил пулинг слой на кастомный.
   Для трейна взял 500 примеров, а для валидации 200 и обучался условно, на 5 эпох.
   </br>
   </br>Скрипт для запуска обучения в voc_object_detection/train.py
   </br>Создание модели в voc_object_detection/models/faster_rcnn.py
   </br>Код для пулинг слоя в voc_object_detection/models/layers/AreaDependentRoIPooling.py
   
2) Объекты площадью <= 32х32 буду считать маленькими, а >= 96х96 - большими. Сделал так, чтобы на выходе vgg16 маленькие объекты занимали площадь <= 1 пикселя, а большие >= 3. 

3) Чтобы учитывать как FP, так и FN, попробовал считать F1 для IoU(0.5:0.95:0.05) для скоров больше 0.0
   </br>Код для вычисления метрики в voc_object_detection/metrics/f1.py
   </br>Код для вычисления TP, FP, FN в voc_object_detection/metrics/calc_errors.py
   
4) F1 на маленьких объектах заметно хуже, чем на больших 
   
### Выводы:
  Маленькие объекты изначально несут меньше информации, чем большие. При прохождении через сеть, эта информация еще и теряется. 
  Кроме того, одинаковая ошибка в определении ббокса на больших и маленьких объектах неравнозначна, хотя базовый лосс штрафует одинаково.
  В итоге, распознавать маленькие объекты сложнее, чем большие, что видно по метрикам обучения.
