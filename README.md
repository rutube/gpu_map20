GPU_MAP@20
==========

Расчитывает метрику Mean Average Precision @ 20 для размеченной поисковой выдачи.

Краткое описание
----------------
Дано: поисковый движок (к примеру, [sphinxsearch](sphinxsearch.com)), ранжирующий поисковую выдачу с учетом 
определенного ранкера, имеющего набор параметров. Поисковая выдача отображается на сайте и по ней собирается
статистика "релевантности". К примеру, видео на сайте помечается релевантным, если пользователь после перехода 
со страницы поиска посмотрел рекламу и 10 секунд видео.

Задача: определить оптимальные параметры поискового ранкера, максимизирующие метрику MAP@20. Эта метрика хорошо 
описана в статье [Метрики качества ранжирования](https://habrahabr.ru/company/econtenta/blog/303458/).
Нагружать поисковый движок для перебора всех возможных коэффициентов - невыносимо долго, необходимо сделать симулятор
поискового движка на видеокарте, с целью собрать для разных наборов параметров соответствующие им значения поисковой
метрики.

Решение
-------

В общем виде формулу веса документа по запросу можно представить как
```
F(query, document) + G(document).
```
где `query` - поисковый запрос, `document` - один из документов, присутствующих в поиске, `F` - фунция, определяющая,
насколько хорошо конкретный документ соответствует поисковому запросу, `G` - фунция получения "бустинга" для конкретного
документа, независимо от запроса.

Конкретный вид фунции F зависит от специфики каждого сайта, но, к примеру, может выглядеть как в 
[документации](http://sphinxsearch.com/docs/current.html#expression-ranker)

```
SELECT *, ... AS G, WEIGHT() + G AS my_wegiht FROM myindex WHERE MATCH('hello world')
ORDER BY my_weight DESC
OPTION ranker=expr('sum(lcs*user_weight)*1000 + bm25')
```

`user_weight, 1000` - это параметры ранкера (веса конкретных текстовых полей, умноженные на 1000). 
`lcs` и `bm25` - характеристики пары `(query, document)`, не зависящие от параметров ранкера.
Функция `G` специфична для каждого конкретного сайта, и, например, может выглядить так

```
SELECT 1000 * author_is_vip FROM myindex
```
добавляя +1000 к документам VIP-авторов..

Таким образом, прогнав все собранные запросы по поисковому движку и собрав всю выдачу, мы можем для пары
 `(query, document)` создать вектор числовых признаков, умножением которого на вектор параметров ранкера,
 можно получить набор весов всех документов по запросу.
Далее этот набор весов сортируется по убыванию, склеивается с информацией о релевантности конкретного документа в 
выдаче и для топ-20 выполняется вычисление метрики Average Precision.

Получив для всего списка запросов значения Average Precision и взяв среднее арифметическое, получаем 
**Mean Average Precision**.
  
Вся эта арифметика хорошо ложится на GPU и позволяет быстро рассчитывать `MAP@20` для большого (сотни тысяч вариантов)
множества различных векторов параметров ранкера, либо использовать градиентный спуск с частичной производной.

Использование
-------------

Пример рассчета AveragePrecision для одного запроса и указанного набора вариантов весов

```
gpu_map20 --factors=48 matrix.bin relevance.bin weights.bin
```
* `factors` - число признаков для каждого документа
* `matrix.bin` - матрица признаков для документов `float32` размера `rows * factors` по строкам 
(первые `4*factors` байт содержат признаки первого документа выдачи)
* `relevance.bin` - вектор релевантности документов выдачи, `float32` со значениями `0, 1` размера `rows`.
* `weights.bin` - матрица вариантов весов `float32` размера `variants * factors` по строкам
(первые `4*factors` байт содержат набор весов первого варианта)

В файл `map_20.bin` выводится вектор значений AveragePrecision для всех `variants` вариантов ранкера.

Сборка
------

Системные зависимости:
* nvidia-cuda-toolkit

Проект разрабатывается в IDE Clion. Сборка вручную:

```
cd $PROJECT_ROOT
# нужен cmake>=3.6
cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 `pwd`
make
```
