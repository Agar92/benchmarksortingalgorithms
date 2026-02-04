# BenchmarkSortingAlgorithms
Для компиляции кода на CPU использовать:

`$g++ -O3 -std=c++17 -pthread -mcmodel=large -fopenmp benchmark.cpp -o benchmark` :v:

Для использования на кластере VKPP компилятора nvcc необходимо загрузить этот модуль:

`[VKPP:70-gaa@access-3 ~]$module load nvidia_hpc_sdk/22.7`

Для компиляции кода cuda_sort1.cu на GPU NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0) использовать:

`$nvcc -arch=compute_70 -code=sm_70 -O3  cuda_sort1.cu -o cuda_sort1` :v:

Для компиляции кода cuda_sort2_thrust_sort.cu на GPU NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0) использовать:

`$nvcc -arch=compute_70 -code=sm_70 -O0 -extended-lambda cuda_sort2_thrust_sort.cu -o cuda_sort2_thrust_sort` :v:

Для компиляции кода cuda_sort3_mysort_using_thrust.cu на GPU NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0) использовать:

`$nvcc -arch=compute_70 -code=sm_70 -O0 -extended-lambda cuda_sort3_mysort_using_thrust.cu -o cuda_sort3_mysort_using_thrust` :v:

Для компиляции кода cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu на GPU NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0) использовать:

`nvcc -arch=compute_70 -code=sm_70 -O3  --use_fast_math --expt-extended-lambda cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu -o cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU` :v:

Для компиляции кода cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu на 64-ядерном СPU Intel Xeon Gold 6242 (2.8GHz) компилятором nvcc использовать:

`nvcc -std=c++17 -O3 --extended-lambda -Xcompiler -fopenmp cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu -o cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_bin___on_CPU` :v:

Для компиляции кода test_nvc++_stdpar_gpu_sorting.cpp на 64-ядерном СPU Intel Xeon Gold 6242 (2.8GHz) для запуска на GPU NVIDIA Tesla V100-SXM2-32GB компилятором nvc++ NVIDIA HPC SDK использовать:

`nvc++ -stdpar=gpu -O3 test_nvc++_stdpar_gpu_sorting.cpp -o test_nvc++_stdpar_gpu_sorting` :v:

Для компиляции кода cuda_cub_radix_sort_pairs.cu для запуска на GPU NVIDIA Tesla V100-SXM2-32GB компилятором nvcc использовать:

`nvcc -arch=sm_70 -O0 cuda_cub_radix_sort_pairs.cu -o cuda_cub_radix_sort_pairs` :v:

Опция `-extended-lambda` нужна для того, чтобы лямбда-функции были доступны внутри `thrust::sort`.

Как узнать compute capability GPU NVIDIA?

В интернете по модели GPU, которую можно узнать, вызвав `$nvidia-smi` или `$nvtop`, или же скомпилировав и запустив тестовую программу:

`$nvcc get_gpu_compute_capability.cu -o get_gpu_compute_capability; ./get_gpu_compute_capability`

В _benchmark.cpp_ в строке 127 необходимо выбрать алгоритм сортировки:
-   const int SORTING_ALGO=0;
    сортировка std::sort из стандартной библиотеки, вызывается в строках 154-158
-   const int SORTING_ALGO=1;
    многопоточная сортировка mysort_Nthreads(...), описана в строках 57-120 benchmark.cpp.
    Число потоков Nbin=8 (см. globals.h строка 3)
-   const int SORTING_ALGO=2;
    однопоточная последовательная сортировка mysort, описанная в файле seq_sort.h, строки 20-86
-   const int SORTING_ALGO=3;
    многопоточная сортировка TPT3_sort как она реализована в TPT3, описана в строках 59-219 файла extra.h
    

Все варианты сортировки не только сортируют исходный массив частиц в порядке убывания по _ir_, но и сохраняют порядок идентификаторов частиц _id_, как они были до сортировки.

**Текущие результаты сравнения производительности на 4-ядерном CPU Intel Core i7-3770 (если платформа другая, это указывается явно):**

Размер задачи: _globals.h_ строка 3

```
const int LIFE=20'000'000;
const int Nbin=8;
```

**20 миллионов частиц** :fearful:

Величины (x...) после времени счёта показывают примерное замедление данного алгоритма сортировки относительно самого быстрого варианта сортировки, представленного в самом правом столбце таблицы.  

8 потоков :star2:

1) С опцией компилятора GCC/NVCC "-O0" время работы программы:

| | radixsort.cpp | std::sort<br> с сохранением порядка _id_<br> SORTING_ALGO=0 | std::sort<br> без сохранения порядка _id_<br> (закомментировали<br> условие сравнения _id_ в строках 156-157)<br> SORTING_ALGO=0 | mysort из _seq_sort.h_<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=2 | TPT3_sort из _extra.h_<br> по умолчанию<br> $`\textcolor{red}{\text{не}}`$ сохраняет порядок _id_<br> SORTING_ALGO=3 $`\textcolor{red}{\text{Nbin=8}}`$ | TPT3_sort из _extra.h_<br> по умолчанию<br> $`\textcolor{red}{\text{не}}`$ сохраняет порядок _id_<br> SORTING_ALGO=3 $`\textcolor{red}{\text{Nbin=2048}}`$ | mysort_Nthreads<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=1 Число потоков $`\textcolor{red}{\text{Nbin=8}}`$ | mysort_Nthreads<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=1 Число потоков $`\textcolor{red}{\text{Nbin=64}}`$ | cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp: ParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=512}}`$ | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp: ParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=512}}`$ | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp: OpenMPParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=72}}`$ | cuda_sort1.cu: fun Число потоков $`\textcolor{red}{\text{Nbin=256}}`$ (больше не позволяет shared memory of GPU) | cuda_sort2_thrust_sort.cu: thrust::sort | cuda_sort3_mysort_using_thrust.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=8192}}`$ | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=256}}`$ | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=4096}}`$ | test_cpu_boost_spreadsort_sorting_algorithm.cpp: тестирование алгоритма Boost boost::sort::spreadsort::integer_sort | test_nvc++_stdpar_gpu_sorting.cpp: std::sort(std::execution::par, particles.begin(), particles.end()) | cuda_cub_radix_sort_pairs.cu: поразрядная сортировка |
| ------ | ------ |  ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| t | 2.7 с (x100) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 9.8 с)  | 7.9 с ($`\textcolor{red}{\text{x293}}`$) | 5.7 с ($`\textcolor{blue}{\text{x211}}`$) | 650 мс ($`\textcolor{green}{\text{x24.1}}`$) | 600 мс ($`\textcolor{magenta}{\text{x22.2}}`$) | 410 мс ($`\textcolor{magenta}{\text{x15.2}}`$) | 320 мс (x11.9) | 150 мс ($`\textcolor{gray}{\text{x5.6}}`$) | 1842 мс (x68.2) | 1661 мс (x61.5) | 251 мс ($`\textcolor{brown}{\text{x9.3}}`$) | ~964 мс (x35.7) | 91 мс ($`\textcolor{orange}{\text{x3.4}}`$) | 41 мс (x1.5) | 208 мс (x7.7) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 22.5 с) | 4.4 с (x163) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 24.6 с) | 1.2 с (x44.4) (в то же время сортировка std::sort на 72-ядерном CPU Intel Xeon Gold-2697 работает за 9.4 с) | 340 мс (x15.3) (в то же время сортировка std::sort на 72-ядерном CPU Intel Xeon Gold 6242 работает за 9.9 с) | собственно сортировка 27 мс; копирование данных H2D 430 мс | 
| файл | radixsort.cpp | _benchmark.cpp_ | _benchmark.cpp_ | _seq_sort.h_ | _extra.h_ | _extra.h_ | _benchmark.cpp_ | _benchmark.cpp_ | cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp | cuda_sort1.cu | cuda_sort2_thrust_sort.cu | cuda_sort3_mysort_using_thrust.cu | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu | test_cpu_boost_spreadsort_sorting_algorithm.cpp | test_nvc++_stdpar_gpu_sorting.cpp | cuda_cub_radix_sort_pairs.cu |
| платформа | 64-ядерный CPU Intel Xeon Gold 6242 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | 72-ядерный Intel Xeon E5-2697 | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | 64-ядерный CPU Intel Xeon Gold 6242 | 72-ядерный CPU Intel Xeon E5-2697 | 64-ядерный CPU Intel Xeon Gold 6242 + NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB |
| Число потоков Nbin | нет | 8 | 8 | 8 | 8 | 2048 | 8 | 64 | 512 | 512 | 72 | 256 | нет (thrust сам внутри как-то распараллеливает код) | 8192 | 256 | 4096 | нет (Boost сам внутри как-то распараллеливает код) | нет (NVIDIA HPC SDK сам внутри как-то распараллеливает код) | нет |

Разница в производительности составляет примерно $`\textcolor{red}{\text{293}}`$ , $`\textcolor{blue}{\text{211}}`$, $`\textcolor{green}{\text{24.1}}`$, $`\textcolor{magenta}{\text{15.2}}`$, $`\textcolor{gray}{\text{5.6}}`$, $`\textcolor{brown}{\text{9.3}}`$ и $`\textcolor{orange}{\text{3.4}}`$ раз.

2) С опцией компилятора GCC/NVCC "-O3" время работы программы:

| | radixsort.cpp | std::sort<br> с сохранением порядка _id_<br> SORTING_ALGO=0 | std::sort<br> без сохранения порядка _id_<br> (закомментировали<br> условие сравнения _id_ в строках 156-157)<br> SORTING_ALGO=0 | mysort из _seq_sort.h_<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=2 | TPT3_sort из _extra.h_<br> по умолчанию<br> $`\textcolor{red}{\text{не}}`$ сохраняет порядок _id_<br> SORTING_ALGO=3 $`\textcolor{red}{\text{Nbin=8}}`$ | TPT3_sort из _extra.h_<br> по умолчанию<br> $`\textcolor{red}{\text{не}}`$ сохраняет порядок _id_<br> SORTING_ALGO=3 $`\textcolor{red}{\text{Nbin=2048}}`$ | mysort_Nthreads<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=1 Число потоков $`\textcolor{red}{\text{Nbin=8}}`$ | mysort_Nthreads<br> по умолчанию<br> сохраняет порядок _id_<br> SORTING_ALGO=1 Число потоков $`\textcolor{red}{\text{Nbin=64}}`$ | cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp: ParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=512}}`$ | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp: ParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=512}}`$ | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp: OpenMPParallelSorter <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=72}}`$ | cuda_sort1.cu: fun Число потоков $`\textcolor{red}{\text{Nbin=256}}`$ (больше не позволяет shared memory of GPU) | cuda_sort2_thrust_sort.cu: thrust::sort | cuda_sort3_mysort_using_thrust.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=8192}}`$ | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=256}}`$ | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu: mysort_Nthreads <br> по умолчанию<br> сохраняет порядок _id_<br> Число потоков $`\textcolor{red}{\text{Nbin=4096}}`$ | test_cpu_boost_spreadsort_sorting_algorithm.cpp: тестирование алгоритма Boost boost::sort::spreadsort::integer_sort | test_nvc++_stdpar_gpu_sorting.cpp: std::sort(std::execution::par, particles.begin(), particles.end()) | cuda_cub_radix_sort_pairs.cu: поразрядная сортировка |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| t | 2.5 с (x104) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 1.4 с) | 2.1 с ($`\textcolor{red}{\text{x87.5}}`$) | 1040 мс ($`\textcolor{blue}{\text{x43.3}}`$) | 494 мс ($`\textcolor{green}{\text{x20.6}}`$) | 560 мс ($`\textcolor{magenta}{\text{x23.3}}`$) | 365 мс ($`\textcolor{magenta}{\text{x15.2}}`$) | 245 мс (x10.2) | 146 мс ($`\textcolor{gray}{\text{x6.1}}`$) | 439 мс (x18.3) | 259 мс (x10.8) | 211 мс ($`\textcolor{brown}{\text{x8.8}}`$) | ~964 мс (x40.2) | 93 мс ($`\textcolor{orange}{\text{x3.9}}`$) | 41 мс (x1.7) | 183 мс (x7.6) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 1.3 с) | 421 мс (x17.5) (в то же время сортировка std::sort на 64-ядерном CPU Intel Xeon Gold 6242 работает за 1.4 с) | 330 мс (x13.8) (в то же время сортировка std::sort на 72-ядерном CPU Intel Xeon Gold-2697 работает за 1.5 с) | 340 мс (в то же время сортировка std::sort на 64-ядерном CPU Intel Gold 6242 работает за 1.4 с) | собственно сортировка 24 мс; копирование данных H2D 430 мс | 
| файл | radixsort.cpp | _benchmark.cpp_ | _benchmark.cpp_ | _seq_sort.h_ | _extra.h_ | _extra.h_ | _benchmark.cpp_ | _benchmark.cpp_ | cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp | cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp | cuda_sort1.cu | cuda_sort2_thrust_sort.cu | cuda_sort3_mysort_using_thrust.cu | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu | cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu | test_cpu_boost_spreadsort_sorting_algorithm.cpp | test_nvc++_stdpar_gpu_sorting.cpp | cuda_cub_radix_sort_pairs.cu |
| платформа | 64-ядерный CPU Intel Xeon Gold 6242 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | Intel Core i7-3770 | 72-ядерный Intel Xeon E5-2697 | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | NVIDIA Tesla V100-SXM2-32GB | 64-ядерный CPU Intel Xeon Gold 6242 | 72-ядерный CPU Intel Xeon E5-2697 | 64-ядерный CPU Intel Xeon Gold 6242 + NVIDIA Tesla V100-SXM2-32GB | V100-SXM2-32GB |
| Число потоков Nbin | нет | 8 | 8 | 8 | 8 | 2048 | 8 | 64 | 512 | 512 | 72 | 256 | нет (thrust сам внутри как-то распараллеливает код) | 8192 | 256 | 4096 | нет (Boost сам внутри как-то распараллеливает код) | нет (NVIDIA HPC SDK сам внутри как-то распараллеливает код) | нет |

Разница в производительности составляет примерно $`\textcolor{red}{\text{87.5}}`$, $`\textcolor{blue}{\text{43.3}}`$, $`\textcolor{green}{\text{20.6}}`$, $`\textcolor{magenta}{\text{15.2}}`$, $`\textcolor{gray}{\text{6.1}}`$, $`\textcolor{brown}{\text{8.8}}`$  и $`\textcolor{orange}{\text{3.9}}`$ раз.


_Дополнительная информация:_

В файле _extra.h_ полностью приводится сортировка, как она была сделана в TPT3.

В файле _seq_sort.h_ приводится последовательный вариант сортировки _mysort_, который всего чуть более, чем в 2 раза медленнее, чем _mysort_Nthreads_, и в 3-4 раза быстрее _std::sort_. :zap:


**ЗАМЕЧАНИЯ**

1.
В _cuda_sort1.cu_ пробовал увеличить величину Nbin, но при увеличении Nbin>256, компилятор nvcc выдаёт ошибку, что превышено максимально возможный размер shared memory на блок CUDA. Пробовал уйти от использования быстрой, но ограниченной shared memory, путём заказа необходимых массивов в глобальной памяти GPU при помощи cudaMalloc() в функции main() (результат - в файле cuda_sort1_no_shared_memory.cu). Но это не привело к ускорению производительности. Более того, при Nbin>512 (если правильно помню) на кластере на GPU NVIDIA Tesla V100-SXM2-32GB сортировка ломается (выходные результаты не совпадают с результатами сортировки на CPU). При том, что на GPU NVIDIA GeForce GTX 650 Ti ПК c i7-3770 тот же код работает исправно при любых значениях Nbin. Причина этого сбоя на GPU NVIDIA Tesla V100-SXM2-32GB кластера VKPP непонятна, и как искать источник ошибки на GPU NVIDIA Tesla V100-SXM2-32GB кластера VKPP, тоже совершенно непонятно.

2.
Отличие файлов cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp и cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp друг от друга состоит в том, что в первом из них для каждого бина/ящика используется промежуточный буфер/массив для хранения результатов сортировки частиц в этом бине, откуда они должным образом с соответствующими отступами потом копируются в выходной/результирующий массив. Во втором файле частицы каждого бина сортируются прямо в исходном/входном массиве, откуда сразу же с соответствующими отступами копируются в выходной/результирующий массив без использования промежуточных буферов/массивов. Поэтому сортировка во втором файле должна работать быстрее.

3.
ParallelSorter из файлов cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp и cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp сортирует частицы по бинам и затем копирует их с соответствующими отступами в выходной/результирующий массив при использовании стандартной параллельности C++, реализованной при помощи std::thread. Число бинов может выбираться произвольным и никак не связано с числом ядер процессора, на котором производятся расчёты. Более того, оно может быть гораздо больше числа ядер процессора, и иногда это приводит к увеличению производительности расчётов.

4.
OpenMPParallelSorter из файлов cpu_multithreaded_sorting_with_copying_to_temporary_arrays.cpp и cpu_multithreaded_sorting_in_place_without_copying_to_temporary_arrays.cpp сортирует частицы по бинам и затем копирует их с соответствующими отступами в выходной/результирующий массив при использовании библиотеки OpenMP. Так что один бин/ящик соответствует одному потоку OpenMP. В программе выбор числа потоков реализован как
`num_threads = omp_get_num_threads();`
т. е. определяется числом ядер процессора, на котором производятся расчёты.

5.
Почему написано ~964 мс ?

Потому, что на GPU NVIDIA GeForce GTX 650 Ti ПК c CPU Intel Core i7-3770 и GPU NVIDIA Tesla V100-SXM2-32GB кластера VKPP компилятор nvcc не позволяет задать N>120000 (т. е. N>120K). Возникает ошибка компиляции relocation truncated to fit. Эта ошибка на CPU бывает, когда задаются слишком большие размеры массивов (в первую очередь, статических). При этом объём памяти, выделяемый для хранения массивов частиц `particles, arr_minus1, arr0, arr1, arr2, arr3` в файле _cuda_sort1.cu_ не превышает 6-7 ГБ. В GPU NVIDIA Quadro P5000 (кластер VKPP раздел vis) объём встроенной памяти составляет 16 ГБ, а в GPU NVIDIA Tesla V100-SXM2-32GB (кластер VKPP раздел gpu) - 32 ГБ.
Таким образом, время ~964 мс было найдено как 20'000'000/120'000 * 5.8 мс = 167 * 5.8 мс ~ 964 мс. Т. е. код файла _cuda_sort1.cu_ для задачи размером N=120'000 на GPU NVIDIA Tesla V100-SXM2-32GB считается примерно 5.8 мс.

6.
Почему код для GPU не запускался на самых современных GPU NVIDIA A100?

Потому, что раньше GPU NVIDIA A100 располагались в разделе biggpu кластера VKPP. Сейчас его нету, его судя по всему убрали, и я не знаю
 как получить доступ к GPU NVIDIA A100 на серверах ВНИИА.

7. 
В файлах cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_GPU.cu / cuda_sort4_thrust_Nbins_standard_sort_in_each_bin___on_CPU.cu реализован следующий алгоритм сортировки. Весь массив частиц делился на Nbin бинов, в каждом из которых использовалась сортировка thrust::sort / std::sort. После этого параллельно по всем бинам, частицы с одним и тем же ir при помощи memcpy со соответствующими отступами копировались в выходной массив частиц. И на CPU, и на GPU алгоритм сортировки был реализован при помощи библиотеки CUDA Thrust.

8.
В файле test_nvc++_stdpar_gpu_sorting.cpp представлен результат запуска сортировки структур на GPU при помощи стандартной параллельности NVIDIA HPC SDK stdpar. Результаты по производительности выглядят достаточно низкими. Полагаем, что это связано с тем, что при помощи CUDA Managed Memory в это время включено время копирования данных H2D и обратно D2H, которое исходя из полученных нами результатов занимает б`ольшую часть времени всего процесса сортировки и перемещения данных. Если бы была возможность при использовании стандартной параллельности С++ и модели NVIDIA stdpar узнать время копирования данных H2D и обратно D2H, то, полагаем, мы увидели бы очень хороший результат производительности алгоритма сортировки, в разы меньше, чем полученное сейчас значение.

9.
radixsort.cpp - поразрядная сотрировка, неоптимизированная. Это вариант реализации алгоритма поразрядной сортировки работает медленно.

10.
Самый производительный вариант сортировки - cuda_cub_radix_sort_pairs.cu, использующий CUDA и реализацию алгоритма поразрядной сортировки библиотеки cub и работающий на GPU. Его производительность - ~ 25 мс на 20'000'000 частиц. Это при том, что используется синхронизация при помощи cudaDeviceSynchronize(); в строке 120. Если отключить синхронизацию, то алгоритм работает примерно ~ 50 раз быстрее. Хотя этого делать и не рекомендуется, и при работе реального программного кода это может приводить к ошибкам. Но здесь не приводит. И результат вычислений на GPU и на CPU совпадает при копировании данных на GPU и обратно даже без явной синхронизации при помощи cudaDeviceSynchronize().


**_ВЫВОД_**

Лидер производительности - cuda_cub_radix_sort_pairs.cu, использующий CUDA и реализацию алгоритма поразрядной сортировки библиотеки cub и работающий на GPU. Его производительность - ~ 25 мс на 20'000'000 частиц. Это при том, что используется синхронизация при помощи cudaDeviceSynchronize(); в строке 120. Если отключить синхронизацию, то алгоритм работает примерно ~ 50 раз быстрее. Хотя этого делать и не рекомендуется, и при работе реального программного кода это может приводить к ошибкам. Но здесь не приводит. И результат вычислений на GPU и на CPU совпадает при копировании данных на GPU и обратно даже без явной синхронизации при помощи cudaDeviceSynchronize(). В таком случае производительность этого программного кода дополнительно увеличится примерно в ~ 50 раз.


