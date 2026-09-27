# IPC — история оптимизации Stage 1.11

## Цель

Проект: высокопроизводительная реализация iPC (incremental predictive coding) на TileLang/CUDA, специализированная под одну фиксированную видеокарту — NVIDIA GeForce RTX 3060 Laptop GPU, capability `(8, 6)`, 6 GB VRAM, класс мощности около 120 W.

Идея этапов 1.11–1.18: сначала сделать autotuning воспроизводимым и не тратить JIT/benchmark-бюджет на заведомо плохие конфигурации, затем реализовать межслойный параллелизм через `grid.z`, автоматически подобрать размер Z и наконец проверить снижение launch overhead через CUDA Graph.

---

## Stage 1.11 — базовый расширенный autotuning

На этом этапе был значительно расширен поиск конфигураций TileLang для P/I/W kernel families.

Ключевой результат end-to-end на RTX 3060 Laptop GPU:

| Конфигурация | samples/s | Температура | Power | SM |
|---|---:|---:|---:|---:|
| depth 3 baseline | 2.302 M | 68°C | 50.8 W | 1987 MHz |
| depth 3 tuned | **5.118 M** | 70°C | 64.4 W | 1972 MHz |
| depth 4 tuned | **4.363 M** | 71°C | 65.2 W | 1980 MHz |
| depth 6 tuned | **3.279 M** | 70°C | 62.1 W | 1980 MHz |

Сравнение со старыми baseline:

- depth 3: `5.118 / 2.302 = 2.22x`
- depth 4: `4.363 / 1.959 = 2.23x`
- depth 6: `3.279 / 1.518 = 2.16x`

Главный вывод: ускорение около **2.2x** сохраняется при увеличении глубины, то есть это не выигрыш одного shape.

Дополнительное наблюдение: tuned kernel использует примерно 64–65 W против ~50.8 W baseline, при близкой частоте SM. Это указывает на более высокую фактическую загрузку GPU.

---

## Legality-aware adaptive autotuner

Обнаружена проблема старого отбора: `max-configs=10` и `max-configs=100` могли выбирать разные подмножества из общего candidate pool, поэтому больший budget не гарантировал наличие предыдущих кандидатов.

Добавлено:

1. Раздельная static legality для prediction/inference/weight.
2. Предварительное исключение нелегальных `T.gemm` до JIT-компиляции.
3. Persistent cache измерений.
4. Deterministic initial exploration.
5. Adaptive selection по результатам уже измеренных конфигураций.
6. Кумулятивная семантика `max-configs`: увеличение бюджета добавляет новые измерения к уже имеющимся.

Static test:

```text
Stage 1.12 static autotune tests: PASS
global_pool=756 legal_prediction=612 legal_weight=500
```

GPU smoke:

```text
6/6 конфигураций измерены
BEST prediction M=64 K=64 B=128: 0.0204 ms
```

Повторный запуск того же smoke:

```text
cached=6
```

То есть измерения повторно не выполнялись.

Важно: `0.0204 ms` этого smoke не следует считать новым глобальным рекордом против прежних `0.0174 ms`: это небольшой deterministic/adaptive smoke budget, а не полный поиск.

---

## Первый `grid.z` layer batching

В iPC операции одинаковых внутренних слоёв были сгруппированы в один TileLang kernel с третьим измерением grid.

Принцип:

```text
depth 4: 10 -> 64 -> 64 -> 64 -> 784
                       ^     ^
                     общий grid.z
```

Для boundary-слоёв с другими матричными размерами отдельные kernel сохранились; padding и дополнительные копирования не вводились.

Correctness smoke:

```text
Stage 1.13 static grid.z tests: PASS
Stage 1.13 grid.z smoke: PASS groups=1 group_count=2 max_w=0 max_x=0 max_e=0
```

То есть результаты `grid.z` и legacy path совпали по проверяемым массивам/ошибкам.

---

## Первый A/B для `grid.z`

Первое измерение показало, что сам факт увеличения Z не гарантирует ускорение.

```text
depth=4 legacy=0.3493 ms gridz=0.2208 ms speedup=1.582x

depth=6 legacy=0.5281 ms gridz=0.5793 ms speedup=0.912x
```

Ключевой вывод: **размер Z является самостоятельным tuning-параметром**. `grid.z=4` для depth 6 оказался хуже legacy.

---

## Ограничение Z=2

Вместо безусловного объединения всех внутренних слоёв был введён ограничитель `grid_z_max_layers`.

На RTX 3060:

```text
depth=4 legacy=0.3528 ms gridz2=0.2718 ms speedup=1.298x groups=1 z=[2]

depth=6 legacy=0.5463 ms gridz2=0.3870 ms speedup=1.412x groups=2 z=[2, 2]
```

Таким образом, Z=2 дал положительный результат сразу на обеих глубинах.

Correctness:

```text
Stage 1.15 adaptive grid.z static tests: PASS
Stage 1.15 adaptive grid.z smoke: PASS group_count=2 max_w=0 max_x=0 max_e=0
```

---

## Autotuning размера Z

После Stage 1.15 вместо фиксированного Z было сделано маленькое отдельное пространство поиска:

```text
Z candidates = {0, 2, 3, 4}
```

где `0` — legacy/no-grid.z.

Результат настоящего GPU autotune:

```text
depth=4 cap=0 ms=0.3916

depth=4 cap=2 ms=0.2994

depth=4 cap=3 ms=0.2296
BEST depth=4: cap=3 ms=0.2296
```

и:

```text
depth=6 cap=0 ms=0.5394
depth=6 cap=2 ms=0.3787
depth=6 cap=3 ms=0.3936
depth=6 cap=4 ms=0.7574
BEST depth=6: cap=2 ms=0.3787
```

Итого получена shape/depth-specific policy:

```text
depth 4 -> Z=3
depth 6 -> Z=2
```

---

## Интеграция Z-policy в trainer

Найденная policy подключена непосредственно к training path.

Проверки:

```text
Stage 1.17 policy tests: PASS depth4=3 depth6=2 fallback=2
```

GPU activation:

```text
depth=4 selected_cap=3 groups=((1, 3, 3),)
depth=6 selected_cap=2 groups=((1, 2, 2), (4, 5, 2))
Stage 1.17 GPU policy activation: PASS
```

То есть autotuning выполняется отдельно, а обычный training path получает уже готовую policy и не запускает поиск во время обучения.

---

## A/B CUDA Graph

Последний этап проверял уже не изменение вычислений, а снижение overhead множества CUDA kernel launches через CUDA Graph.

На RTX 3060 Laptop GPU:

```text
depth=4 cap=3 groups=((1, 3, 3),)
direct=0.2246 ms
(0.5699 M/s)
graph=0.0612 ms
(2.0918 M/s)
speedup=3.671x
```

```text
depth=6 cap=2 groups=((1, 2, 2), (4, 5, 2))
direct=0.3777 ms
(0.3389 M/s)
graph=0.0831 ms
(1.5397 M/s)
speedup=4.544x
```

Это очень большой выигрыш в данном benchmark именно на уровне repeated execution/launch overhead.

Однако этот коэффициент **нельзя просто умножать** на прежние 2.2x и на `grid.z` 1.3–1.4x: это измерения разных уровней системы и разных benchmark harness. Для итоговой оценки нужен единый end-to-end training benchmark с одинаковым числом iPC steps.
