# Итоговый отчёт iPC / TileLang — Stage 1.11–1.18

## Конфигурация

GPU: **NVIDIA GeForce RTX 3060 Laptop GPU**, capability `(8, 6)`, 6 GB VRAM, класс TGP около 120 W.

Проект: собственная TileLang/CUDA реализация iPC с отдельными kernel families для prediction, inference и weight update.

---

## 1. Главный подтверждённый результат

Самый важный end-to-end результат был получен ещё на Stage 1.11:

| Depth | Baseline | Tuned | Ускорение |
|---:|---:|---:|---:|
| 3 | 2.302 M samples/s | **5.118 M samples/s** | **2.22x** |
| 4 | 1.959 M samples/s | **4.363 M samples/s** | **2.23x** |
| 6 | 1.518 M samples/s | **3.279 M samples/s** | **2.16x** |

Таким образом, оптимизация не исчезает при росте глубины. Полученный выигрыш устойчив на depth 3/4/6.

Профиль GPU также изменился в ожидаемую сторону: baseline был около 50.8 W, tuned — около 64–65 W, при температуре порядка 70–71°C. Это согласуется с более эффективной загрузкой вычислительных ресурсов GPU.

---

## 2. Autotuner стал пригодным для дальнейшей оптимизации

Stage 1.12 закрыл две практические проблемы.

### Static legality

В общем candidate pool было:

```text
756
```

Для prediction:

```text
612 legal
```

Для weight:

```text
500 legal
```

Нелегальные `T.gemm` конфигурации теперь отбрасываются до JIT.

### Adaptive search

Ранее разные значения `max-configs` могли выбирать разные подмножества одного общего пула. Это разрушало ожидаемую монотонность бюджета.

После изменения cache является накопительным: уже измеренные конфигурации повторно не benchmark'ятся, а новый budget добавляет новые конфигурации.

Проверка второго запуска smoke дала:

```text
cached=6
```

что подтверждает работоспособность persistent cache.

---

## 3. `grid.z` действительно дал межслойный выигрыш

Первая проверка дала:

```text
depth=4: 1.582x
```

но:

```text
depth=6: 0.912x
```

То есть объединять произвольное количество слоёв нельзя.

После ограничения группы до Z=2:

```text
depth=4: 1.298x
depth=6: 1.412x
```

И после отдельного Z-autotune была найдена более точная политика:

```text
depth=4 -> Z=3
    0.2296 ms

depth=6 -> Z=2
    0.3787 ms
```

Это подтверждает, что размер `grid.z` должен быть отдельным hardware-specific tuning parameter.

---

## 4. Policy теперь встроена в training path

Stage 1.17 подтвердил:

```text
depth=4 -> selected_cap=3
depth=6 -> selected_cap=2
```

и correctness/activation tests прошли.

Следовательно, autotuning не должен запускаться на каждом training run: training использует заранее выбранную policy.

---

## 5. CUDA Graph показал ещё один большой резерв

Stage 1.18 дал:

### Depth 4

```text
direct = 0.2246 ms
 graph = 0.0612 ms
speedup = 3.671x
```

### Depth 6

```text
direct = 0.3777 ms
 graph = 0.0831 ms
speedup = 4.544x
```

Это сильное свидетельство того, что после kernel optimization существенной частью оставшегося overhead становится стоимость повторного запуска множества GPU kernels / Python-side orchestration, когда workload фиксирован.

Но это именно результат Stage 1.18 A/B benchmark. Его нельзя арифметически умножать на `2.2x` Stage 1.11 или на `grid.z` speedup, потому что benchmark уровни различны.

---

# Итоговая архитектура оптимизации

```text
Fixed hardware: RTX 3060 Laptop GPU
            │
            ▼
P/I/W-specific TileLang kernels
            │
            ▼
Static legality filter
            │
            ▼
Adaptive cached autotuning
            │
            ▼
Depth/shape-specific grid.z policy
            │
            ▼
CUDA Graph for repeated fixed execution
            │
            ▼
Высокооптимизированный iPC execution path
```

---

# Что считать доказанным на этом checkpoint

1. Получен устойчивый примерно **2.2x end-to-end speedup** относительно исходного baseline на depth 3/4/6.
2. Autotuner умеет заранее исключать нелегальные конфигурации.
3. Autotuner стал накопительным и использует persistent cache.
4. `grid.z` реально ускоряет внутренние layer groups.
5. Оптимальный размер `grid.z` зависит от глубины: depth 4 → 3, depth 6 → 2.
6. Policy автоматически подключается в trainer.
7. CUDA Graph показал **3.671x** и **4.544x** ускорение в отдельном launch-sensitive benchmark.
8. Все smoke/correctness проверки Stage 1.12–1.17, приведённые в журнале этапов, прошли.

---

# Что НЕ следует утверждать по этим результатам

Нельзя утверждать, что конечный iPC training throughput уже равен `2.2 × 3.7` или `2.2 × 4.5`. Эти коэффициенты измерены на разных уровнях оптимизации.

Также пока нельзя утверждать, что найден абсолютно глобальный оптимум для RTX 3060: autotuner имеет ограниченный budget, а некоторые Stage 1.18 результаты относятся к специальному benchmark harness.

Корректная формулировка: **мы последовательно закрыли несколько независимых bottleneck'ов и получили подтверждённые выигрыши на kernel/end-to-end уровнях; следующий общий checkpoint должен быть единым full-training benchmark.**

---

# Финальная точка проекта на данный момент

На Stage 1.18 работу сознательно останавливаем.

Следующий потенциальный этап имеет смысл только после принятия этого checkpoint и, при продолжении работы, должен начинаться с одного воспроизводимого end-to-end измерения всего iPC training loop с уже активными:

```text
Stage 1.12 autotuning
+ Stage 1.17 grid.z policy
+ Stage 1.18 CUDA Graph
```

Никаких новых оптимизационных эвристик до такого измерения добавлять не требуется.
