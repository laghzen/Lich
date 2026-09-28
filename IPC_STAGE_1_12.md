# IPC Stage 1.12 — итоговая фиксация

**Статус:** завершено как рабочий baseline для дальнейшей оптимизации iPC/TileLang.

**Дата фиксации:** 2026-09-29

---

## 1. Цель этапа

Цель Stage 1.12 и последующей цепочки Stage 1.12–1.19 — превратить iPC training/inference workload в практический GPU-optimized pipeline на фиксированном железе и построить online autotuner, который:

1. работает непосредственно под целевой GPU;
2. использует только legal TileLang-конфигурации;
3. ищет хороший kernel быстро, не превращая autotuning в полный перебор;
4. принимает решения по свежим GPU-измерениям;
5. не зависит от устаревшего measurement cache;
6. может работать на всём конечном legal-пространстве, но прекращать поиск, когда дальнейшая стоимость экспериментов уже не оправдывает ожидаемое улучшение.

---

## 2. Целевая аппаратно-программная конфигурация

Проект тестируется на фиксированной платформе:

- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU
- **VRAM:** 6 GB
- **Compute Capability:** SM86 / `(8, 6)`
- **Python:** 3.10.0
- **PyTorch:** 2.5.1+cu121
- **CUDA build:** 12.1
- **TileLang:** 0.1.13
- **OS:** Windows 10

Все ограничения autotuner и допустимость TileLang kernel-конфигураций ориентированы прежде всего на эту платформу.

---

## 3. Базовое legal-пространство autotuner

Генерируемый глобальный pool содержит:

- `block_m ∈ {16, 32, 64, 128, 256}`
- `block_n ∈ {32, 64, 128, 256}`
- `block_k ∈ {16, 32, 64}`
- `threads ∈ {64, 128, 256}`
- `num_stages ∈ {1, 2, 3}`
- `swizzle ∈ {False, True}`
- `swizzle_panel = 8`
- `shared_swizzle = False`

Полный предгенерированный pool: **756 конфигураций**.

Для prediction GEMM после статической фильтрации на размере `M=64, K=64, B=128` остаётся:

- **612 legal**
- **144 static-invalid**

Статическая фильтрация выполняется до дорогого JIT/GPU запуска. В неё входят аппаратно/структурно невозможные комбинации, включая ограничения tile shape, warp partition, shared-memory footprint и `block_k` alignment.

---

## 4. Пройденные базовые проверки

До online autotuning были зафиксированы smoke-проверки TileLang kernels:

```text
run_cli.py smoke --M 64 --K 64 --B 128
→ PASS (regular + tail-safe + save-A paths)

run_cli.py smoke --M 10 --K 64 --B 128
→ PASS (regular + tail-safe + save-A paths)
```

Это подтверждает корректность основных путей kernel generation для обычного и tail-safe случая.

---

## 5. Прогресс autotuner

### Stage 1.11 / ранний поиск

Первоначальный autotuner использовал ограниченное количество GPU-оценок. В результате исторически были получены конфигурации порядка `0.018–0.020 ms` для prediction workload.

Эти результаты стали ориентиром для последующего online search, но не являются обязательной частью нового запуска: современный tuner должен уметь найти хороший вариант с нуля.

### Stage 1.12

Была введена legality-aware адаптивная модель поиска:

```text
global_pool = 756
legal_prediction = 612
legal_weight = 500
```

Также появились persistent cache и adaptive candidate selection. Позднее стало ясно, что measurement cache не должен определять поведение нового online запуска: прошлое измерение может быть устаревшим из-за изменения runtime conditions, JIT state, thermals и самого kernel generator.

### Stage 1.19 v7/v8

Была исправлена ошибка порядка correctness/probe execution:

```text
compile → real kernel launch → synchronize → correctness → benchmark
```

После этого все 38 экспериментальных/verification kernels в одном из запусков были успешно обработаны.

Ограничение `max-evals` показало себя недостаточным для требуемой постановки: tuner мог остановиться после десятков точек и никогда не получить шанс исследовать действительно перспективную конфигурацию.

### Stage 1.19 v9

В v9 numerical `max-evals` был убран из алгоритма поиска. Tuner получил semantic search tree, factorized GP, LCB и region pruning.

Однако эксперимент показал, что модель могла слишком рано признать крупные части пространства статистически плохими, а исторический JSON cache не был гарантированно доступен в среде запуска.

### Stage 1.19 v10

Measurement cache был отключён по умолчанию.

Tuner перешёл на online search с:

- startup probes;
- local racing;
- EI;
- global scouts;
- online feedback;
- conservative convergence.

Первый успешный результат:

```text
16 новых GPU-оценок
BEST = 0.0152 ms
```

Конфигурация:

```text
KernelConfig(
    block_m=32,
    block_n=32,
    block_k=64,
    threads=64,
    num_stages=1,
    swizzle=True,
    swizzle_panel=8,
    shared_swizzle=False,
)
```

---

## 6. Финальный результат online autotuning

Финальный рабочий запуск был выполнен с чистого online состояния:

```text
measurement cache: OFF
```

Команда:

```bat
"D:\Setup\Python 3.10\python.exe" .\scripts\run_cli.py autotune --kind prediction --M 64 --K 64 --B 128 --seed-evals 10 --batch-size 6 --topk 6 --probe-warmup 2 --probe-reps 5 --verify-warmup 10 --verify-reps 40 --out results\autotune_stage_1_19_v12.json
```

Ход поиска:

```text
startup probes → 0.0184 ms

12 evaluations → 0.0173 ms

16 evaluations → 0.0152 ms
```

Финальный найденный кандидат:

```text
block_m       = 32
block_n       = 32
block_k       = 64
threads       = 64
num_stages    = 1
swizzle       = True
swizzle_panel = 8
shared_swizzle= False
```

Измеренный результат:

```text
BEST = 0.0152 ms
```

Для сравнения, более ранние online/cached runs давали результаты около `0.018–0.020 ms`. Таким образом, новый online search самостоятельно воспроизвёл и превзошёл прежний рабочий уровень без зависимости от старого measurement cache.

---

## 7. Почему текущая стратегия считается завершённой

В этом проекте JIT-компиляция существенно дороже, чем сама последующая оценка latency. Поэтому полный перебор всех 612 legal-конфигураций не является автоматически лучшим решением.

Практическая цель autotuner:

> получить практически минимальную latency за минимальное число дорогих compile/GPU experiments.

Текущая стратегия достигает именно этого режима:

```text
1. Статически отбрасываются невозможные configurations.
2. Выполняется небольшой диверсифицированный startup.
3. Каждый результат сразу изменяет следующий выбор.
4. Улучшение incumbent переводит поиск в новый локальный basin.
5. Global scouts не дают застрять исключительно в одном локальном регионе.
6. После устойчивого plateau поиск завершается.
7. Measurement cache не требуется для нахождения оптимума текущего запуска.
```

Важно различать два кэша:

### Measurement cache

В текущем рабочем режиме **OFF**.

Прошлые latency observations не используются для принятия решений нового запуска.

### TileLang compiler cache

Остаётся включённым и используется только как технический механизм повторного использования уже скомпилированных артефактов. Это не является частью модели оптимальности.

---

## 8. Что было признано неправильным и исправлено

### 8.1. Жёсткий `max-evals`

Плохая схема:

```text
search until N evaluations
```

Она искусственно ограничивала пространство независимо от того, насколько перспективной была текущая search trajectory.

Текущая схема не использует численный budget как критерий оптимизации.

### 8.2. Зависимость от исторического measurement cache

Прошлый measurement может быть устаревшим. Новый tuner должен быть способен стартовать полностью с нуля и сам получать runtime truth от GPU.

### 8.3. Глобальный surrogate-only LCB stop

Стратегия:

```text
model says every untested point is probably worse
→ stop
```

оказалась слишком агрессивной для дискретного TileLang-пространства.

Она была удалена из принятой логики.

### 8.4. Почти последовательный перебор

После удаления global stop tuner не должен превращаться в:

```text
point 1
point 2
point 3
...
point 300
```

Логика должна оставаться model-guided и incumbent-driven.

---

## 9. Принцип текущего online tuner

Итоговый подход можно записать следующим образом:

```text
                     ┌─────────────────────┐
                     │  legal finite pool  │
                     │     612 configs     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ startup exploration │
                     │   diverse probes    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  current incumbent  │
                     └──────────┬──────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
       local challengers                 global scouts
       around incumbent                  unexplored regions
                │                               │
                └───────────────┬───────────────┘
                                ▼
                     ┌─────────────────────┐
                     │  next GPU experiment│
                     └──────────┬──────────┘
                                │
                                ▼
                     latency / correctness
                                │
                                ▼
                     update search online
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              improvement               plateau
                    │                       │
                    ▼                       ▼
             re-center search       convergence test
```

Это является рабочим определением «интеллектуального online autotuning» для данного проекта.

---

## 10. Текущее лучшее состояние prediction kernel

```text
Workload:
  kind = prediction
  M    = 64
  K    = 64
  B    = 128

Best latency:
  0.0152 ms

Best config:
  block_m       = 32
  block_n       = 32
  block_k       = 64
  threads       = 64
  num_stages    = 1
  swizzle       = True
  swizzle_panel = 8
  shared_swizzle= False

Search mode:
  online
  measurement cache = OFF
```

---

## 11. Validation and quality criteria

На уровне software search infrastructure были зафиксированы следующие свойства:

- static legality tests проходят;
- cache-off startup работает;
- correctness выполняется после реального kernel invocation;
- persistent failures не должны повторно запускаться в рамках cache-aware режима;
- semantic-tree splitting работает по фактическим значениям параметров;
- adaptive search способен завершаться без полного перебора на гладком/синтетическом workload;
- online tuner способен находить улучшение incumbent после startup.

Практический GPU-результат подтверждает, что поиск действительно меняет траекторию по обратной связи:

```text
0.0184 ms → 0.0173 ms → 0.0152 ms
```

---

## 12. Итоговый вывод Stage 1.12

Stage 1.12 считается **закрытым в качестве рабочего autotuning baseline**.

Основной результат этапа — не просто наличие набора TileLang kernels, а сформированная практическая схема:

> **legal finite-space + fresh online GPU measurements + model-guided racing + local exploitation + global exploration + adaptive convergence.**

Критический принцип проекта:

> **Кэш не должен находить оптимум вместо autotuner. Оптимум нового запуска должен определяться текущими измерениями GPU.**

При этом нет смысла механически измерять все 612 legal-конфигураций. Экспериментально показано, что хорошая конфигурация может быть найдена за малую долю пространства — в последнем запуске `0.0152 ms` найдено после 16 новых GPU-оценок.

Поэтому дальнейшее увеличение количества итераций должно выполняться только тогда, когда ожидаемое улучшение latency достаточно велико, чтобы оправдать стоимость дополнительных JIT-компиляций и GPU-измерений.

---

## 13. Следующий этап

Stage 1.12 не требует дальнейшего раздувания autotuner ради самого autotuner.

Рациональная стратегия после фиксации этого baseline:

1. сохранить текущую лучшую конфигурацию `32×32×64 / 64 threads / 1 stage / swizzle=True` как рабочий prediction candidate;
2. перейти к следующему узкому месту iPC;
3. повторно применять тот же online tuning pattern только там, где измеренный выигрыш способен окупить стоимость поиска;
4. периодически валидировать итоговый kernel в составе полного training loop, а не только на isolated GEMM microbenchmark.

---

## 14. Финальная фиксация

**Stage 1.12 — DONE.**

**Autotuning mode:** online, cache-independent.

**Best observed prediction latency:** **0.0152 ms**.

**Best observed configuration:** **`block_m=32, block_n=32, block_k=64, threads=64, num_stages=1, swizzle=True`**.

**Позиция по дальнейшему поиску:** не расширять число запусков без доказуемого экономического смысла; текущий adaptive stopping является частью дизайна, а не недостатком полноты перебора.
