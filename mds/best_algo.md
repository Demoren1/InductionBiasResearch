# Кооперативный поиск общей маски

Документ задаёт текущий joint-алгоритм, правила реальных измерений и порядок
запечатывания held-out данных. У каждой обучающей задачи свои functional bank и
генератор; общий evaluator сравнивает маски на всех обучающих задачах.

## Задачи и реальные измерения

Пусть есть $N$ обучающих и $J$ held-out задач. Маска связей
$M\in\{0,1\}^{F\times H}$ имеет фиксированный бюджет $K$:

$$
\mathcal M_K=\left\{M:\sum_{f=1}^{F}\sum_{h=1}^{H}M_{fh}=K\right\}.
$$

В основном режиме $K$ не меняется во время поиска; плотности teacher-карт в
functional banks не задают бюджет выходных масок.

Для каждой обучающей задачи $i$ выделены support $S_i$, query $Q_i$ и независимый
selection-набор $V_i$. Query и selection содержат разные наблюдения, но используют
один support и его контекст. Для оценки кандидата модель заново обучается только
на $S_i$ по фиксированному протоколу, затем измеряется на $Q_i$:

$$
\theta^T_{i,r}(M)=\operatorname{Fit}_T(M,S_i,\xi_{i,r}),\qquad
q_i^Q(M)=\frac1R\sum_{r=1}^{R}\mathcal L_{Q_i}\!\left(f_{\theta^T_{i,r}(M),M}\right).
$$

Пусть $d_i^Q=q_i^Q(\mathbf 1)$ — стоимость dense-контроля и
$\Delta_i^Q(M)=q_i^Q(M)-d_i^Q$. Чем меньше стоимость, тем лучше. Evaluator
предсказывает стоимость для маски и support-only контекста $c_i$. Его единственный
train set составляют реальные cross-task fresh fits масок из неизменяемого
начального набора functional banks. Membership определяется самой маской и
фиксируется до bootstrap feedback; оценки других масок не добавляются в обучение.

Selection выполняет отдельные fresh-fit измерения на $V_i$, сравнивает кандидатов
с dense по тем же задачам и выбирает frozen маску по общему критерию
$\Psi(\boldsymbol\Delta^V)$. Эти наблюдения не обучают evaluator. Held-out задачи и
их метки остаются запечатанными до сохранения frozen результата; после этого
итоговую маску оценивают на held-out данных отдельно от поиска.

В DeepSets-задаче стоимость набора изображений задаётся суммой меток классов,
а используемый masked MLP имеет вид

$$
y_i(X)=\sum_{a=1}^{s}\gamma_{i,d(x_a)},\qquad
f_{\theta,M}(X)=\sum_{a=1}^{s}\left[\sum_{h=1}^{H}a_h\tanh\!\left(\sum_{f=1}^{F}W_{fh}M_{fh}x_{a,f}+b_h\right)+o\right].
$$

Его метрика NMSE — это MSE, делённая на размер набора, а не на дисперсию меток:

$$
\mathcal L_A=\frac1{s|A|}\sum_{(X,y)\in A}\bigl(f_{\theta,M}(X)-y\bigr)^2.
$$

При обучении дочерней модели L2 применяется только к support-цели; query и
selection измеряют terminal-модель без этого штрафа.

## Functional banks и стартовый evaluator

Начальный сбор functional maps сохраняет прежнее правило. Для каждой train-задачи
отдельно обучаются source teachers на её support с кандидатными масками разных
плотностей. Source-query оценка используется для выбора карт внутри плотности,
но не становится quality label для evaluator. Из terminal-состояния teacher и его
маски на отдельном приватном train-probe строится functional card:

$$
\mathcal B_i=\{(C_{i,b},M_{i,b})\}_{b=1}^{B_i},\qquad
C_{i,b}=\Phi(\theta_{i,b},M_{i,b};\mathcal P_i).
$$

Probe, source-query, quality query и selection имеют разные роли. Для DeepSets,
обозначив $\widetilde W=W\odot M$, на probe-объекте $x_p$ вычисляем

$$
u_{ph}=\sum_f x_{p,f}\widetilde W_{fh}+b_h,\qquad
\psi_{ph}=a_h\tanh(u_{ph}),\qquad
g_{ph}=a_h\bigl(1-\tanh^2(u_{ph})\bigr).
$$

Профили связи для токена нейрона $h$ равны

$$
q^{\rm signed}_{fh}=\widetilde W_{fh}\frac1{|\mathcal P_i|}\sum_p x_{p,f}g_{ph},\qquad
q^{\rm abs}_{fh}=|\widetilde W_{fh}|\frac1{|\mathcal P_i|}\sum_p|x_{p,f}g_{ph}|,
$$

$$
q^{\rm rms}_{fh}=|\widetilde W_{fh}|\sqrt{\frac1{|\mathcal P_i|}\sum_p x_{p,f}^2g_{ph}^2}.
$$

Каждый token содержит активации $\psi_{:h}$, эти три профиля и столбец исходной
маски $M_{:h}$. Activation-профиль нормируется по RMS, профили связей — по
максимуму $q^{\rm rms}$; знаменатель каждого масштаба ограничен снизу $10^{-8}$.
Так generator получает функциональное описание teacher, а не только координаты
связей.

Все уникальные начальные маски, собранные из функциональных карт и стартовых
control/random кандидатов, измеряются real fresh fits на каждой train-задаче и
сохраняются в replay. Evaluator один раз обучается на cross-task fits масок,
входящих в неизменяемый начальный набор functional banks. Membership определяется
топологией маски, даже если её доступная cached fit-строка получена из другого
источника. Evaluator фиксируется после этого обучения и остаётся неизменным на
генераторном bootstrap и в search. Source-query, selection, held-out labels и fits
масок вне начального набора не входят в evaluator train set; их labels сохраняются
для real archive, replay и functional feedback генераторам.

## Три члена joint objective

Для каждой задачи действует отдельный генератор $G_{\omega_i}$ со своим банком.
Пусть $A_{i,s}=G_{\omega_i}(\mathcal B_i,z_s)$ — logits для draw $s$ и
$M_{i,s}=D_K(A_{i,s})$ — hard top-$K$ маска. Для соответствующего draw генераторы
получают общий шум $z_s$. Прямая величина в loss — hard exact-$K$ маска; градиент
идёт через straight-through relaxed mask

$$
\widetilde M_{i,s}=\operatorname{sg}(M_{i,s}-S_{i,s})+S_{i,s},\qquad
S_{i,s;fh}=\sigma\!\left(\frac{A_{i,s;fh}-\tau_{i,s}}{t_{i,s}}\right),\qquad
\sum_{f,h}S_{i,s;fh}\approx K.
$$

$\tau$ подбирает relaxed cardinality, $t$ задаёт температуру; оба вычисляются без
градиента. При сравнении масок скрытые столбцы совмещаются Hungarian assignment по
hard-маскам. Сопоставление detached, а MSE вычисляется по aligned straight-through
маскам.

**Own-task quality.** Фиксированный evaluator оценивает генератор $i$ только в его
собственном контексте $c_i$. Его веса не меняются, но градиент проходит от оценки
через relaxed mask к генератору. Положим $E(M,c_i)$ равным evaluator cost и $d_i^Q$
— реальной dense-стоимости:

$$
\mathcal L_{\rm quality}=\frac1{NS}\sum_{i=1}^{N}\sum_{s=1}^{S}
\left(E(\widetilde M_{i,s},c_i)-d_i^Q\right).
$$

**Pairwise agreement.** Для каждой неупорядоченной пары задач $i<j$ подбирается
$P_{ij,s}=\arg\min_{P\in\mathfrak S_H}\|M_{i,s}-M_{j,s}P\|_F^2$ по hard-маскам.
Agreement — средний MSE по draws, парам и координатам:

$$
\mathcal L_{\rm agreement}=\frac1{S\binom N2 FH}
\sum_{s=1}^{S}\sum_{i<j}
\|\widetilde M_{i,s}-\widetilde M_{j,s}P_{ij,s}\|_F^2.
$$

**Distillation к real archive.** Архив $\mathcal E_K$ содержит до
$L_{\rm elite}$ лучших реально измеренных train-масок бюджета $K$, ранжированных
по фактическому общему score $\Psi(\boldsymbol\Delta^Q)$. Архив не ограничен
масками, которые уже лучше dense на каждой задаче. Для каждого генератора, draw и
всех целей архива Hungarian matching находится отдельно по hard-маскам:

$$
P_{i,s,E}=\arg\min_{P\in\mathfrak S_H}\|M_{i,s}-EP\|_F^2,\qquad
\mathcal L_{\rm distill}=\frac1{NS|\mathcal E_K|FH}
\sum_{i,s,E\in\mathcal E_K}
\|\widetilde M_{i,s}-EP_{i,s,E}\|_F^2.
$$

Общий objective одного совместного обновления равен

$$
\mathcal L_{\rm joint}=\mathcal L_{\rm quality}
+\lambda\mathcal L_{\rm agreement}
+\mu\mathcal L_{\rm distill}.
$$

Веса по умолчанию $\lambda=0.1$ и $\mu=0.1$. Для каждого генератора все его
члены объединяются в один loss, затем выполняется один backward и один optimizer
step. Совместный режим не добавляет отдельные policy-gradient, BCE или
reconstruction шаги. Reconstruction pretraining выключен по умолчанию и
включается отдельно числом эпох больше нуля.

## Search, feedback и выбор результата

В начале и на каждом refresh полный candidate pool собирается из предложений всех
генераторов, random exact-$K$ масок, буквального top real archive и edge-swap
мутаций архивных масок. Уже известные топологии переиспользуют сохранённые
измерения; случайная ветка остаётся источником новых кандидатов.

Замороженный evaluator предсказывает cost каждой маски на всех train-контекстах. Общий
критерий по дельтам по умолчанию — worst task:

$$
\Psi_{\rm worst}(\boldsymbol\Delta)=\max_i\Delta_i.
$$

При acquisition budget $B$ сначала берутся первые $B$ кандидатов полного пула по
evaluator ranking, включая уже измеренные. Их измеренные строки доступны для
обновления selection и архива без дополнительных fits. Отдельно выбираются до
$B$ лучших ещё не измеренных масок, чтобы кэшированные кандидаты не занимали весь
новый measurement budget. Бюджет новых измерений считает маски-кандидаты: каждая
новая маска проходит fresh fits на всех $N$ train-задачах. Отдельных квот для
uncertainty или случайного acquisition нет.

После этих fits реальные строки добавляются в replay, real archive обновляется по
измеренному общему score, а fresh-fit состояния дают functional maps для банков
соответствующих задач. Evaluator остаётся неизменным: новые labels не запускают
его refresh. Selection сравнивает cached pool-top и вновь измеренные маски по независимым
$V_i$; сохраняется лучший результат по $\Psi(\boldsymbol\Delta^V)$. Только после
этого создаётся неизменяемый frozen checkpoint. Held-out задачи материализуются
лишь после checkpoint и не участвуют в bank, replay, evaluator, archive или selection.

## Bootstrap и запуски

Bootstrap собирает initial functional banks и cross-task real fits, один раз
обучает evaluator по фиксированному membership масок этих банков, затем сохраняет
frozen evaluator, membership, replay и банки. Default policy — `initial_bank_only`,
checkpoint schema v7.
Необязательный генераторный bootstrap включается флагом `--bootstrap-generators`;
он использует тот же frozen evaluator и не меняет определение joint objective.
В bash-запускателях `BOOTSTRAP_GENERATORS=1` передаёт флаг в обе фазы. Параметр
`EVALUATOR_EPOCHS` управляет только начальным обучением evaluator. `REFRESH_EVERY`
задаёт cadence раундов новых измерений и не запускает обучение evaluator.
Search через `WARM_START_FROM` переиспользует frozen evaluator bootstrap.

При warm-start из старого online-evaluator bootstrap его веса evaluator
переинициализируются: новый evaluator один раз обучается на cross-task fits масок
начального functional-bank набора, затем замораживается. Membership этого набора
сохраняется отдельно от расширенных live banks и повторно используется при
warm-start. Совместимость со старыми bank pickles сохраняется.

В `pattern.sh` стандартный train-набор —
`0000 0001 0011 0101 0110 0111 1000 1001 1010 1100 1110 1111`, held-out —
`0010 0100 1011 1101`. Они принадлежат разным reversal/complement-орбитам.
`TRAIN_PATTERNS` и `TEST_PATTERNS` переопределяют эти наборы сразу в обеих фазах.
В `deepsets.sh` роли задаются числами `TRAIN_TASKS` и `TEST_TASKS`.

Запускатели [pattern.sh](../sh_scripts/pattern.sh) и
[deepsets.sh](../sh_scripts/deepsets.sh) используют несколько видимых GPU и
батчинг независимых child fits. Реализации: [цикл](../generator_evaluator/runners/cooperative.py),
[pool и archive](../generator_evaluator/search/policy.py),
[evaluator ranking](../generator_evaluator/search/quality.py),
[joint objectives](../generator_evaluator/training/objectives.py),
[functional banks](../generator_evaluator/storage/functional.py),
[типы и адаптеры данных](../generator_evaluator/data/types.py) и
[реальные измерения](../generator_evaluator/evaluation/measurements.py).
