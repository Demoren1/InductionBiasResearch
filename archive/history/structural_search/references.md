# Связанные работы

Первичные источники проверены 25 октября 2026. Это рабочая карта литературы,
а не завершённая проверка новизны.

| Работа | Отношение к исследованию |
|---|---|
| [Shalev-Shwartz, Shamir, Shammah, Weight Sharing is Crucial to Succesful Optimization, 2017](https://arxiv.org/abs/1706.00687) | Added 26 October 2026 after inspecting the main separation theorems. Direct theoretical predecessor: weight sharing yields efficient optimization where the corresponding untied realizable model is hard for GD, with an exponential time gap in a constructed family. The general optimization advantage of structural constraints cannot be claimed as new. |
| [Li et al., Measuring the Intrinsic Dimension of Objective Landscapes, ICLR 2018](https://arxiv.org/abs/1804.08838) | Added 26 October 2026. Trains networks in randomly oriented low-dimensional subspaces; an important predecessor for the subspace formulation, without asserting the same source-to-target search protocol. |
| [Shamir, Distribution-Specific Hardness of Learning Neural Networks, JMLR 2018](https://jmlr.org/papers/v19/17-537.html) | Предшествующий анализ трудностей градиентного обучения даже при регулярных распределениях. Наше конкретное конечновременное вычисление для Gaussian sine нужно сопоставить с полными результатами этой работы; общая идея слабого градиентного сигнала не заявляется новой. |
| [Paul et al., Unmasking the Lottery Ticket Hypothesis, 2022](https://arxiv.org/abs/2210.03044) | Маска передаёт информацию о полезном осевом подпространстве. Здесь исследуется перенос распределения масок между задачами и конечный бюджет поиска. |
| [Liu, Simonyan, Yang, DARTS, ICLR 2019](https://arxiv.org/abs/1806.09055) | Предшествующая работа по дифференцируемому архитектурному поиску через непрерывное ослабление дискретной структуры. Наша sampling-схема и достаточные условия не делают сам двухуровневый поиск новой идеей. |
| [Peebles et al., Learning to Learn with Generative Models of Neural Network Checkpoints, 2022](https://arxiv.org/abs/2209.12892) | Генеративная модель обновлений весов, обученная на checkpoints. В нашей основной серии генерируются структуры, после чего веса обучаются заново. |
| [Wang et al., Neural Network Diffusion, 2024](https://arxiv.org/abs/2402.13144) | Генерация параметров через автоэнкодер и diffusion в latent-пространстве. Связана с направлением переноса численных codebook, которое следует отделять от переноса назначений. |
| [Kingma, Welling, Auto-Encoding Variational Bayes, 2013](https://arxiv.org/abs/1312.6114) | Исходная VAE-постановка. Наше условие покрытия относится к агрегированному posterior и дискретному декодеру, а не автоматически к любому используемому training objective. |
| [Kusner, Paige, Hernández-Lobato, Grammar Variational Autoencoder, 2017](https://arxiv.org/abs/1703.01925) | Структурированный decoder и оптимизация в latent-пространстве дискретных объектов. Важный предшественник идеи генеративного prior для поиска. |
| [Nowlan, Hinton, Simplifying Neural Networks by Soft Weight-Sharing, 1992](https://www.cs.toronto.edu/~hinton/absps/sunspots.pdf) | Историческая связь с группированием весов и общими значениями. Не следует приписывать этой работе нашу конкретную схему поиска назначений на новой задаче. |
| [Hinton, van Camp, Keeping Neural Networks Simple by Minimizing the Description Length of the Weights, COLT 1993](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) | Длина описания весов, простота модели и способы кодирования. DOI: [10.1145/168304.168306](https://doi.org/10.1145/168304.168306). |
| [Han, Mao, Dally, Deep Compression, ICLR 2016](https://arxiv.org/html/1510.00149v5) | Явные codebook/indices, квантование и дообучение общих центроидов после обучения сети. Полезная техническая отправная точка; цель нашей схемы — оптимизационный поиск, а не только сжатие. |

Потенциальное позиционирование: достаточно конкретные условия, при которых
поиск по дискретно выбранным подпространствам повышает вероятность достижения
хорошего решения при ограниченном бюджете; разделение источников выигрыша
между структурой, инициализацией и генератором. До подачи нужен полный обзор
learned optimization, subspace optimization, weight sharing и quantized
training, с сопоставлением не только аннотаций, но и теорем/экспериментов.
