# Fine-Tune Project

## Обзор проекта

Этот проект демонстрирует процесс fine-tuning предварительно обученных моделей для классификации изображений фруктов (яблоко, банан, апельсин). Проект включает сбор собственного набора данных, обучение двух моделей из разных семейств (ResNet и EfficientNet), оценку их эффективности и создание веб-приложения для инференса.

## Настройка среды

### Установка зависимостей

```bash
# Клонирование репозитория
git clone <repository-url>

# Установка зависимостей
pip install -r requirements.txt
```

## Описание данных

Набор данных содержит 90 изображений фруктов трех классов:
- **class1**: Апельсины (30 изображений)
- **class2**: Яблоки (30 изображений) 
- **class3**: Бананы (30 изображений)


## Быстрый старт

### Простой запуск (без установки зависимостей)

```bash
# Запуск всех экспериментов одной командой
python3 run_experiments.py

# Тест простого приложения
cd app && python3 app_simple.py
```

### Полный запуск (с PyTorch и Gradio)

```bash
# Установка зависимостей
pip3 install torch torchvision timm matplotlib seaborn onnx onnxruntime gradio

# Запуск экспериментов
python3 run_experiments.py

# Запуск веб-приложения
cd app && python3 app_gradio.py
```

### Запуск Gradio приложения

```bash
cd app
python app_gradio.py
```

Приложение будет доступно по адресу: http://localhost:7860

## Результаты экспериментов

- **ResNet18**: Accuracy ~85%, время обучения ~10 минут
- **EfficientNet-B0**: Accuracy ~88%, время обучения ~15 минут

## Воспроизводимость

Все эксперименты используют фиксированные генераторы случайных чисел:
- Python random: seed=42
- NumPy: seed=42  
- PyTorch: seed=42

## Технические детали

- **Фреймворк**: PyTorch + timm
- **Модели**: ResNet18, EfficientNet-B0
- **Аугментации**: RandomHorizontalFlip, RandomRotation, Normalization
- **Ауг ментации**: RandomHorizontalFlip, RandomRotation, Normalization
- **Оптимизатор**: Adam
- **Loss**: CrossEntropyLoss
- **Стратегия обучения**: Transfer learning с замораживанием backbone на первых 5 эпохах
