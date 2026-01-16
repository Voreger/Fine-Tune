from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Пути
    data_dir: str = "data/raw"       # папка с изображениями
    out_dir: str = "outputs"         # куда сохранять модели и логи

    # Настройки обучения
    img_size: int = 224              # размер картинок
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 2  # сколько эпох замораживаем backbone

    # Настройки модели
    model_name: str = "resnet18"     # timm модель. 
    pretrained: bool = True
    num_classes: int = 3             # количество классов (автообновляется в train.py)

    # Настройки валидации и генерации
    val_ratio: float = 0.2
    num_workers: int = 2

    # Случайность
    seed: int = 42
    device: str = "cuda"             # "cuda" или "cpu"
