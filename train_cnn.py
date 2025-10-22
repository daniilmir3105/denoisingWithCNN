#!/usr/bin/env python3
"""
Скрипт для удаления шума из сейсмограмм.
Шаги:
1. Загрузка исходных данных двух сейсмограмм (чистая и шумная) с формой (1151, 400000).
2. Для каждой трассы вычисляется амплитудный спектр с использованием rfft (получается 576 точек).
3. Сохранение вычисленных спектров в файлы .npy для дальнейшего использования.
4. Создание PyTorch Dataset и DataLoader для формирования пар (X, Y), где X – спектры с шумом, Y – чистые спектры.
5. Реализация двух архитектур: полносвязной (MLP) и небольшой 1D-CNN.
6. Обучение сети с разделением данных на train/val, использованием MSELoss и оптимизатора Adam.
7. Сохранение обученной модели.
8. Для работы с большим объёмом данных используется обработка чанками и memory mapping.

Перед запуском обновите пути к файлам данных.
"""

import numpy as np
import os
import matplotlib.pyplot as plt  # для возможной визуализации
import scipy.signal as scs  # если нужны дополнительные сигнальные операции
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# =============================================================================
# Функция для прямого спектрального преобразования (с использованием rfft)
# =============================================================================
# def seis_fft(seismogram, dt):
#     """
#     Выполняет FFT для каждой трассы сейсмограммы.

#     :param seismogram: numpy массив размером (num_traces, num_samples), где каждая строка — трасса.
#     :param dt: временной шаг (сек).
#     :return: freqs (в Гц) и комплексное спектральное представление (размер: (num_traces, num_freq_bins)).
#     """
#     num_traces, num_samples = seismogram.shape
#     sampling_rate = 1 / dt  # в Гц
#     freqs = np.fft.rfftfreq(num_samples, d=1 / sampling_rate)
#     spectrum = np.fft.rfft(seismogram, axis=1)
#     return freqs, spectrum

# Прямое преобразование Фурье
def seis_fft(seismogram, dt):
    """
    Выполняет полное FFT для каждой трассы сейсмограммы.
    
    :param seismogram: numpy массив (n_traces, n_samples)
    :param dt: шаг по времени (сек)
    :return: (частоты, спектр)
    """
    num_traces, num_samples = seismogram.shape
    sampling_rate = 1 / dt
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=dt))  # Симметричные частоты
    spectrum = np.fft.fftshift(np.fft.fft(seismogram, axis=1), axes=1)  # Центрируем спектр
    
    return freqs, spectrum

# =============================================================================
# Функция для обработки сейсмограммы (рассчитывает спектры с использованием rfft)
# =============================================================================
def process_seismic_data(file_path, output_path):
    """
    Загружает сейсмограмму из файла .npy, транспонирует её так, чтобы каждая трасса была строкой,
    и вычисляет амплитудный спектр с использованием rfft (т.е. спектры для положительных частот).

    :param file_path: Путь к исходному файлу с сейсмограммой (ожидается форма: (1151, 400000)).
    :param output_path: Путь для сохранения файла с спектрами (каждая трасса -> вектор из 576 значений).
    """
    data = np.load(file_path, mmap_mode='r')  # data.shape = (1151, 400000)
    # Транспонируем, чтобы каждая трасса стала строкой: (num_traces, num_samples)
    data = data.T  # теперь shape: (400000, 1151)
    num_traces, num_samples = data.shape
    # Вычисляем временной шаг dt (используем, например, значение последнего отсчёта первой трассы)
    dt = data[0, -1] / num_samples
    # Вычисляем спектральное представление для всех трасс
    freqs, spectrum = seis_fft(data, dt)
    amplitude_spectra = np.abs(spectrum)  # (num_traces, num_freq_bins); для 1151 отсчетов → 576 точек
    np.save(output_path, amplitude_spectra)
    print(f"Сохранены спектры в файл: {output_path}")


# =============================================================================
# Определение PyTorch Dataset
# =============================================================================
class SeismicDataset(Dataset):
    """
    Датасет для работы с парами спектров:
    X – шумный спектр, Y – чистый спектр.
    """

    def __init__(self, noisy_file, clean_file):
        self.noisy_spectra = np.load(noisy_file)
        self.clean_spectra = np.load(clean_file)

    def __len__(self):
        return len(self.noisy_spectra)

    def __getitem__(self, idx):
        x = self.noisy_spectra[idx]
        y = self.clean_spectra[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# =============================================================================
# Определение модели MLP (полносвязная сеть)
# =============================================================================
class MLPModel(nn.Module):
    def __init__(self, input_size=576, hidden_size=256):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Определение модели 1D-CNN
# =============================================================================
class CNNModel(nn.Module):
    def __init__(self, input_length=576):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = out.squeeze(1)
        return out


# =============================================================================
# Функция для обучения модели
# =============================================================================
def train_model(model, train_loader, val_loader, device, num_epochs=20, learning_rate=1e-3):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Эпоха {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    return model


# =============================================================================
# Основная функция
# =============================================================================
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    clean_data_path = os.path.join(current_dir, "data", "Anisotropic_FD_Model_Shots_part1_without_noise.npy")
    noisy_data_path = os.path.join(current_dir, "data", "Anisotropic_FD_Model_Shots_part1_snr=5.npy")

    # Пути для сохранения спектров
    clean_spectra_path = os.path.join(current_dir, "spectres", "clean_spectra.npy")
    noisy_spectra_path = os.path.join(current_dir, "spectres", "noisy_spectra.npy")

    # Шаг 1. Обработка сейсмограмм для получения спектров (используем seis_fft)
    print("Обработка чистой сейсмограммы...")
    process_seismic_data(clean_data_path, clean_spectra_path)
    print("Обработка шумной сейсмограммы...")
    process_seismic_data(noisy_data_path, noisy_spectra_path)

    # Шаг 2. Формирование датасета
    dataset = SeismicDataset(noisy_spectra_path, clean_spectra_path)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Шаг 3. Выбор архитектуры модели
    # Раскомментируйте нужный вариант:
    # model = MLPModel(input_size=576, hidden_size=256)
    model = CNNModel(input_length=576)

    # Определяем устройство (GPU, если доступно)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Шаг 4. Обучение модели
    num_epochs = 1000
    model = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, learning_rate=1e-3)

    # Шаг 5. Сохранение весов модели
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Веса модели сохранены в 'model_weights.pth'.")


if __name__ == "__main__":
    main()
