import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_epochs(csv_files, metric_column='system_mean_waiting_time', label_column='step', legend_prefix='Epoch'):
    plt.figure(figsize=(12, 6))

    for idx, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            plt.plot(df[label_column], df[metric_column], label=f'{legend_prefix} {idx+1}')
        except Exception as e:
            print(f"Lỗi đọc file {file}: {e}")
            
    plt.title(f'{metric_column} theo {label_column} qua các epoch')
    plt.xlabel(label_column)
    plt.ylabel(metric_column)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epoch_averages(csv_files, metrics, legend_prefix='Epoch'):

    averages = {metric: [] for metric in metrics}
    
    for file in csv_files:
        try:
            print(f"Đang xử lý file: {file}")
            df = pd.read_csv(file)
            for metric in metrics:
                avg_value = df[metric].mean()
                averages[metric].append(avg_value)
        except Exception as e:
            print(f"Lỗi đọc file {file}: {e}")
            for metric in metrics:
                averages[metric].append(np.nan)

    averages["system_total_waiting_time_lag"] = averages["system_total_waiting_time"].copy() 
    for i in range(1, len(averages["system_total_waiting_time_lag"])):
        averages["system_total_waiting_time_lag"][i] = averages["system_total_waiting_time_lag"][i-1]

    averages["diff_waiting_time"] = np.array(averages["system_total_waiting_time_lag"])  - np.array(averages["system_total_waiting_time"]) 
    averages["reward"] = 0.1 * np.array(averages["system_mean_speed"]) + np.array(averages["diff_waiting_time"])

    # averages["reward"] =  0.1 * averages["system_mean_speed"] + averages["system_total_waiting_time"]
    metrics.append("reward")


    plt.figure(figsize=(14, 6))

    metrics = [ "reward", 'system_total_waiting_time' ]
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        plt.plot(range(1, len(averages[metric]) + 1), averages[metric], marker='o')
        plt.title(f'Trung bình {metric} theo epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Đường dẫn tới thư mục chứa file CSV của bạn
uploaded_csv_files = glob.glob(r"D:/UIT/Subjects/AI/DOAN/new_network/sumo-rl/outputs/4x4real/*.csv")
uploaded_csv_files.sort(key=os.path.getctime)

# Vẽ từng đường theo step
# plot_multiple_epochs(uploaded_csv_files, metric_column='agents_total_stopped')

choosen_epochs = [1, 10, 30, 100, 248]  # Chọn các 5 epoch cụ thể
some_epochs_csv_files = {uploaded_csv_files[i-1]: i for i in choosen_epochs}
plot_multiple_epochs(some_epochs_csv_files, metric_column='system_total_waiting_time')

# Vẽ trung bình mỗi epoch
plot_epoch_averages(uploaded_csv_files, metrics=['system_mean_speed', 'system_total_waiting_time'])
