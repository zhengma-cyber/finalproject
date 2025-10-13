from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf


def load_occupancy_data(save=False):
    occupancy_detection = fetch_ucirepo(id=357)

    X = occupancy_detection.data.features
    y = occupancy_detection.data.targets

    df = pd.concat([X, y], axis=1)

    num_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Occupancy'])
    if save:
        df.to_csv("occupancy_full.csv", index=False)
        print("Saved occupancy_full.csv")
    return df

def run_eda(df):
    print("=== Basic Info of Data ===")
    print(df.info())
    print(df.describe())
    print("\nClass distribution:\n", df['Occupancy'].value_counts(normalize=True))

    plt.figure(figsize=(5,4))
    sns.countplot(x='Occupancy', data=df, palette="Set2")
    plt.title("Occupancy Distribution (0=Empty, 1=Occupied)")
    plt.show()

    features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    for col in features:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='Occupancy', y=col, data=df, palette="Set3", inner="quartile")
        plt.title(f"{col} Distribution by Occupancy")
        plt.show()

    for col in features:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df, x=col, hue="Occupancy", fill=True, common_norm=False, alpha=0.5)
        plt.title(f"Distribution of {col} by Occupancy")
        plt.show()

    plt.figure(figsize=(8, 6))
    corr = df.drop(columns=['date']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    df['date'] = pd.to_datetime(df['date'])

    df['hour'] = df['date'].dt.hour
    hourly_occ = df.groupby('hour')['Occupancy'].mean()

    plt.figure(figsize=(8,5))
    plt.plot(hourly_occ.index, hourly_occ.values, marker='o')
    plt.title("Average Occupancy by Hour of Day")
    plt.xlabel("Hour of Day (0–23)")
    plt.ylabel("Average Occupancy (0~1)")
    plt.xticks(range(0,24))
    plt.grid(True)
    plt.show()


def plot_custom_autocorr_with_marks(series, max_lag=4000, freq_label="minutes"):
    series = series.reset_index(drop=True)
    lags = np.arange(max_lag)
    corr_coefs = np.zeros(max_lag)

    for i in range(1, max_lag):
        x = series.iloc[i:].reset_index(drop=True)
        y = series.iloc[:-i].reset_index(drop=True)
        corr_coefs[i] = x.corr(y, method="pearson")

    plt.figure(figsize=(10,5))
    plt.plot(lags, corr_coefs, color="steelblue", linewidth=1.5)
    plt.ylim([-1,1])
    plt.xlabel(f"Lag ({freq_label})")
    plt.ylabel("Pearson Correlation")
    plt.title("Custom Autocorrelation of Occupancy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axvline(x=1440, color="red", linestyle="--", label="1 day lag")
    plt.legend()
    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df = df.drop(columns=['date'])
    
    missing_info = df.isnull().sum()
    print("Missing values per column:\n", missing_info)
    
    y = df['Occupancy']
    X = df.drop(columns=['Occupancy'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print("Preprocessing done. Shape:", X.shape, y.shape)
    return X, y, scaler

def make_timeseries_splits(X, y, n_splits=5, test_size=None):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        splits.append((X_train, X_val, y_train, y_val))

    return splits

def print_timeseries_split_shapes(X, y, n_splits=5, test_size=None):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    print(f"\n=== TimeSeriesSplit ({n_splits} splits, test_size={test_size}) ===")
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Split {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
