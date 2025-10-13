from occupancy_utils import load_occupancy_data, run_eda, preprocess_data, plot_custom_autocorr_with_marks, make_timeseries_splits, print_timeseries_split_shapes

df = load_occupancy_data(save=True)
run_eda(df)
plot_custom_autocorr_with_marks(df['Occupancy'], max_lag=4000, freq_label="minutes")
X, y, scaler = preprocess_data(df)

print(X.head())
print(y.value_counts(normalize=True))
test_ratio = 0.2
test_size = int(len(X) * test_ratio)

X_trainval, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_trainval, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

splits = make_timeseries_splits(X_trainval, y_trainval, n_splits=5, test_size=2000)
X_train, X_val, y_train, y_val = splits[0]
print_timeseries_split_shapes(X_trainval, y_trainval, n_splits=5, test_size=2000)

