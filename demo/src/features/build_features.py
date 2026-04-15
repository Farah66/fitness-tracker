import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
# %%

from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[2]
data_path = BASE_DIR / "data" / "interim" / "02_outliers_removed_chauvenet.pkl"


df = pd.read_pickle(data_path)

predictor_columns = list(df.columns[:6])

#plot setting 
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# df.info()
# subset = df[df["set"] == 35]["gyr_y"].plot()

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info() 
   

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"]  == 1].index[-1] - df[df["set"] == 1].index[0]    
duration.seconds

for s in df["set"].unique():
    
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    df.loc[df["set"] == s, "duration"] = duration.seconds
   
duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5 
duration_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_low_pass = df.copy() 
low_pass = LowPassFilter()

fs = 1000 / 200
cutoff = 1

df_low_pass = low_pass.low_pass_filter(df_low_pass, "acc_y", fs, cutoff, order=5)

subset = df_low_pass[df_low_pass["set"] == 35]
print(subset[subset["label"][0]])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))

# Raw data
ax[0].plot(   
    subset["acc_y"].reset_index(drop=True),
    label="raw data"
)

# Filtered data
ax[1].plot(
    subset["acc_y_lowpass"].reset_index(drop=True),
    label="butterworth filter"
)

# Legends
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_low_pass = low_pass.low_pass_filter(df_low_pass, col, fs, cutoff, order=5)
    df_low_pass[col] = df_low_pass[col + "_lowpass"]
    del df_low_pass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_low_pass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

# Apply PCA (keep 3 components)
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Plot PCA components for one set
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
# Copy dataframe
df_squared = df_pca.copy()

# Sum of squares for accelerometer
acc_r = (
    df_squared["acc_x"]**2 +
    df_squared["acc_y"]**2 +
    df_squared["acc_z"]**2
)

# Sum of squares for gyroscope
gyr_r = (
    df_squared["gyr_x"]**2 +
    df_squared["gyr_y"]**2 +
    df_squared["gyr_z"]**2
)

# Take square root (magnitude)
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Select subset
subset = df_squared[df_squared["set"] == 14]

# Plot results
subset[["acc_r", "gyr_r"]].plot(subplots=True)

df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

window_size = int(1000 / 200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], window_size, "std")
    
df_temporal.list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std")
    df_temporal.list.append(subset)
    
df_temporal = pd.concat(df_temporal.list, ignore_index=True)  

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
  
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreAbs = FourierTransformation()

fs = int(1000 / 200)
window_size =  int(2800 / 200)

df_freq = FreAbs.abstract_frequency(df_freq, ["acc_y"], window_size, fs)  

#visualize result 
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot() 
subset[
    [
    "acc_y",
    "acc_y_max_freq",
    "acc_y_freq_weighted",
    "acc_y_pse",
    "acc_y_freq_1.418_Hz_ws_14",
    "acc_y_freq_2.482_Hz_ws_14",
    ]
].plot()
subset[cols_to_plot].plot(subplots=True, figsize=(12, 10))

df_freq.list = []
for s in df_freq["set"].unique():
    subset = df_freq[df_freq["set"] == s].copy()
    print(f"applying frequency abstraction for set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreAbs.abstract_frequency(subset, predictor_columns, window_size, fs)
    df_freq.list.append(subset)
    
df_freq = pd.concat(df_freq.list).set_index("epoch (ms)", drop=True) 
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()
df_freq= df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()


cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns].dropna()
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)    
plt.xlabel("k")
plt.ylabel("sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=3, n_init=20, random_state=0)
subset = df_cluster[cluster_columns].dropna()
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=c
    )

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

plt.legend()
plt.show()

# plot accelerometer data to compare with clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

for I in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == I]
    
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=I
    )

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

plt.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
# --------------------------------------------------------------
# %%
 