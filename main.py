import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data
data = pd.read_csv("GDP_per_capita(current_US$).csv")

# Normalize data
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(data.iloc[:, 1:]), columns=data.columns[1:], index=data.index)

# Perform clustering using KMeans algorithm
kmeans = KMeans(n_clusters=3, random_state=42).fit(data_norm)

# Add cluster labels as a new column to the dataframes
data['cluster'] = kmeans.labels_

# Plot cluster membership and cluster centers
plt.scatter(data['Time'], data['United States'], c=data['cluster'])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.title('Exports of goods and services (% of GDP)')

# Save the plot as a PNG file
plt.savefig('GDP_Clustering.png')

# Predict GDP values using polynomial regression
x = data['Time'].values.reshape(-1, 1)
y = data['United States'].values

poly_features = PolynomialFeatures(degree=3)
x_poly = poly_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# Generate predictions
future_years = range(2000, 2031)
future_x = np.array(list(future_years)).reshape(-1, 1)
future_x_poly = poly_features.transform(future_x)
predicted_values = model.predict(future_x_poly)

# Plot the actual GDP vs. predicted GDP
plt.figure()
plt.plot(data['Time'], data['United States'], 'o-', label='Actual GDP')
plt.plot(future_years, predicted_values, label='Predicted GDP')
plt.xlabel('Years')
plt.ylabel('GDP')
plt.title('Actual GDP vs. Predicted GDP')
plt.legend()

# Save the plot as a PNG file
plt.savefig('GDP_Forecast.png')

# Show the plots
plt.show()
