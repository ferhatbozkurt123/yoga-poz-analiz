import pandas as pd
import numpy as np
# Load your data as usual
data = pd.read_csv("results2.csv")

# Select only numeric columns before calculating the mean
numeric_data = data.select_dtypes(include=['number'])
ideal_angles = data.groupby('Label')[numeric_data.columns].mean()
#ideal_angles = ideal_angles.applymap(lambda x: x if x <= 180 else 180 - x)

# Display the ideal angles for each pose
print(ideal_angles)

ideal_angles.to_csv('ideal_angles.csv')