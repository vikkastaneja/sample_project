import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

heart_disease = pd.read_csv("/Users/vtaneja/AI_ML/sample_project/heart-disease.csv")
over_50 = heart_disease[heart_disease.age > 50]
fig, ax = plt.subplots(figsize=(10,6))

# Define colors based on cholesterol levels
arr = np.where(over_50.chol < 200, 'g', np.where(over_50.chol <300, 'y', 'r'))

# Scatter plot with custom colors
scatter = ax.scatter(x=over_50['age'], y=over_50.chol, c=arr)

# Customize
ax.set(title='Heart Disease and Cholesterol Levels', xlabel='Age', ylabel='Cholesterol', mouseover=True)

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Low Cholesterol', markerfacecolor='g', markersize=10),
                   plt.Line2D([0], [0], marker='o', color='w', label='Medium Cholesterol', markerfacecolor='y', markersize=10),
                   plt.Line2D([0], [0], marker='o', color='w', label='High Cholesterol', markerfacecolor='r', markersize=10),
                   plt.Line2D([0], [0], linestyle='--', label='Mean Cholesterol', markerfacecolor='b', markersize=10)]

ax.legend(handles=legend_elements, title='Cholesterol Levels')

# Add a horizontal line for mean cholesterol
mean_chol = over_50.chol.mean()
line = ax.axhline(mean_chol, linestyle='--', label='Mean Cholesterol')

# Add legend
# ax.legend()
print(mean_chol)
# Function to format the cursor
def cursor_formatter(sel):
    print('called')
    x, y, *_ = sel.target
    sel.annotation.set_text(f"Cholesterol: {mean_chol:.2f}")

# Add cursor to show current value on mouseover
mplcursors.cursor(line).connect("add", cursor_formatter)
plt.show();