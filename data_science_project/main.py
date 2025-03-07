
import pandas as pd      # Pandas is a library used for the data processing
import seaborn as sns
from matplotlib import pyplot as plt


# data read and cleaning
data = pd.read_csv("iris.csv")     #CSV: comma seperated value
print(data.head())         # the head() function is used to view the first 5 data

data.info()     # information about the data


print(data.columns)     #['sepal.length', 'sepal.width', 'petal.length', 'petal.width',  'variety'],dtype='object'

print(data.groupby("variety").agg(["min", "max", "std", "mean"]))      # variety e göre  sınıflandırma

print(data.isna())   # False means that there is no missing value in the data set

print(data.isna().sum())   # count of the na values

# data visualization
# The scatter plot is used for seeing the effect or relationship of one variable to another.
# The hue parameter displays the graph according to the categorical variable
for column in data.columns[1:-1]:
    sns.scatterplot(data=data, x=data.index, y=column, hue="variety")
    plt.show()


# outlier detection part
# 3 sigma
for column in data.columns[1:-1]:
    for var in data["variety"].unique():
        selected_var = data[data["variety"] == var]
        selected_column = selected_var[column]

        std = selected_column.std()
        avg = selected_column.mean()

        three_sigma_plus = avg + (3 * std)
        three_sigma_minus = avg - (3 * std)
        outliers = selected_column[((selected_var[column] > three_sigma_plus) | (selected_var[column] < three_sigma_minus))]
        outlier_indices = outliers.index
        data.drop(index=outlier_indices, inplace=True)
        print(outliers)

# IQR - Quantile

for column in data.columns[1:-1]:
    for var in data["variety"].unique():
        selected_var = data[data["variety"] == var]
        selected_column = selected_var[column]

        q1 = selected_column.quantile(0.25)
        q3 = selected_column.quantile(0.75)
        iqr = q3 - q1
        minimum = q1 - (1.5 * iqr)
        maximum = q3 + (1.5 * iqr)

        print(column, var,"| min = ",minimum,"max = ",maximum)
        max_ids = data[(data["variety"] == var) & (data[column] > maximum)].index
        min_ids = data[(data["variety"] == var) & (data[column] < minimum)].index

        data.drop(min_ids, inplace=True)
        data.drop(max_ids, inplace=True)

for column in data.columns[1:-1]:  # Repeat scatter plot after cleaning data
    sns.scatterplot(data=data, x=data.index, y=column, hue="variety")
    plt.title(f"Final Scatter Plot for {column}")  # Add title for clarity
    plt.show()  # Show final plot for each column
    plt.close()  # Close the plot to avoid further issues

#save the model
data.to_csv("final_data.csv")
















