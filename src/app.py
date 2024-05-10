from utils import db_connect
engine = db_connect()

# your code here

#LOADING THE DATASET
#Step 1: Problem statement and data collection

import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

data.head()

#Store the raw data in the ./data/raw folder
data.to_csv("../data/raw/data.csv", index = False)

#Step 2: Exploration and data cleaning
data.shape

#Obtain information about data types and non-null values
data.info()
#In conclusion:
#There are a total of 48895 rows and 16 columns.
#The variables las_review and reviews_per_months only have 38843 instances with values, so it would contain more than 10000 null values. 
#The variables name and host_name also have null values, but in a much smaller number than the previous ones. 
#The rest of the variables always have a value.
#The data has 10 numerical characteristics and 6 categorical characteristics.

print(f"The number of duplicated Name records is: {data['name'].duplicated().sum()}")
print(f"The number of duplicated Host ID records is: {data['host_id'].duplicated().sum()}")
print(f"The number of duplicated ID records is: {data['id'].duplicated().sum()}")
#There are duplicates results in Name and Host ID variables, however they are should not be deleted because names of different houses may coincide or the same host may have  few apartments/houses, so thus, the variables are not duplicated.

#Eliminate irrelevant information
data.drop(["id", "name", "host_name", "latitude", "longitude", "reviews_per_month", "last_review", "calculated_host_listings_count"], axis = 1, inplace = True)
data.head()

#Step 3: Analysis of univariate variables

#Analysis on categorical variables
import matplotlib.pyplot as plt 
import seaborn as sns

fig, axis = plt.subplots(2, 3, figsize=(10, 7))

# Create Histogram
sns.histplot(ax = axis[0,0], data = data, x = "host_id")
sns.histplot(ax = axis[0,1], data = data, x = "neighbourhood_group").set_xticks([])
sns.histplot(ax = axis[0,2], data = data, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = data, x = "room_type")
sns.histplot(ax = axis[1,1], data = data, x = "availability_365")
fig.delaxes(axis[1, 2])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
#With the representation of each variable, we can determine that:
#There are few hosts that have many properties.
#The majority of real estate belongs to Brooklyn and Manhattan neighbourhoods.
#The mostly offered room type is entire home/apartment, private room comes next, and shared room is the least common.
#The houses are available 365 days, however, most of them has a low availability, perhaps, due to being highly demanded.

#Analysis on numeric variables
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
fig, axis = plt.subplots(3, 2, figsize = (10, 7))

# Create a multiple histogram
sns.histplot(ax = axis[0, 0], data = data, x = "price").set_xlim(0, 1000)
sns.boxplot(ax = axis[1, 0], data = data, x = "price")
sns.histplot(ax = axis[0, 1], data = data, x = "minimum_nights").set_xlim(0, 50)
sns.boxplot(ax = axis[1, 1], data = data, x = "minimum_nights")
sns.histplot(ax = axis[2, 0], data = data, x = "number_of_reviews").set_xlim(0, 150)
sns.boxplot(ax = axis[2, 1], data = data, x = "number_of_reviews")

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

#Step 4: Analysis of multivariate variables
#Numerical-numerical analysis
fig, axis = plt.subplots(2, 2, figsize = (10, 7))

# Create a multiple scatter diagram
sns.regplot(ax = axis[0, 0], data = data, x = "minimum_nights", y = "price")
sns.heatmap(data[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = data, x = "number_of_reviews", y = "price").set(ylabel=None)
sns.heatmap(data[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

#From the above diagrams we can see that there's no relation between the price and minimum number of nights, neither between the price and number of reviews.

#Categorical-categorical analysis

fig, axis = plt.subplots(figsize = (7, 4))

sns.countplot(data = data, x = "room_type", hue = "neighbourhood_group")

plt.show()

#From the chart we can see:
#Brooklyn and Manhattan are the neighbourhoods with the highest rent offer, while in Brooklyn the mostly offered room type is private room, in Manhattan is entire home/apartment. Queens comes next offering more private rooms to rent than entire apartments. Bronx represents a very small part of the properties offered, mainly offering private rooms. The last and the least popular neighbourhood for renting is Staten Island, where private room and entire apartment are equally popular.   

#Numerical-categorical analysis (complete)

# Factorizing (convert categorical values into numerical labels) values:
data["neighbourhood_group"] = pd.factorize(data["neighbourhood_group"])[0]
data["room_type"] = pd.factorize(data["room_type"])[0]
data["neighbourhood"] = pd.factorize(data["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(10, 7))

sns.heatmap(data[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

# Draw Plot
plt.show()

#From the heatmap it can be concluded that there is a very low correlation between all the values. It´s likely that price may depend on the room_type. 

#Analysing all the Data at Once
import matplotlib.pyplot as plt

sns.pairplot(data = data)

#Store the intermediate data in the ./data/interim folder
data.to_csv("../data/interim/data_day1.csv", index = False)

#Step 5: Feature engineering

#Outlier analysis
data.drop(["id", "name", "host_name", "latitude", "longitude", "reviews_per_month", "last_review", "calculated_host_listings_count"], axis = 1, inplace = True)
data.describe()

fig, axis = plt.subplots(2, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = data, y = "neighbourhood_group")
sns.boxplot(ax = axis[0, 1], data = data, y = "room_type")
sns.boxplot(ax = axis[0, 2], data = data, y = "price")
sns.boxplot(ax = axis[1, 0], data = data, y = "minimum_nights")
sns.boxplot(ax = axis[1, 1], data = data, y = "number_of_reviews")
sns.boxplot(ax = axis[1, 2], data = data, y = "availability_365")

plt.tight_layout()

plt.show()

#It can be determined that the variables affected by outliers are price, minimum_nights and number_of_reviews. We should set some upper and lower bounds to determine whether a data point should be considered an outlier.

#Outlier for price
price_stats = data["price"].describe()
price_stats

price_iqr = price_stats["75%"] - price_stats["25%"]
upper_limit = price_stats["75%"] + 1.5 * price_iqr
lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(price_iqr, 2)}")

data = data[data["price"] > 0]

#Outlier for minimum_nights
minimum_nights_stats = data["minimum_nights"].describe()
minimum_nights_stats

minimum_nights_iqr = minimum_nights_stats["75%"] - minimum_nights_stats["25%"]
upper_limit = minimum_nights_stats["75%"] + 1.5 * minimum_nights_iqr
lower_limit = minimum_nights_stats["25%"] - 1.5 * minimum_nights_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(minimum_nights_iqr, 2)}")

data = data[data["minimum_nights"] <= 14]

#Outlier for number_of_reviews
number_of_reviews_stats = data["number_of_reviews"].describe()
number_of_reviews_stats

number_of_reviews_iqr = number_of_reviews_stats["75%"] - number_of_reviews_stats["25%"]
upper_limit = number_of_reviews_stats["75%"] + 1.5 * number_of_reviews_iqr
lower_limit = number_of_reviews_stats["25%"] - 1.5 * number_of_reviews_iqr

print(f"The upper and lower limits for finding outliers are {round(upper_limit, 2)} and {round(lower_limit, 2)}, with an interquartile range of {round(number_of_reviews_iqr, 2)}")

data = data[data["number_of_reviews"] >= 0]

#Missing value analysis
data.isnull().sum().sort_values(ascending = False)

#Feature scaling
from sklearn.model_selection import train_test_split

num_variables = ["neighbourhood_group", "room_type", "minimum_nights", "number_of_reviews", "availability_365"]

# We divide the dataset into training and test samples
X = data.drop("price", axis = 1)[num_variables]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

#Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = num_variables)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = num_variables)

X_train_scal.head()

#Store the intermediate data in the ./data/interim folder
data.to_csv("../data/interim/data_day2.csv", index = False)

#Step 6: Feature selection
from sklearn.feature_selection import f_classif, SelectKBest

# With a value of k = 4 we implicitly mean that we want to remove 1 feature from the dataset
selection_model = SelectKBest(f_classif, k = 4)
selection_model.fit(X_train_scal, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_scal), columns = X_train_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_scal), columns = X_test_scal.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/AB_NYC_2019_train.csv", index=False)
X_test_sel.to_csv("/workspaces/machine-learning-python-template/data/processed/AB_NYC_2019_test.csv", index=False)