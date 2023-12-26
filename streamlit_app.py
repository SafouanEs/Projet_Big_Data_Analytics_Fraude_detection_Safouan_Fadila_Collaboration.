import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_date, dayofweek, hour
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import random

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize a Spark session
spark = SparkSession.builder.appName("Fraud Detection").getOrCreate()

# Load the data
df = spark.read.csv('/content/corrected_enriched_financial_transactions_30k.csv', header=True, inferSchema=True)

"""

# 1.   Analyse Exploratoire des données:

---


"""

df.show(5)

df.printSchema()

#afficher le schéma d'un DataFrame
df.describe().show()

from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import DoubleType, FloatType
# Adjusted code to handle non-numeric columns properly
df.select([
    count(when(col(c).isNull(), c)).alias(c) if df.schema[c].dataType not in [DoubleType(), FloatType()]
    else count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns
]).show()

# Replace 'categorical_column' with your actual column names
df.groupBy('Transaction_Type').count().show()
df.groupBy('Merchant').count().show()
df.groupBy('Location').count().show()
df.groupBy('Entry_Mode').count().show()

from pyspark.mllib.stat import Statistics

# Select only numerical columns and convert to RDD
numerical_data = df.select(['Amount', 'User_ID','Account_Age','Previous_Transactions','Transaction_ID']).rdd.map(lambda row: row[0:])

# Calculate the correlation matrix
correlation_matrix = Statistics.corr(numerical_data, method="pearson")

# Print the correlation matrix
print(correlation_matrix)

"""# 2.Preprocessing:

---


"""

from pyspark.sql.functions import col, to_date, date_format
# Modify the 'Date' column to contain only the date part
df = df.withColumn("Date", to_date(col("Date")))

# Add a new 'Time' column extracting the time part in HH:mm:ss format
df = df.withColumn("Time", date_format(col("Date"), "HH:mm:ss"))

df.select("Date", "Time").show(5)

"""

* Define a UDF to Generate Random Times:

"""

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import random
def random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"

# Register the UDF
random_time_udf = udf(random_time, StringType())

df = df.withColumn("Time", random_time_udf())
df.select("Time").show(5)

from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour


# Extract hour from 'Time'
df = df.withColumn("Hour", hour(df["Time"]))

# Assuming 'Date' is in a standard format like 'YYYY-MM-DD'
df = df.withColumn("Year", year(col("Date")))
df = df.withColumn("Month", month(col("Date")))
df = df.withColumn("Day", dayofweek(col("Date")))

# Assuming 'Time' is in a standard format like 'HH:MM:SS'
df = df.withColumn("HourOfDay", hour(col("Time")))

indexers = [
    StringIndexer(inputCol=column, outputCol=column + "_Index").fit(df)
    for column in ["Transaction_Type", "Location", "Merchant"]
]

from pyspark.ml.feature import StandardScaler
categorical_columns = ["Location", "Merchant", "Transaction_Type"]
# Assume 'features' will include all feature columns after preprocessing
assembler = VectorAssembler(inputCols=["Amount", "Year", "Month", 'Day', "HourOfDay","Location_Index"] + [c + "_Index" for c in categorical_columns], outputCol="assembled_features")


# Scale features
scaler = StandardScaler(inputCol="assembled_features", outputCol="scaledFeatures")

# Convert 'Fraudulent' to double type for the classifier
df = df.withColumn("label", col("Fraudulent").cast(DoubleType()))

"""# construction du modèle:

---


"""

from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_date, dayofweek, hour
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import random

# Classifier
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label")
# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Split data
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Train the model
model = pipeline.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print("AUC:", auc)
# Initialize a Spark session
spark = SparkSession.builder.appName("FraudDetectionApp").getOrCreate()
model_path = "/content/STREAM.py"  # Changez le chemin selon vos besoins
model.write().overwrite().save(model_path)

def initialize_model(model_path):
    # Load the trained RandomForest model
    model = PipelineModel.load(model_path)
    return model

def predict_fraud(model, input_data):
    # Convert input data to Spark DataFrame
    # Adjust the columns based on your model's requirements
    columns = ["Amount", "Transaction_Type", "Location", ...]  # Add all necessary features
    data_df = spark.createDataFrame([input_data], columns)

    # Make predictions
    predictions = model.transform(data_df)
    return predictions

def main():
    st.title('Fraud Detection Application')

    model_path = st.sidebar.text_input("Enter the path to your RandomForest model")

    amount = st.number_input('Transaction Amount', min_value=0.0, format="%.2f")
    transaction_type = st.selectbox('Transaction Type', ['Deposit', 'Transfer', 'Payment', ...])  # Add all options
    location = st.text_input('Location')  # Adjust based on your features
    # ... Add more input fields as needed

    if st.button('Predict Fraud'):
        if model_path and amount and transaction_type and location:
            try:
                # Initialize and load the model
                model = initialize_model(model_path)

                # Prepare input data
                input_data = [amount, transaction_type, location, ...]  # Match the order and number of your features

                # Get predictions
                prediction = predict_fraud(model, input_data)

                # Extract and display the prediction result
                prediction_label = prediction.select('prediction').first()[0]
                st.write(f"The transaction is predicted to be: {'Fraudulent' if prediction_label == 1 else 'Not Fraudulent'}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please provide all required inputs.")

if __name__ == "__main__":
    main()
