import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
# Initialize a local Spark Session
spark = (
    SparkSession.builder.appName("TestSetup").master("local[*]").getOrCreate()
)

# Create a tiny dataset in memory
data = [("Setup", "Successful"), ("PySpark", "Working")]
df = spark.createDataFrame(data, ["Task", "Status"])

# Action: Print the data to the console
print("\n--- Spark Output ---")
df.show()
print("--------------------\n")

# Clean up
spark.stop()
