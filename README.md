# Flatten Pyspark dataframe
### This module flattens a given spark dataframe
### All struct and array of struct columns will be flattened

###### Sample input pyspark dataframe
```
data = [
        ((("A","James"),None,"Smith"),"OH","M",("F","Mike")),
        ((("B","Anna"),"Rose",""),"NY","F",("E","Jen")),
        ((("C","Julia"),"","Williams"),"OH","F",("D","Maria")),
        ((("D","Maria"),"Anne","Jones"),"NY","M",("C","Julia")),
        ((("E","Jen"),"Mary","Brown"),"NY","M",("B","Anna")),
        ((("F","Mike"),"Mary","Williams"),"OH","M",("A","James"))
        ]

from pyspark.sql.types import StructType,StructField, StringType        
schema = StructType([
    StructField('name', StructType([
         StructField('firstname', StructType([
         StructField('initial', StringType(), True),
         StructField('actualname', StringType(), True)])),
         StructField('middlename', StringType(), True),
         StructField('lastname', StringType(), True)
         ])),
     StructField('state', StringType(), True),
     StructField('gender', StringType(), True),
	 StructField('country', StructType([
         StructField('city', StringType(), True),
         StructField('street', StringType(), True)])),
     ])
dfn = spark.createDataFrame(data = data, schema = schema)

```
```
import flatten_spark_dataframe
flattened_dataframe = flatten_spark_dataframe.flatten(dfn)
```
### Output text
```
flat_cols: ['state AS state', 'gender AS gender']
nested_cols: ['name.`firstname` AS name_firstname', 'name.`middlename` AS name_middlename', 'name.`lastname` AS name_lastname', 'country.`city` AS country_city', 'country.`street` AS country_street']
array_cols: []
---------- Nested level: 1  -------------------
flat_cols: ['state AS state', 'gender AS gender', 'name_middlename AS name_middlename', 'name_lastname AS name_lastname', 'country_city AS country_city', 'country_street AS country_street']
nested_cols: ['name_firstname.`initial` AS name_firstname_initial', 'name_firstname.`actualname` AS name_firstname_actualname']
array_cols: []
---------- Nested level: 2  -------------------
flat_cols: ['state AS state', 'gender AS gender', 'name_middlename AS name_middlename', 'name_lastname AS name_lastname', 'country_city AS country_city', 'country_street AS country_street', 'name_firstname_initial AS name_firstname_initial', 'name_firstname_actualname AS name_firstname_actualname']
nested_cols: []
array_cols: []
```
```
flattened_dataframe.take(10)
```
```
[Row(state='OH', gender='M', name_middlename=None, name_lastname='Smith', country_city='F', country_street='Mike', name_firstname_initial='A', name_firstname_actualname='James'),
 Row(state='NY', gender='F', name_middlename='Rose', name_lastname='', country_city='E', country_street='Jen', name_firstname_initial='B', name_firstname_actualname='Anna'),
 Row(state='OH', gender='F', name_middlename='', name_lastname='Williams', country_city='D', country_street='Maria', name_firstname_initial='C', name_firstname_actualname='Julia'),
 Row(state='NY', gender='M', name_middlename='Anne', name_lastname='Jones', country_city='C', country_street='Julia', name_firstname_initial='D', name_firstname_actualname='Maria'),
 Row(state='NY', gender='M', name_middlename='Mary', name_lastname='Brown', country_city='B', country_street='Anna', name_firstname_initial='E', name_firstname_actualname='Jen'),
 Row(state='OH', gender='M', name_middlename='Mary', name_lastname='Williams', country_city='A', country_street='James', name_firstname_initial='F', name_firstname_actualname='Mike')]
```
