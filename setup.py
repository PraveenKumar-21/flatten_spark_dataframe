import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flatten_spark_dataframe",
    version="0.0.1",
    author="Praveen Kumar B",
    author_email="bpraveenkumar21@gmail.com",
    description="Databricks PySpark module to flatten nested spark dataframes, basically struct and array of struct till the specified level",
    keywords = ['PySpark flatten dataframe', 'Databricks PySpark flatten dataframe', 'PySpark nested dataframe', 'databricks pyspark nested dataframe', 'flatten dataframe', 'nested dataframe'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PraveenKumar-21/flatten_spark_dataframe",
    packages=setuptools.find_packages(),
    install_requires=[
        'pyspark',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)