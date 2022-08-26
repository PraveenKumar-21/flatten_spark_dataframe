__version__ = "0.0.1"

from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def flatten(df, flatten_till_level='complete', exclude_list=[], _current_level=0):
    '''
    This function flattens a given spark dataframe
    All struct and array of struct columns will be flattened
    :param df: input dataframe
    :param flatten_till_level: by default it is complete meaning it will flatten all the nested levels, else we can mention integer value which specifies number of levels to flatten
    :param exclude_list: by default empty list, mention the list of columns that needs to be excluded
    '''
    object_name = f"flattening_temp_view"
    # get all flat_cols except ingest_ts as column_prefix is not needed for ingest_ts
    flat_cols = [f'{c[0]} AS {c[0]}' for c in df.dtypes if
                 (c[1][:6] != 'struct' and c[0] != 'ingest_ts' and c[1][:5] != 'array') or (
                         c[1][:5] == 'array' and not (
                     isinstance(df.schema[f"{c[0]}"].dataType.elementType, StructType)))]

    # get struct columns
    nested_cols = [c[0] for c in df.dtypes if
                   c[1][:6] == 'struct' and c[0].lower() not in exclude_list]
    nested_cols = [f"{nc}.`{c}` AS {nc}_" + re.sub('[\s+.]', '_', re.sub('[^a-zA-Z0-9._+ ]', '', c)) for
                   nc in nested_cols for c in df.select(nc + '.*').columns]

    # get struct columns that are in exclude list and melt_column_list, this wont be flattened
    nested_cols_exclude_col = [c[0] for c in df.dtypes if
                               c[1][:6] == 'struct' and c[0].lower() in exclude_list]

    # get array columns
    array_cols = [c[0] for c in df.dtypes if
                  c[1][:5] == 'array' and isinstance(df.schema[f"{c[0]}"].dataType.elementType, StructType) and c[
                      0].lower() not in exclude_list]
    array_cols_exclude_col = [c[0] for c in df.dtypes if
                              c[1][:5] == 'array' and isinstance(df.schema[f"{c[0]}"].dataType.elementType,
                                                                 StructType) and c[0].lower() in exclude_list]

    # find and replace duplicate cols if any
    nested_cols_alias_dup = [nc for nc in nested_cols if
                             f"{nc.split(' AS ')[-1]} AS {nc.split(' AS ')[-1]}" in flat_cols]

    # suffix the duplicate column with 1
    for nc in nested_cols_alias_dup:
        nested_cols[nested_cols.index(nc)] = f"{nc}1"

    if flatten_till_level == 'complete' or (flatten_till_level != 'complete' and flatten_till_level > _current_level):
        flatten_flag = True
    elif flatten_till_level == _current_level:
        flatten_flag = False

    print(f"flat_cols: {flat_cols}")
    print(f"nested_cols: {nested_cols}")
    print(f"array_cols: {array_cols}")

    df.createOrReplaceTempView(object_name)

    if flatten_flag and (len(nested_cols) or len(array_cols)):
        print(f"---------- Nested level: {_current_level + 1}  -------------------")

        sql_col_str = ','.join(array_cols_exclude_col + nested_cols_exclude_col + flat_cols + nested_cols)

        sql_stmt = f"select {sql_col_str} from {object_name}"
        # Generate the spark sql query to explode the array of struct columns one by one
        for ind, ac in enumerate(array_cols):
            temp_array_cols = array_cols[:ind] + array_cols[ind + 1:]
            if ind == 0:
                sql_col_str_l = sql_col_str.split(',') + temp_array_cols
                sql_stmt = f"select {','.join(list(set(sql_col_str_l)))}, explode_outer({ac}) AS {ac} from {object_name}"
            else:
                sql_col_str_ll = sql_col_str.split(',')
                sql_col_str_l = [col_str.split(' AS ')[-1].strip() for col_str in sql_col_str_ll] + temp_array_cols
                sql_stmt = f"select {','.join(list(set(sql_col_str_l)))}, explode_outer({ac}) AS {ac} from ({sql_stmt}) {object_name}{ind}"
            array_cols[ind] = f"{ac}"
        df = spark.sql(sql_stmt)
        return flatten(df, flatten_till_level, exclude_list, _current_level + 1)
    else:
        if _current_level == 0:
            df = spark.sql(f"select {','.join(flat_cols)} from {object_name}")
        return df
