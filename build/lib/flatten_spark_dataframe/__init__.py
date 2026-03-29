__version__ = "0.0.2"

from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
from pyspark.sql import SparkSession

# Initialize a Spark session (reuses existing one if available)
spark = SparkSession.builder.getOrCreate()


def flatten(df, flatten_till_level='complete', exclude_list=[], _current_level=0):
    '''
    Recursively flattens a nested Spark DataFrame by expanding StructType and
    ArrayType(StructType) columns into top-level columns.

    Each recursion handles one level of nesting. The function keeps calling itself
    until all nested levels are flattened (or until the specified depth is reached).

    :param df: Input Spark DataFrame (potentially with nested columns)
    :param flatten_till_level: 'complete' to flatten all levels, or an int to limit depth
                               (e.g., 2 means flatten only 2 levels deep)
    :param exclude_list: List of column names (lowercase) to skip during flattening.
                         These columns are kept as-is in the output.
    :param _current_level: Internal counter tracking the current recursion depth.
                           Callers should NOT set this manually.
    '''

    # Temp view name used to run Spark SQL queries against the DataFrame
    object_name = f"flattening_temp_view"

    # -----------------------------------------------------------------------
    # Step 1: Classify columns into categories
    # -----------------------------------------------------------------------

    # FLAT COLUMNS: Primitive types + arrays of primitives (not arrays of structs).
    # These don't need flattening and are carried forward as-is.
    # Excludes 'ingest_ts' from aliasing (treated specially).
    flat_cols = [f'{c[0]} AS {c[0]}' for c in df.dtypes if
                 (c[1][:6] != 'struct' and c[0] != 'ingest_ts' and c[1][:5] != 'array') or (
                         c[1][:5] == 'array' and not (
                     isinstance(df.schema[f"{c[0]}"].dataType.elementType, StructType)))]

    # NESTED (STRUCT) COLUMNS: Columns of type StructType that are NOT in the exclude list.
    # First, collect the struct column names...
    nested_cols = [c[0] for c in df.dtypes if
                   c[1][:6] == 'struct' and c[0].lower() not in exclude_list]
    # ...then expand each struct into its sub-fields, aliased as "parent_child".
    # Special characters are stripped and spaces/dots replaced with underscores
    # to produce clean column names (e.g., address.street_name → address_street_name).
    nested_cols = [f"{nc}.`{c}` AS {nc}_" + re.sub('[\s+.]', '_', re.sub('[^a-zA-Z0-9._+ ]', '', c)) for
                   nc in nested_cols for c in df.select(nc + '.*').columns]

    # EXCLUDED STRUCT COLUMNS: Struct columns the user wants to keep nested (not flattened).
    nested_cols_exclude_col = [c[0] for c in df.dtypes if
                               c[1][:6] == 'struct' and c[0].lower() in exclude_list]

    # ARRAY OF STRUCT COLUMNS: Array columns whose element type is StructType (not in exclude list).
    # These will be exploded (one row per array element) and then their struct fields
    # will be flattened in subsequent recursions.
    array_cols = [c[0] for c in df.dtypes if
                  c[1][:5] == 'array' and isinstance(df.schema[f"{c[0]}"].dataType.elementType, StructType) and c[
                      0].lower() not in exclude_list]

    # EXCLUDED ARRAY COLUMNS: Array-of-struct columns the user wants to keep as-is.
    array_cols_exclude_col = [c[0] for c in df.dtypes if
                              c[1][:5] == 'array' and isinstance(df.schema[f"{c[0]}"].dataType.elementType,
                                                                 StructType) and c[0].lower() in exclude_list]

    # -----------------------------------------------------------------------
    # Step 2: Handle duplicate column names
    # -----------------------------------------------------------------------
    # If a flattened struct sub-field has the same name as an existing flat column,
    # detect the collision and append "1" to the struct sub-field alias to avoid ambiguity.
    nested_cols_alias_dup = [nc for nc in nested_cols if
                             f"{nc.split(' AS ')[-1]} AS {nc.split(' AS ')[-1]}" in flat_cols]

    for nc in nested_cols_alias_dup:
        nested_cols[nested_cols.index(nc)] = f"{nc}1"

    # -----------------------------------------------------------------------
    # Step 3: Determine whether to continue flattening at this level
    # -----------------------------------------------------------------------
    # flatten_flag = True  → there are still nested columns and we haven't hit the depth limit
    # flatten_flag = False → we've reached the requested depth, stop recursing
    if flatten_till_level == 'complete' or (flatten_till_level != 'complete' and flatten_till_level > _current_level):
        flatten_flag = True
    elif flatten_till_level == _current_level:
        flatten_flag = False

    # Debug prints showing what was detected at this level
    print(f"flat_cols: {flat_cols}")
    print(f"nested_cols: {nested_cols}")
    print(f"array_cols: {array_cols}")

    # Register the DataFrame as a temp SQL view so we can query it with Spark SQL
    df.createOrReplaceTempView(object_name)

    # -----------------------------------------------------------------------
    # Step 4: Build and execute the flattening SQL query
    # -----------------------------------------------------------------------
    if flatten_flag and (len(nested_cols) or len(array_cols)):
        # There are still nested/array columns to flatten at this level
        print(f"---------- Nested level: {_current_level + 1}  -------------------")

        # Combine all column expressions: excluded cols (kept as-is) + flat cols + expanded struct fields
        sql_col_str = ','.join(array_cols_exclude_col + nested_cols_exclude_col + flat_cols + nested_cols)

        # Start with a simple SELECT of all non-array columns
        sql_stmt = f"select {sql_col_str} from {object_name}"

        # For each array-of-struct column, wrap the query with explode_outer().
        # explode_outer() creates one row per array element (NULL row if array is empty/null).
        # Multiple array columns are exploded sequentially — each explode wraps the previous query
        # as a subquery, producing a cross-join effect across all array columns.
        for ind, ac in enumerate(array_cols):
            # All other array columns except the current one (carried forward unexploded for now)
            temp_array_cols = array_cols[:ind] + array_cols[ind + 1:]
            if ind == 0:
                # First array column: explode directly from the temp view
                sql_col_str_l = sql_col_str.split(',') + temp_array_cols
                sql_stmt = f"select {','.join(list(set(sql_col_str_l)))}, explode_outer({ac}) AS {ac} from {object_name}"
            else:
                # Subsequent array columns: wrap the previous SQL as a subquery and explode the next array
                sql_col_str_ll = sql_col_str.split(',')
                # Use only the alias names (after AS) since the subquery already resolved full expressions
                sql_col_str_l = [col_str.split(' AS ')[-1].strip() for col_str in sql_col_str_ll] + temp_array_cols
                sql_stmt = f"select {','.join(list(set(sql_col_str_l)))}, explode_outer({ac}) AS {ac} from ({sql_stmt}) {object_name}{ind}"
            array_cols[ind] = f"{ac}"

        # Execute the constructed SQL to produce a partially-flattened DataFrame
        df = spark.sql(sql_stmt)

        # Recurse to flatten the next level of nesting
        return flatten(df, flatten_till_level, exclude_list, _current_level + 1)
    else:
        # BASE CASE: No more nested/array columns to flatten, or depth limit reached.
        if _current_level == 0:
            # If the input had no nesting at all, just return flat columns
            return spark.sql(f"select {','.join(flat_cols)} from {object_name}")
        # Otherwise return all columns from the fully-flattened DataFrame
        return spark.sql(f"select * from {object_name}")
