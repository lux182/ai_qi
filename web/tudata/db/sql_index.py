import re


def get_create_table_string(tablename, connection):
    sql = """
    select * from sqlite_master where name = "{}" and type = "table"
    """.format(tablename)
    result = connection.execute(sql)

    create_table_string = result.fetchmany()[0][4]
    return create_table_string


def add_pk_to_create_table_string(create_table_string, colname):
    regex = "(\n.+{}[^,]+)(,)".format(colname)
    return re.sub(regex, "\\1 PRIMARY KEY,", create_table_string, count=1)


def add_pk_to_sqlite_table(tablename, index_column, connection):
    cts = get_create_table_string(tablename, connection)
    cts = add_pk_to_create_table_string(cts, index_column)
    template = """
    BEGIN TRANSACTION;
        ALTER TABLE {tablename} RENAME TO {tablename}_old_;

        {cts};

        INSERT INTO {tablename} SELECT * FROM {tablename}_old_;

        DROP TABLE {tablename}_old_;

    COMMIT TRANSACTION;
    """

    create_and_drop_sql = template.format(tablename=tablename, cts=cts)
    connection.executescript(create_and_drop_sql)


# Example:

# import pandas as pd
# import sqlite3

# df = pd.DataFrame({"a": [1,2,3], "b": [2,3,4]})
# con = sqlite3.connect("deleteme.db")
# df.to_sql("df", con, if_exists="replace")

# add_pk_to_sqlite_table("df", "index", con)
# r = con.execute("select sql from sqlite_master where name = 'df' and type = 'table'")
# print(r.fetchone()[0])