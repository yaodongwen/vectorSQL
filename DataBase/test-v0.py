import sqlite3 
import sqlite_vec 

# 连接到SQLite数据库
conn = sqlite3.connect('example.db')
cursor = conn.cursor() 

# 加载sqlite-vec扩展
conn.enable_load_extension(True)
sqlite_vec.load(conn)  # 创建使用vec0的虚拟表

cursor.execute("DROP TABLE IF EXISTS vec_table")

# 创建表并插入数据
cursor.execute('''CREATE virtual TABLE "vec_table" using vec0(vec float[3], num INTEGER);''')
# cursor.execute("PRAGMA table_info(vec_table)")
# print(cursor.fetchall())
# print("--------")
cursor.execute("INSERT INTO vec_table (rowid, vec, num) VALUES (1, '[0.1, 0.2, 0.3]', 2)")
cursor.execute("INSERT INTO vec_table (rowid, vec, num) VALUES (2, '[0.4, 0.2, 0.3]', 22)")
cursor.execute("INSERT INTO vec_table (rowid, vec, num) VALUES (3, '[0.5, 0.4, 0.7]', 7)")
# cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS "affected_entities" USING vec0(
#   impact_embedding float[3] /* Embedding vector for the impact description */,
#   entity_id INTEGER /* Unique identifier for each affected entity */,
#   breach_id FLOAT /* Identifier for the data breach affecting the entity */,
#   entity_name TEXT /* Name of the affected entity */,
#   impact TEXT /* Description of the impact on the entity */,
#   industry TEXT /* Industry of the affected entity */,
#   notification_date TEXT /* Date when the entity was notified */
# );''')
# cursor.execute("""alter TABLE affected_entities MODIFY entity_id INTEGER PRIMARY KEY""")

# cursor.execute("INSERT INTO affected_entities (vec, num) VALUES ('[0.1, 0.2, 0.3]',45)")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (1, '[0.1, 0.2, 0.3]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (2, '[0.3, 0.2, 0.3]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (3, '[0.2, 0.2, 0.3]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (4, '[0.4, 0.22, 0.5]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (5, '[0.13, 0.2, 0.7]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (6, '[0.51, 0.52, 0.7]')")
# cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (7, '[0.61, 0.42, 0.5]')")

# 执行KNN查询
cursor.execute('''
    SELECT rowid, vec 
    FROM vec_table 
    WHERE vec MATCH '[0.4, 0.2, 0.7]' 
    ORDER BY distance 
    LIMIT 2;
''')

cursor.execute('''
    SELECT *
    FROM "vec_table" limit 2
''')

# 打印查询结果
# print("KNN查询结果：")
for row in cursor.fetchall():
    print(row)
    print("------------")
    # print(f"rowid: {row[0]}, vec: {row[1]}, num: {row[2]}")

# 关闭连接
conn.close()