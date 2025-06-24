import sqlite3 
import sqlite_vec 

# 连接到SQLite数据库
conn = sqlite3.connect('example.db')
cursor = conn.cursor() 

# 加载sqlite-vec扩展
conn.enable_load_extension(True)
sqlite_vec.load(conn)  # 创建使用vec0的虚拟表

# 创建表并插入数据
cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS vec_table USING vec0(vector_column float[3]);''')
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (1, '[0.1, 0.2, 0.3]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (2, '[0.3, 0.2, 0.3]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (3, '[0.2, 0.2, 0.3]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (4, '[0.4, 0.22, 0.5]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (5, '[0.13, 0.2, 0.7]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (6, '[0.51, 0.52, 0.7]')")
cursor.execute("INSERT INTO vec_table (rowid, vector_column) VALUES (7, '[0.61, 0.42, 0.5]')")

# 执行KNN查询
cursor.execute('''
    SELECT rowid, distance 
    FROM vec_table 
    WHERE vector_column MATCH '[0.4, 0.2, 0.7]' 
    ORDER BY distance 
    LIMIT 2;
''')

# 打印查询结果
print("KNN查询结果：")
for row in cursor.fetchall():
    print(f"rowid: {row[0]}, 距离: {row[1]}")

# 关闭连接
conn.close()