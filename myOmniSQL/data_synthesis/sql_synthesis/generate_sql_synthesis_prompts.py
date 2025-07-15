import json
import os
import random
import sqlite3
import sqlite_vec
import sqlite_lembed
import numpy as np
import traceback
import re
from typing import List, Dict

from tqdm import tqdm

sql_func_template = '''
### SQL Functions
You may consider one or more of the following SQL functions while generating the query:
{sql_funcs}
Important tips:
Except for the functions listed above, you may use any other functions as long as they conform to the syntax of the database engine.
'''

insert_stmts_template = '''
### INSERT INTO Statements
Below are several `INSERT INTO` statements. Use these to help generate predicates (i.e., `WHERE` clauses) in your SQL query:

{insert_statements}
'''

simple_criterion = '''**Criteria:**
Simple SQL queries may satisfy one or more of the following criteria:
- Simple queries should select data from a single table only.
- Basic aggregate functions are permitted, such as `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- No joins are allowed; the query must operate on a single table.

**Example of Simple SQL Query:**
```sql
SELECT name, department_name
FROM employees
WHERE level > 5
ORDER BY age DESC;
```'''

moderate_criterion = '''**Criteria:**
Moderate SQL queries may satisfy one or more of the following criteria:
- Involves table joins, such as `JOIN`, `INNER JOIN`, `LEFT JOIN`, `CROSS JOIN`, etc.
- Includes subqueries within the `SELECT` or `WHERE` clauses.
- Utilizes aggregate functions alongside a `GROUP BY` clause.
- Contains complex `WHERE` conditions, including `IN`, `BETWEEN`, `LIKE`.
- Incorporate a `HAVING` clause to filter aggregated results.
- Uses aggregate functions like `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, etc.

**Example of Moderate SQL Query:**
```sql
SELECT e.name, d.department_name, AVG(s.salary) AS average_salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
LEFT JOIN salaries s ON e.employee_id = s.employee_id
WHERE e.age > 30 AND e.status = 'active'
GROUP BY e.name, d.department_name
HAVING AVG(s.salary) > 50000;
```'''

complex_criterion = '''**Criteria:**
Complex SQL queries may satisfy one or more of the following criteria:
- Contains complex nested subqueries.
- Utilizes multiple types of joins, including self-joins.
- Includes window functions, such as `ROW_NUMBER`, `RANK`, etc.
- Uses Common Table Expressions (CTEs) for improved readability.
- Combines multiple aggregate functions.
- Involves complex `WHERE` and `HAVING` clauses with multiple conditions.
- Utilizes advanced functions and operators.

**Example of Complex SQL Query:**
```sql
WITH EmployeeCTE AS (
    SELECT employee_id, name, department_id, ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT e.name, d.department_name
FROM EmployeeCTE e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE e.rank <= 3;
```'''

highly_complex_criterion = '''**Criteria:**
Highly complex SQL queries may satisfy one or more of the following criteria:
- Includes multiple Common Table Expressions (CTEs) for readability.
- Combines nested subqueries and various joins.
- Utilizes recursive CTEs for hierarchical or recursive queries.
- Extensively uses advanced window functions.
- May involve `UNION` or `UNION ALL` to combine result sets.
- Implements complex logic with advanced analytical functions.
- Employs a wide range of SQL clauses and conditions.
- Utilizes a broad spectrum of SQL functions and advanced features.

**Example of Highly Complex SQL Query:**
```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT employee_id, name, manager_id, department_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.name, e.manager_id, e.department_id, eh.level + 1
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
),
DepartmentSalaries AS (
    SELECT eh.employee_id, eh.name, eh.level, d.department_name, s.salary, d.department_id
    FROM EmployeeHierarchy eh
    INNER JOIN departments d ON eh.department_id = d.department_id
    INNER JOIN salaries s ON eh.employee_id = s.employee_id
),
DepartmentStats AS (
    SELECT 
        d.department_id,
        COUNT(e.employee_id) AS employee_count,
        AVG(s.salary) AS average_salary
    FROM employees e
    INNER JOIN salaries s ON e.employee_id = s.employee_id
    INNER JOIN departments d ON e.department_id = d.department_id
    GROUP BY d.department_id
)
SELECT ds.name, ds.level, 
    SUM(ds.salary) OVER (PARTITION BY ds.department_id ORDER BY ds.level, ds.name) AS cumulative_salary
FROM DepartmentSalaries ds
INNER JOIN DepartmentStats dstat ON ds.department_id = dstat.department_id
ORDER BY ds.level, ds.name;
```'''

simple_vec_criterion = '''**Criteria:**
Simple KNN queries in SQLite-vec may satisfy one or more of the following criteria:
- Basic vector similarity search on a single table
- Uses simple `MATCH` operator with target vector
- Contains basic `LIMIT` or `AND` clause to restrict results after `MATCH` operator
- No joins or complex filtering beyond the vector search

**Example of Simple KNN Query:**
```sql
SELECT rowid, location_embedding 
FROM vec_table 
WHERE location_embedding MATCH lembed('all-MiniLM-L6-v2',"572 Main Street Los Angeles, CA 90210 USA")
ORDER BY distance 
LIMIT 1;
```'''

moderate_vec_criterion = '''**Criteria:**
Moderate KNN queries in SQLite-vec may satisfy one or more of the following criteria:
- Includes simple joins with metadata tables
- Contains basic post-filtering of vector results
- May use multiple vector columns in query

**Example of Moderate KNN Query:**
```sql
SELECT d.doc_id, d.title, d.content
FROM documents d
JOIN categories c ON d.category_id = c.id
WHERE d.content_embedding MATCH lembed('all-MiniLM-L6-v2',"OmniSQL is a unified SQL engine that integrates Vector search and LLM augmentation.")
AND k = 2
AND c.name = 'science'
ORDER BY d.distance;
```'''

complex_vec_criterion = '''**Criteria:**
Complex KNN queries in SQLite-vec may satisfy one or more of the following criteria:
- Combines vector search with complex joins
- Uses CTEs to organize vector search logic
- Contains hybrid search (vector + full-text)
- Implements multi-stage filtering of results
- May use window functions with vector results
- Includes complex distance threshold conditions

**Example of Complex KNN Query:**
```sql
"WITH HighWDVOATeams AS (\n    SELECT team_id, team_name\n    FROM teams\n    WHERE team_id IN (\n        SELECT team_id\n        FROM team_metrics\n        WHERE wdvoa > 30 AND season = 2019\n    )\n),\nSimilarTeams AS (\n    SELECT team_id, team_name, distance\n    FROM teams\n    WHERE team_name_embedding MATCH lembed('all-MiniLM-L6-v2',"Woven Shadows")\n    ORDER BY distance\n    LIMIT 5\n)\nSELECT h.team_name, AVG(p.confidence_level) AS average_confidence\nFROM HighWDVOATeams h\nJOIN SimilarTeams s ON h.team_id = s.team_id\nJOIN game_predictions p ON p.game_id IN (\n    SELECT game_id\n    FROM games\n    WHERE home_team_id = h.team_id OR away_team_id = h.team_id\n)\nGROUP BY h.team_name\nHAVING average_confidence > 0.7;"
```'''

highly_complex_vec_criterion = '''**Criteria:**
Highly complex KNN queries in SQLite-vec may satisfy one or more of the following criteria:
- Uses multiple CTEs with vector operations
- Combines multiple vector searches in one query
- Implements advanced hybrid search techniques
- Contains recursive vector search patterns
- Uses complex window functions over vector results
- May involve vector aggregation operations
- Implements custom distance calculations

**Example of Highly Complex KNN Query:**
```sql
WITH BettingAnalysis AS (\n    SELECT \n        g.game_id,\n        AVG(bd.betting_spread) AS avg_initial_spread,\n        COUNT(*) AS total_bets\n    FROM games g\n    JOIN betting_data bd ON g.game_id = bd.game_id\n    GROUP BY g.game_id\n),\nPredictionAnalysis AS (\n    SELECT \n        gp.game_id,\n        AVG(gp.confidence_level) AS avg_confidence,\n        SUM(CASE WHEN gp.make_pick = 1 AND g.pick_right = 1 THEN 1 ELSE 0 END) AS correct_predictions\n    FROM games g\n    JOIN game_predictions gp ON g.game_id = gp.game_id\n    GROUP BY gp.game_id\n),\nTeamPerformance AS (\n    SELECT \n        tm.team_id,\n        tm.season,\n        AVG(tm.wdvoa) AS avg_wdvoa\n    FROM team_metrics tm\n    GROUP BY tm.team_id, tm.season\n),\nLocationSimilarity AS (\n    SELECT \n        g.game_id,\n        g.location,\n        vec.distance AS location_similarity\n    FROM games g\n    JOIN (\n        SELECT rowid, distance \n        FROM games \n        WHERE location_embedding MATCH lembed('all-MiniLM-L6-v2',"New York") \n        ORDER BY distance \n        LIMIT 5\n    ) AS vec ON g.rowid = vec.rowid\n)\nSELECT \n    g.game_id,\n    ba.avg_initial_spread,\n    pa.avg_confidence,\n    tp.avg_wdvoa,\n    ls.location_similarity\nFROM games g\nJOIN BettingAnalysis ba ON g.game_id = ba.game_id\nJOIN PredictionAnalysis pa ON g.game_id = pa.game_id\nJOIN TeamPerformance tp ON g.home_team_id = tp.team_id\nJOIN LocationSimilarity ls ON g.game_id = ls.game_id\nWHERE pa.correct_predictions > 2\nORDER BY g.game_id;
```'''


def contains_virtual_table(statements):
    """
    检测 statements 中是否包含虚拟表
    
    参数:
        statements: 包含 SQL 语句的列表或字符串
    
    返回:
        bool: 如果检测到虚拟表返回 True，否则返回 False
    """
    if isinstance(statements, str):
        statements = [statements]
    
    # 定义虚拟表关键词模式（不区分大小写）
    patterns = [
        r'\bvirtual\b',  # 匹配 "virtual" 单词
        r'\bvec0\b',     # 匹配 "vec0" 单词
        r'_embedding\b',  # 匹配以 "_embedding" 结尾的单词
        r'\bfloat\[',    #匹配类型
        r'\]\b'
    ]
    
    for stmt in statements:
        if not stmt:
            continue
            
        # 检查每个关键词模式
        for pattern in patterns:
            if re.search(pattern, stmt, re.IGNORECASE):
                return True
                
    return False

def obtain_db_schema(db_file_dir):
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 
    sqlite_lembed.load(conn)

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_names = []
    create_statements = []
    for table in tables:
        table_name, create_statement = table
        table_names.append(table_name)
        create_statements.append(create_statement)

    cursor.close()
    conn.close()

    return table_names, create_statements

def obtain_insert_statements(db_file_dir, table_names):
    table_name2insert_statements = dict()
    conn = sqlite3.connect(db_file_dir)
    cursor = conn.cursor()

    # load sqlite-vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn) 
    sqlite_lembed.load(conn)

    for table_name in table_names:
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 2')
            rows = cursor.fetchall()

            column_names = [description[0] for description in cursor.description]

            insert_statements = []
            for row in rows:
                values = ', '.join([f"'{str(value)}'" if isinstance(value, str) else str(value) for value in row])
                insert_statement = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({values});"
                insert_statements.append(insert_statement)

            # for statement in insert_statements:
            #     print(statement)
            table_name2insert_statements[table_name] = insert_statements

        except Exception as e:
            print(e)
            traceback.print_exc()


    cursor.close()
    conn.close()

    return table_name2insert_statements

def write_large_json(data: List[Dict], output_path: str, chunk_size: int = 500):
    """分块写入字典数组到 JSON 文件（避免嵌套数组）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('[')  # 开始 JSON 数组
        
        # 写入第一个元素（避免开头多余逗号）
        if len(data) > 0:
            json.dump(data[0], f, ensure_ascii=False, indent=None)
        
        # 分块写入剩余元素
        for i in range(1, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            f.write(',\n')  # 添加分隔符
            # 逐元素写入（而非整个 chunk）
            for j, item in enumerate(chunk):
                if j > 0:
                    f.write(',')
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        f.write(']')  # 结束 JSON 数组

if __name__ == "__main__":
    random.seed(42)
    db_path = "../database_synthesis/synthetic_sqlite_databases"
    prompt_template = open("./prompt_templates/sql_synthesis_prompt.txt", "r", encoding = "utf-8").read()
    functions = json.load(open("./prompt_templates/sqlite_funcs.json"))

    complexity2criterion = {}

    db_names = os.listdir(db_path)
    prompts = []
    for db_name in tqdm(db_names):
        try:
            db_file_dir = os.path.join(db_path, db_name, db_name + ".sqlite")
            table_names, create_statements = obtain_db_schema(db_file_dir)
            if contains_virtual_table(create_statements):
                complexity2criterion = {
                    "Simple": simple_vec_criterion,
                    "Moderate": moderate_vec_criterion,
                    "Complex": complex_vec_criterion, 
                    "Highly Complex": highly_complex_vec_criterion
                }
            else:
                complexity2criterion = {
                    "Simple": simple_criterion,
                    "Moderate": moderate_criterion,
                    "Complex": complex_criterion, 
                    "Highly Complex": highly_complex_criterion
                }
            # print("states: ")
            # print(create_statements)
            # print("\n\n")
            table_name2insert_statements = obtain_insert_statements(db_file_dir, table_names)

            for _ in range(0, 12):
                complexity = random.sample(["Simple", "Moderate", "Complex", "Highly Complex"], 1)[0] 

                insert_statements = []
                for table_name in table_names:
                    insert_statements += table_name2insert_statements.get(table_name, [])
                
                if len(insert_statements) == 0:
                    db_value_prompt = ""
                else:
                    if len(insert_statements) > 4:
                        insert_statements = random.sample(insert_statements, 4)
                    db_value_prompt = insert_stmts_template.format(insert_statements = "\n\n".join(insert_statements))

                function_num = random.randint(0, 2)
                if function_num == 0:
                    sql_function_prompt = "### SQL Functions\nYou can use any function supported by the database engine."
                else:
                    sql_funcs = ""
                    sampled_functions = random.sample(functions, function_num)
                    for idx, func in enumerate(sampled_functions):
                        sql_funcs += f"Function {idx + 1}:\n" + func.strip() + "\n"
                    sql_function_prompt = sql_func_template.format(sql_funcs = sql_funcs)

                column_count = np.random.geometric(0.6, 1)[0]

                # 在模板中替换 {embedding_model} 占位符（可以多次出现）
                # embedding_model = "all-MILM-L6-v2"
                # question_synthesis_template = question_synthesis_template.replace("{embedding_model}", embedding_model)
                prompt = prompt_template.format(
                    schema_str = "\n\n".join(create_statements),
                    sql_function_prompt = sql_function_prompt.strip(),
                    db_value_prompt = db_value_prompt.strip(),
                    complexity = complexity,
                    criterion = complexity2criterion[complexity].strip(),
                    db_engine = "SQLite",
                    column_count = column_count,
                    db_extension = "SQLite-vec and sqlite-lembed",
                    embedding_model="all-MILM-L6-v2"
                )

                prompts.append({"prompt": prompt, "db_id": db_name})

        except Exception as e:
            print(e)
            traceback.print_exc()

    # random.shuffle(prompts)
    # 定义文件夹路径
    prompts_dir = "./prompts"
    output_path = "./prompts/sql_synthesis_prompts.json"

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(prompts_dir):
        try:
            os.makedirs(prompts_dir)
            print(f"成功创建文件夹: {prompts_dir}")
        except OSError as e:
            print(f"创建文件夹失败: {e}")
    else:
        print(f"文件夹已存在: {prompts_dir}")
    # with open("./prompts/sql_synthesis_prompts.json", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(prompts, indent=2, ensure_ascii=False))
    
    write_large_json(prompts, output_path, 500)
