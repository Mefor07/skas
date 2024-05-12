import mysql.connector

# Replace the placeholders with your actual database information
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='game_dev'
)

cursor = connection.cursor()

# Example: Execute a simple query
cursor.execute("SELECT * FROM games")

# Example: Fetch all rows
rows = cursor.fetchall()
# for row in rows:
#     print(row)

file_path = "static/docs/skas.txt"
with open(file_path, "r") as file:
    sample_text = file.read()
    print(sample_text)


