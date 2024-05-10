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
for row in rows:
    print(row)


