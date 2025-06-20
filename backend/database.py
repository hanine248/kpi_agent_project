import pyodbc

# Replace with your actual SQL Server settings
def get_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=DESKTOP-LI1DPPU\SQLEXPRESS;'
        'DATABASE=HR_DB;'
        'Trusted_Connection=yes;'  # if using Windows auth
    )
    return conn

def get_all_employees():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Employees")

    rows = cursor.fetchall()
    employees = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]
    conn.close()
    return employees

def get_employee_by_id(emp_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Employees WHERE ID = ?", emp_id)
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(zip([column[0] for column in cursor.description], row))
    return None
