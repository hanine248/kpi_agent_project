import pyodbc

conn_str = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=DESKTOP-LI1DPPU\SQLEXPRESS;"
    "Database=EmployeePerformanceDB;"
    "UID=kpi_service_account;"
    "PWD=kpi_service_account;"
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Companies")
    count = cursor.fetchone()[0]
    print(f"Connection successful! Found {count} companies.")
    conn.close()
except Exception as e:
    print(f"Connection failed: {str(e)}")