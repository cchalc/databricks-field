# Databricks notebook source
CATALOG_NAME = "cjc"
SCHEMA_NAME = "ml_serv"

spark.sql("CREATE CATALOG IF NOT EXISTS cjc")
spark.sql("CREATE SCHEMA IF NOT EXISTS cjc.ml_serv")
spark.sql("CREATE VOLUME IF NOT EXISTS cjc.ml_serv.myc")
spark.sql("use cjc.ml_serv")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("T-SQL Queries").getOrCreate()

# Define schema
schema = StructType([
    StructField("query_id", IntegerType(), False),
    StructField("query_description", StringType(), False),
    StructField("t_sql_query", StringType(), False)
])

# List of tuples with query descriptions and corresponding SQL queries
queries = [
    # Refined Join Queries
    ("Select employees and their department names using INNER JOIN",
     "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e INNER JOIN Departments d ON e.DepartmentID = d.DepartmentID;"),
    
    ("Select employees and their department names using LEFT JOIN",
     "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e LEFT JOIN Departments d ON e.DepartmentID = d.DepartmentID;"),
    
    ("Select all departments, even those without employees, using RIGHT JOIN",
     "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e RIGHT JOIN Departments d ON e.DepartmentID = d.DepartmentID;"),
    
    ("Select all employees and all departments using FULL OUTER JOIN",
     "SELECT e.FirstName, e.LastName, d.DepartmentName FROM Employees e FULL OUTER JOIN Departments d ON e.DepartmentID = d.DepartmentID;"),
    
    ("Select employees with their projects using a CROSS JOIN",
     "SELECT e.FirstName, p.ProjectName FROM Employees e CROSS JOIN Projects p;"),
    
    ("Select employees with matching salary ranges using a SELF JOIN",
     "SELECT e1.FirstName, e2.FirstName FROM Employees e1 INNER JOIN Employees e2 ON e1.Salary BETWEEN e2.Salary - 1000 AND e2.Salary + 1000;"),
    
    ("Select employees who work in the same department using a SELF JOIN",
     "SELECT e1.FirstName, e2.FirstName, d.DepartmentName FROM Employees e1 INNER JOIN Employees e2 ON e1.DepartmentID = e2.DepartmentID INNER JOIN Departments d ON e1.DepartmentID = d.DepartmentID;"),
    
    ("Select departments and count of employees using GROUP BY with JOIN",
     "SELECT d.DepartmentName, COUNT(e.EmployeeID) AS EmployeeCount FROM Employees e INNER JOIN Departments d ON e.DepartmentID = d.DepartmentID GROUP BY d.DepartmentName;"),
    
    ("Select departments with more than 5 employees using HAVING with JOIN",
     "SELECT d.DepartmentName, COUNT(e.EmployeeID) AS EmployeeCount FROM Employees e INNER JOIN Departments d ON e.DepartmentID = d.DepartmentID GROUP BY d.DepartmentName HAVING COUNT(e.EmployeeID) > 5;"),
    
    ("Select projects assigned to employees using LEFT JOIN and ISNULL",
     "SELECT e.FirstName, e.LastName, ISNULL(p.ProjectName, 'No Project') AS ProjectName FROM Employees e LEFT JOIN Projects p ON e.EmployeeID = p.EmployeeID;"),
    
    # T-SQL specific advanced queries
    ("Select top 5 records", "SELECT TOP 5 * FROM Employees;"),
    
    ("Create a temporary table", "CREATE TABLE #TempEmployees (EmployeeID INT, FirstName VARCHAR(50), LastName VARCHAR(50));"),
    
    ("Insert into a temp table", "INSERT INTO #TempEmployees (EmployeeID, FirstName, LastName) VALUES (1, 'John', 'Doe');"),
    
    ("Use ISNULL function", "SELECT ISNULL(ManagerID, 'No Manager') FROM Employees;"),
    
    ("String concatenation using +", "SELECT FirstName + ' ' + LastName AS FullName FROM Employees;"),
    
    ("Coalesce function example", "SELECT COALESCE(ManagerID, SupervisorID, 'No Supervisor') FROM Employees;"),
    
    ("Using CASE statement", "SELECT EmployeeID, CASE WHEN Department = 'HR' THEN 'Human Resources' ELSE 'Other' END AS DepartmentName FROM Employees;"),
    
    ("Select with subquery", "SELECT FirstName, LastName FROM Employees WHERE DepartmentID IN (SELECT DepartmentID FROM Departments WHERE DepartmentName = 'IT');"),
    
    ("Select using EXISTS", "SELECT FirstName, LastName FROM Employees WHERE EXISTS (SELECT 1 FROM Projects WHERE Projects.EmployeeID = Employees.EmployeeID);"),
    
    ("Using PIVOT", "SELECT * FROM (SELECT Department, Salary FROM Employees) AS SourceTable PIVOT (AVG(Salary) FOR Department IN ([HR], [IT], [Finance])) AS PivotTable;")
]

# Generate data programmatically with query_id
data = [(i+1, description, query) for i, (description, query) in enumerate(queries)]

# Create Spark DataFrame from the data
df = spark.createDataFrame(data, schema)

# Show the Spark DataFrame
display(df)

# COMMAND ----------

_ = (
  df.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.queries")
)

# COMMAND ----------


