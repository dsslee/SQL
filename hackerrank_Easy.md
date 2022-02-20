# BASIC SELECT  

## Selecting Exercisees with CITY Table
Given the CITY table is described as follows:
|  Field | Type |
|-------|-----|
| ID  | NUMBER |
| NAME | VARCHAR2(17)   |
| COUNTRY CODE  | VARCHAR2(3)  |
| DISTRICT |  VARCHAR2(20) |
| POPULATION | NUMBER |

Q1) SELECT ALL - Query all columns for every row in the CITY table.
```sql
SELECT * FROM CITY;
```

Q2) SELECT BY ID -  Query all columns for a city in CITY with the ID 1661.
```sql
SELECT * FROM CITY 
WHERE ID = 1661; 
```

Q3) JAPANESE CITY ATTRIBUTES - Query all attributes of every Japanese city in the CITY table. The COUNTRYCODE for Japan is JPN.
```sql
SELECT * FROM CITY 
WHERE COUNTRYCODE = 'JPN'; 
```

Q4) JAPANESE CITY NAME - Query the names of all the Japanese cities in CITY. The COUNTRYCODE for Japan is JPN.
```sql
SELECT NAME FROM CITY 
WHERE COUNTRYCODE = 'JPN';


Q5) REVISING THE SELECT QUERY I - Query all columns for all American cities in CITY with populations larger than 100,000. The CountryCode for America is USA.
```sql
SELECT * FROM CITY 
WHERE COUNTRYCODE = 'USA' 
AND POPULATION > 100000;
```

Q6) REVISING THE SELECT QUERY II - Query the names of all American cities in CITY with populations larger than 120,000. The CountryCode for America is USA.
```sql
SELECT NAME FROM CITY 
WHERE COUNTRYCODE = 'USA' 
AND POPULATION > 120000;
```

  
## WEATHER OBSERVATION STATION - Easy
Given the  STATION table is described as follows:
|  Field | Type |
|---|---|
| ID  | NUMBER |
| CITY | VARCHAR2(21)   |
| STATE  | VARCHAR2(2)  |
| LAT_N |  NUMBER |
| LONG_W | NUMBER |
where LAT_N is the northern latitude and LONG_W is the western longitude.


Q1) Query a list of CITY and STATE from STATION.
```sql
SELECT CITY, STATE FROM STATION 
```

Q2) Find the sum of all values in LAT_N rounded to a scale of 2 decimal places and the sum of all values in LONG_W rounded to a scale of  decimal places.
```sql
SELECT ROUND(SUM(LAT_N),2), ROUND(SUM(LONG_W ),2)
FROM STATION
```

Q3) Query a list of CITY names from STATION with even ID numbers only. You may print the results in any order, but must exclude duplicates from your answer.
```sql
SELECT DISTINCT CITY 
FROM STATION 
WHERE MOD(ID,2)=0 
ORDER BY CITY ASC;       
```

Q4) Find the difference between the total number of CITY entries in the table and the number of distinct CITY entries in the table.
```sql
SELECT COUNT(CITY) - COUNT(DISTINCT CITY) FROM STATION;       
```

Q5) Query the two cities in STATION with the shortest and longest CITY names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically.
Let's say that CITY only has four entries: DEF, ABC, PQRS and WXY
*Sample Output*
ABC 3 
PQRS 4
```sql
SELECT CITY, LENGTH(CITY) FROM STATION ORDER BY LENGTH(CITY) ASC LIMIT 1;
SELECT CITY, LENGTH(CITY) FROM STATION ORDER BY LENGTH(CITY) DESC LIMIT 1;

SELECT TOP 1 CITY, LENGTH(CITY) FROM STATION ORDER BY LENGTH(CITY) ASC;

```

Q6) Query the list of CITY names starting with vowels (a, e, i, o, u) from STATION. Your result cannot contain duplicates.
```sql
SELECT DISTINT(CITY)
FROM STATION
WHERE CITY REGEX '^[aeiou]'

SELECT DISTINCT(CITY) FROM STATION 
WHERE CITY LIKE 'A%' 
OR CITY LIKE 'E%' 
OR CITY LIKE 'I%' 
OR CITY LIKE 'O%' 
OR CITY LIKE 'U%' 
ORDER BY CITY ASC;       
```

Q7) Query the list of CITY names ending with vowels (a, e, i, o, u) from STATION. Your result cannot contain duplicates.
```sql
SELECT DISTINT(CITY)
FROM STATION
WHERE CITY REGEX '[aeiou]$'

SELECT DISTINT(CITY)
FROM STATION
WHERE CITY RLIKE '.*[aeiou]'

SELECT DISTINCT(CITY) FROM STATION 
WHERE CITY LIKE '%a' 
OR CITY LIKE '%e' 
OR CITY LIKE '%i' 
OR CITY LIKE '%o' 
OR CITY LIKE '%u' 
ORDER BY CITY ASC;
```

Q8) Query the list of CITY names from STATION which have vowels (i.e., a, e, i, o, and u) as both their first and last characters. Your result cannot contain duplicates.
```sql
SELECT DISTINCT(CITY) FROM STATION
WHERE LOWER(CITY) RLIKE '^[aeiou].*[aeiou]$'
```

Q9) Query the list of CITY names from STATION that do not start with vowels. Your result cannot contain duplicates.
```sql
SELECT DISTINCT(CITY) FROM STATION
WHERE LOWER(CITY) NOT RLIKE '^[aeiou].*$'
```

Q10) Query the list of CITY names from STATION that do not end with vowels. Your result cannot contain duplicates.
```sql
SELECT DISTINCT(CITY) FROM STATION
WHERE LOWER(CITY) NOT RLIKE '.*[aeiou]$'
```

Q11) Query the list of CITY names from STATION that either do not start with vowels or do not end with vowels. Your result cannot contain duplicates.
```sql
SELECT DISTINCT(CITY) FROM STATION
WHERE LOWER(CITY) NOT RLIKE '^[aeiou].*[aeiou]$'
```

Q12) Query the list of CITY names from STATION that do not start with vowels and do not end with vowels. Your result cannot contain duplicates.
```sql
SELECT DISTINCT(CITY) FROM STATION
WHERE LOWER(CITY) NOT RLIKE '^[aeiou].*'
AND LOWER(CITY) NOT RLIKE '.*[aeiou]$'
```


**HIGHER THAN 75 MARKS**  
Q) Query the Name of any student in STUDENTS who scored higher than 75 Marks. Order your output by the last three characters of each name. If two or more students both have names ending in the same last three characters (i.e.: Bobby, Robby, etc.), secondary sort them by ascending ID.
Given the STUDENTS table is described as follows:
|  Column | Type |
|---|---|
| ID  | INTEGER |
| NAME | STRING   |
| MARKS  | INTEGER  |

```sql
SELECT NAME FROM STUDENTS
WHERE MARKS > 75
ORDER BY SUBSTRING(NAME,-3,3) ASC, ID ASC 
```

**EMPLOYEE NAMES**
Given the Employee table containing employee data for a company is described as follows:
|  Column | Type |
|---|---|
| employee_id  | INTEGER |
| name | STRING   |
| months | INTEGER  |
| salary | INTEGER |
where employee_id is an employee's ID number, name is their name, months is the total number of months they've been working for the company, and salary is their monthly salary.

Sample Input
|  employee_id | name | months | salary  |
|---|---|----|-----|
| 12228 | Rose | 15 | 1968 |
| 33645 | Angela   | 1 | 3443 |
| 45692  | Frank  | 17  | 1608  |
| 56118  | Patrick  |  7 | 1345
| 59725 | Lisa | 11 | 2330 |
| 74197 | Kimberly   | 16 | 4372 |
| 78454  | Bonnie  |  8 | 1771 |
| 83565 | Michael |  6 | 2017
| 98607  | Todd  |  5 | 3396 |
| 99989 | Joe |  9 | 3573 |

Q1) Write a query that prints a list of employee names (i.e.: the name attribute) from the Employee table in alphabetical order.
```sql
SELECT NAME FROM EMPLOYEE 
ORDER BY NAME;   
```

Q2) Write a query that prints a list of employee names (i.e.: the name attribute) for employees in Employee having a salary greater than $2000 per month who have been employees for less than 10 months. Sort your result by ascending employee_id.
```sql
SELECT NAME FROM EMPLOYEE 
WHERE SALARY > 2000  AND MONTHS < 10 
ORDER BY EMPLOYEE_ID;  
```

# Blunder
**Question:** Samantha was tasked with calculating the average monthly salaries for all employees in the EMPLOYEES table, but did not realize her keyboard's  key was broken until after completing the calculation. She wants your help finding the difference between her miscalculation (using salaries with any zeros removed), and the actual average salary.
Write a query calculating the amount of error (i.e.:  average monthly salaries), and round it up to the next integer.
```sql
SELECT CEIL(AVG(salary) - AVG(REPLACE(salary, 0, ""))) 
--    AVG(salary)          -- 4046.7500
--    , FLOOR(AVG(salary)) -- 4046
--    , CEIL(AVG(salary))  -- 4047
FROM EMPLOYEES;
```

## TOP EARNERS
**Questions:** We define an employee's total earnings to be their monthly  worked, and the maximum total earnings to be the maximum total earnings for any employee in the Employee table. Write a query to find the maximum total earnings for all employees as well as the total number of employees who have maximum total earnings. Then print these values as  space-separated integers.
Total Earnings =  salary x month  ve maximum total earnings
```sql
SELECT E.months * E.salary AS Earning
, COUNT(*)  -- total num of employees who have maximum total earnings
FROM Employee E
GROUP BY 1
ORDER BY Earning DESC
LIMIT 1

SELECT MAX(A.Earnings)
, COUNT(*)
FROM ( -- find Earnings
      SELECT E.months * E.salary AS Earnings FROM EMPLOYEE E
      ) AS A
GROUP BY A.Earnings
ORDER BY A.Earnings DESC
LIMIT 1
```
