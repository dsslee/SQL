
Q1) Query a count of the number of cities in CITY having a Population larger than 100,000.
Given City table as follows:
|  Field | Type |
|-------|-----|
| ID  | NUMBER |
| NAME | VARCHAR2(17)   |
| COUNTRY CODE  | VARCHAR2(3)  |
| DISTRICT |  VARCHAR2(20) |
| POPULATION | NUMBER |

**SOLUTION**
```sql
SELECT COUNT(NAME)
FROM CITY
WHERE POPULATION > 100000
```

Q2) Query the total population of all cities in CITY where District is California.
**SOLUTION**
```sql
SELECT SUM(POPULATION)
FROM CITY
WHERE DISTRICT='California'
```

Q3) Query the average population of all cities in CITY where District is California.
**SOLUTION**
```sql
SELECT AVG(POPULATION)
FROM CITY
WHERE DISTRICT='California'
```

Q4) Query the average population for all cities in CITY, rounded down to the nearest integer.
**SOLUTION**
```sql
SELECT FLOOR(AVG(POPULATION))
FROM CITY
```

Q5) Query the sum of the populations for all Japanese cities in CITY. The COUNTRYCODE for Japan is JPN.
**SOLUTION**
```sql
SELECT SUM(POPULATION)
FROM CITY
WHERE COUNTRYCODE = 'JPN'
```

Q) Query the difference between the maximum and minimum populations in CITY.
**SOLUTION**
```sql
SELECT MAX(POPULATION) - MIN(POPULATION)
FROM CITY
```

Q) 
**SOLUTION**
```sql

```

Q) 
**SOLUTION**
```sql

```

Q) 
**SOLUTION**
```sql

```

Q) 
**SOLUTION**
```sql

```

Q) 
**SOLUTION**
```sql

```

Q) 
**SOLUTION**
```sql

```
