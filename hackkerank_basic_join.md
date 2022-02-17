# Popularion Census (Lvl: Easy)
Given the CITY and COUNTRY tables, query the sum of the populations of all cities where the CONTINENT is 'Asia'.  
Note: CITY.CountryCode and COUNTRY.Code are matching key columns.  

**SOLUTIONS**
```sql
SELECT SUM(CT.POPULATION)
FROM CITY CT
INNER JOIN COUNTRY CY
ON CT.COUNTRYCODE = CY.CODE
WHERE CY.CONTINENT = 'Asia'
```

# African City (Lvl: Easy)
Given the CITY and COUNTRY tables, query the names of all cities where the CONTINENT is 'Africa'.

**SOLUTIONS**
```sql
SELECT CT.NAME
FROM CITY CT
INNER JOIN COUNTRY CY
ON CT.COUNTRYCODE = CY.CODE
WHERE CY.CONTINENT = 'Africa'
```

# Average Population of Each Continent (Lvl: Easy)
Given the CITY and COUNTRY tables, query the names of all the continents (COUNTRY.Continent) and their respective average city populations (CITY.Population) rounded down to the nearest integer.

**SOLUTIONS**
```sql
SELECT CY.CONTINENT, FLOOR(AVG(CT.POPULATION))
FROM CITY CT
JOIN COUNTRY CY
ON CT.COUNTRYCODE = CY.CODE
GROUP BY CY.CONTINENT
```

# Report (Lvl: Medium)
Q
**SOLUTIONS**
```sql
SELECT IF(grade >= 8, S.name, "NULL")
, G.grade
, s.marks
FROM STUDENTS S
INNER JOIN GRADES G
ON S.marks BETWEEN G.min_mark AND G.max_mark
ORDER BY G.grade DESC, S.name, S.marks
```

# Top Competitors (Lvl: Medium)
**SOLUTIONS**
```sql
```

# (Lvl: Medium)
**SOLUTIONS**
```sql
```

# (Lvl: Medium)
**SOLUTIONS**
```sql
```

# (Lvl: Medium)
**SOLUTIONS**
```sql
```

# (Lvl: Medium)
**SOLUTIONS**
```sql
```
