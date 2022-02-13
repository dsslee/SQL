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
