# TYPE OF TRIANGLE - Alternative
Write a query identifying the type of each record in the TRIANGLES table using its three side lengths. 
Output one of the following statements for each record in the table:
 - Equilateral: It's a triangle with  sides of equal length.
 - Isosceles: It's a triangle with  sides of equal length.
 - Scalene: It's a triangle with  sides of differing lengths.
 - Not A Triangle: The given values of A, B, and C don't form a triangle.

Given the TRIANGLES table is described as follows:
|  Column | Type |
|---|---|
| A | INTEGER |
| B | INTEGER |
| C | INTEGER |

Sample Input:
|  A | B | C | 
|---|---|----|
| 20 | 20 | 23 |
| 20 | 20 | 20 |
| 20 | 21 | 22 |
| 13 | 14 | 30 |

```sql
SELECT CASE
    WHEN A + B > C AND A + C > B AND B + C > A THEN CASE
        WHEN A = B AND B = C THEN 'Equilateral'
        WHEN A = B OR A = C OR B = C THEN 'Isosceles'
        ELSE 'Scalene' END
    ELSE 'Not A Triangle' END
FROM TRIANGLES
```
   
     
# OCCUPATIONS - Advanced Select
Given the following table
|  Column | Type |
|---|---|
| Name | String |
| Occupation | String |

Sample Input:
|  A | B |  
|---|---|
|Samantha|Doctor|
|Julia|Actor|
|Maria|Actor|
|Meera|Singer|
|Ashely|Professor|
|Ketty|Professor|
|Christeen|Professor|
|Jane|Actor|
|Jenny|Doctor|
|Priya|Singer|

Q1) Generate the following two result sets:
Query an alphabetically ordered list of all names in OCCUPATIONS, immediately followed by the first letter of each profession as a parenthetical (i.e.: enclosed in parentheses). For example: AnActorName(A), ADoctorName(D), AProfessorName(P), and ASingerName(S).
Query the number of ocurrences of each occupation in OCCUPATIONS. Sort the occurrences in ascending order, and output them in the following format:
There are a total of [occupation_count] [occupation]s.
 - where [occupation_count] is the number of occurrences of an occupation in OCCUPATIONS 
 - [occupation] is the lowercase occupation name. 
 - If more than one Occupation has the same [occupation_count], they should be ordered alphabetically.
Note: There will be at least two entries in the table for each type of occupation.

Sample Output
Ashely(P)
Christeen(P)
There are a total of 3 actors.
There are a total of 3 professors.

**SOLUTION**
```sql
SELECT CONCAT(name, '(', LEFT(occupation, 1), ')')
--SELECT CONCAT(name, '(', SUBSTR(occupation,1,1), ')') --SUBSTR can also work
FROM OCCUPATIONS
ORDER BY name;

SELECT CONCAT('There are a total of ', COUNT(*), ' ', LOWER(occupation), 's.')
FROM OCCUPATIONS
GROUP BY occupation
ORDER BY COUNT(*), occupation;
```
   
Q2) Pivot the Occupation column in OCCUPATIONS so that each Name is sorted alphabetically and displayed underneath its corresponding Occupation. 
The output column headers should be Doctor, Professor, Singer, and Actor, respectively.
Note: Print NULL when there are no more names corresponding to an occupation.

Sample Output
Jenny    Ashley     Meera  Jane
Samantha Christeen  Priya  Julia
NULL     Ketty      NULL   Maria

Explanation
The first column is an alphabetically ordered list of Doctor names.
The second column is an alphabetically ordered list of Professor names.
The third column is an alphabetically ordered list of Singer names.
The fourth column is an alphabetically ordered list of Actor names.
The empty cell data for columns with less than the maximum number of names per occupation (in this case, the Professor and Actor columns) are filled with NULL values.

**SOLUTION**
```sql
set @r1=0, @r2=0, @r3=0, @r4=0;
select min(Doctor), min(Professor), min(Singer), min(Actor)
from(
  select
      case when Occupation='Doctor' then (@r1:=@r1+1)
           when Occupation='Professor' then (@r2:=@r2+1)
           when Occupation='Singer' then (@r3:=@r3+1)
           when Occupation='Actor' then (@r4:=@r4+1) end as RowNumber,

      case when Occupation='Doctor' then Name end as Doctor,
      case when Occupation='Professor' then Name end as Professor,
      case when Occupation='Singer' then Name end as Singer,
      case when Occupation='Actor' then Name end as Actor
  from OCCUPATIONS
  order by Name
) Temp
group by RowNumber

select  
    max(case when occupation = 'Doctor' then name end) 'Doctor',
    max(case when occupation = 'Professor' then name end) 'Professor',
    max(case when occupation = 'Singer' then name end) 'Singer',
    max(case when occupation = 'Actor' then name end) 'Actor'
from (
  select *, row_number() over (partition by occupation order by name) rn
  from occupations
) t
group by rn
```

# BINARY TREE NODES
|  Column | Type |
|---|---|
| A | INTEGER |
| B | INTEGER |
| C | INTEGER |

Sample Input:
|  A | B | C | 
|---|---|----|

```sql

```

# NEW COMPANIES
Q) Amber's conglomerate corporation just acquired some new companies. Each of the companies follows this hierarchy: F > LM > SM > M > E
Given the table schemas below, write a query to print the company_code, founder name, total number of lead managers, total number of senior managers, total number of managers, and total number of employees. Order your output by ascending company_code.

Note:
The tables may contain duplicate records.
The company_code is string, so the sorting should not be numeric. For example, if the company_codes are C_1, C_2, and C_10, then the ascending company_codes will be C_1, C_10, and C_2.

```sql
SELECT C.company_code
        , C.founder
        , COUNT(DISTINCT(E.lead_manager_code))
        , COUNT(DISTINCT(E.senior_manager_code))
        , COUNT(DISTINCT(E.manager_code))
        , COUNT(DISTINCT(E.employee_code))
FROM Company C
JOIN Employee E 
ON C.company_code = E.company_code
GROUP BY C.company_code, C.founder 
ORDER BY C.company_code ASC
```

**SOLUTION**
```sql
```
