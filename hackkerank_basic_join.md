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
Q) You are given two tables: Students and Grades. Students contains three columns ID, Name and Marks.
**SOLUTIONS**
Task
1) no name below grade 8 -> null
2) grade in descending order, then name alpha, marks
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
Q) print the respective hacker_id and name of hackers who achieved full scores for more than one challenge. Order your output in descending order by the total number of challenges in which the hacker earned a full score. If more than one hacker received full scores in same number of challenges, then sort them by ascending hacker_id.

**To Do**
1. print out id, name of hackers who submitted the challenges
2. Filter for hacker who earned full score
3. order by total number of challenges DESC, hacker_id ASC

Table Explanation
* Hackers: hackers info
* Difficulty: 문제의 난이도 별 Full score 정보
* Challenges: 문제와, 문제를 제작한 Hacker 정보, 문제의 난이도 정보
* Submissions: 문제를 제출한 사람의 정보와, 제출 시 score 정보

**SOLUTIONS**
```sql
SELECT S.hacker_id, H.name
FROM Submissions S
INNER JOIN Hackers H ON S.hacker_id = H.hacker_id 
INNER JOIN Challenges C ON S.challenge_id = C.challenge_id
INNER JOIN Difficulty D ON C.difficulty_level = D.difficulty_level
-- filter for hacker with full score
WHERE S.score = D.score AND C.difficulty_level = D.difficulty_level
GROUP BY S.hacker_id, H.name
HAVING COUNT(S.challenge_id) > 1
ORDER BY COUNT(S.challenge_id) DESC, S.hacker_id 
/*
NOTE:
submission table: submission_id, hacker_id, challenge_id, score(of submission)
      Join order: name, difficulty_level, score
*/
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

