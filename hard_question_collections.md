# Monthly Percentage Difference (Amazon, Dec 2020)
Given a table of purchases by date, calculate the month-over-month percentage change in revenue. The output should include the year-month date (YYYY-MM) and percentage change, rounded to the 2nd decimal point, and sorted from the beginning of the year to the end of the year.
The percentage change column will be populated from the 2nd month forward and can be calculated as ((this month's revenue - last month's revenue) / last month's revenue)*100.
```sql
-- PLAN
-- 1. make yearmonth feature
-- 2. find month sum
-- 3. lag monthly sum
-- 4. find percentage change: ((this month's revenue - last month's revenue) / last month's revenue)*100.

method1 : groupby to find montly sum
WITH tmp AS (-- monthly sum
SELECT date_format(created_at, '%Y-%m') AS yearmonth
        , SUM(value) AS monthly_sum
        -- , LAG(SUM(value), 1) OVER(ORDER BY yearmonth)
FROM sf_transactions
GROUP BY yearmonth
),

tmp2 AS (-- lag
SELECT yearmonth
        , monthly_sum
        , LAG(monthly_sum, 1) OVER(ORDER BY yearmonth) AS prev_monthly_sum
FROM tmp
)

SELECT yearmonth
    , ROUND((((monthly_sum - prev_monthly_sum)/prev_monthly_sum)*100),2) AS precentage_change
FROM tmp2
-- ORDER BY yearmonth;


method2: sum over() to find monthly sum
WITH tmp AS (-- monthly sum
SELECT DISTINCT date_format(created_at, '%Y-%m') AS yearmonth
        , SUM(value) OVER (PARTITION BY YEAR(created_at), MONTH(created_at)) AS monthly_sum
FROM sf_transactions
),

tmp2 AS (-- lag
SELECT yearmonth
        , monthly_sum
        , LAG(monthly_sum, 1) OVER(ORDER BY yearmonth) AS prev_monthly_sum
FROM tmp
)

SELECT yearmonth
    , ROUND((((monthly_sum - prev_monthly_sum)/prev_monthly_sum)*100),2) AS precentage_change
FROM tmp2
-- ORDER BY yearmonth;
```

# Find the number of unique properties (AirBnB)
Find how many the number of different property types in the dataset.
 * filter_room_types value Example: ,Entire home/apt,Entire home/apt,Private room,Entire home/apt,Private room,Shared room
```sql
SELECT DISTINCT property_type
FROM
(
    SELECT substring_index(filter_room_types, ',', -1) AS property_type
    FROM airbnb_searches
) AS a
```

# Arizona, California, and Hawaii Employees (Amazon, May 2019)
Find employees from Arizona, California, and Hawaii while making sure to output all employees from each city. Output column headers should be Arizona, California, and Hawaii. Data for all cities must be ordered on the first name.
Assume unequal number of employees per city.
```sql
-- PLAN
-- 1. List Arizona employee
-- 2. List California employee
-- 3. List Hawaii employee
-- 4. JOIN by rownumber

WITH arizona AS (
SELECT first_name
        , ROW_NUMBER() OVER (ORDER BY first_name) AS A
FROM employee
WHERE city LIKE 'Arizona'
),
cali AS (
SELECT first_name
        , ROW_NUMBER() OVER (ORDER BY first_name) AS C
FROM employee
WHERE city LIKE 'California'
),
hawaii AS (
SELECT first_name
        , ROW_NUMBER() OVER (ORDER BY first_name) AS H
FROM employee
WHERE city LIKE 'Hawaii'
)

SELECT arizona.first_name AS 'Arizona'
        , cali.first_name AS 'California'
        , hawaii.first_name AS 'Hawaii'
FROM arizona
LEFT JOIN cali ON arizona.A = cali.C
LEFT JOIN hawaii ON arizona.A = hawaii.H

```

# Popularity Percentage (Facebook, Nov 2020)
Find the popularity percentage for each user on Meta/Facebook. The popularity percentage is defined as the total number of friends the user has divided by the total number of users on the platform, then converted into a percentage by multiplying by 100.
Output each user along with their popularity percentage. Order records in ascending order by user id.
The 'user1' and 'user2' column are pairs of friends.
```sql
-- PLAN
-- 1. total num of friends / total number of users on platform
WITH tmp AS(
SELECT user1
        , COUNT(user2) as f1
FROM facebook_friends
GROUP BY user1
UNION ALL
SELECT user2
        , COUNT(user1) as f2
FROM facebook_friends
GROUP BY user2
ORDER BY user1
)

SELECT user1
        , SUM(f1) / COUNT(user1) OVER() * 100 AS popularity_percent
FROM tmp
GROUP BY user1
```

# Premium vs Freemium (Microsoft, Nov 2020)
Find the total number of downloads for paying and non-paying users by date. Include only records where non-paying customers have more downloads than paying customers. The output should be sorted by earliest date first and contain 3 columns date, non-paying downloads, paying downloads.
```sql

```

# Player with Longest Streak (Amazon, Sep 2021)
You are given a table of tennis players and their matches that they could either win (W) or lose (L). Find the longest streak of wins. A streak is a set of consecutive won matches of one player. 
The streak ends once a player loses their next match. Output the ID of the player or players and the length of the streak.  
GIVEN TABLE: players_results 
|columns | dtype |
|--|--|
|player_id | int |
|match_date | datetime |
|match_result | varchar |

```sql
-- PLAN
-- 1. assign unique id numbers to different streaks
-- 2. aggregate by streak and count number of rows
-- 3. rank the streak from longest to shortest
WITH tmp AS (
SELECT *
, ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY match_date) AS rn1
, ROW_NUMBER() OVER (PARTITION BY player_id, match_result ORDER BY match_date) AS rn2
FROM players_results
ORDER BY player_id, match_date
),

tmp2 AS (
SELECT *
        , rn1-rn2 AS streak_id
FROM tmp
)

SELECT DISTINCT player_id
        , streak_length
FROM (
        SELECT  player_id
                , match_result
                , streak_id
                , COUNT(streak_id) AS streak_length
                , RANK() OVER (ORDER BY COUNT(streak_id) DESC) AS rnk
        FROM tmp2
        WHERE match_result = 'W'
        GROUP BY player_id, match_result, streak_id
        ORDER BY streak_length DESC
      ) AS tmp3
WHERE rnk = 1
;
```

