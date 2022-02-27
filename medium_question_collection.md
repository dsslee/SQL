# Acceptance Rate By Date (Facebook)
What is the overall friend acceptance rate by date? Your output should have the rate of acceptances by the date the request was sent. Order by the earliest date to latest.
Assume that each friend request starts by a user sending (i.e., user_id_sender) a friend request to another user (i.e., user_id_receiver) that's logged in the table with action = 'sent'. If the request is accepted, the table logs action = 'accepted'. If the request is not accepted, no record of action = 'accepted' is logged.

```sql
-- Plan:
-- 1. LEFT: request sent table 
-- 2. RIGHT: accepted table
-- 3. Join 1. and 2. to understand where there was an acceptance at the same/different day
--  * condition: same id on sender, receiver, and filter right table on accepted
-- 4. Filter joined table for LEFT action as sent
-- 5. Compute acceptance rate based on 3.
-- 6. Order by DATE ASC

SELECT S.date
        ,(COUNT(A.user_id_receiver)/COUNT(S.user_id_sender)) AS acceptance_rate
-- SELECT * 
FROM fb_friend_requests AS S
LEFT JOIN fb_friend_requests AS A
ON S.user_id_sender = A.user_id_sender
AND S.user_id_receiver = A.user_id_receiver
AND A.action = 'accepted' -- join only accepted
WHERE S.action = 'sent' -- filter table by sent
GROUP BY S.date
ORDER BY S.date
```

# Highest Energy Consumption (Facebook)
Find the date with the highest total energy consumption from the Meta/Facebook data centers. Output the date along with the total energy consumption across all data centers.
```sql
-- Plan
-- 1. concat tables
-- 2. sum consumption by date
-- 3. Order by total energy consumption

-- method1
SELECT T.date
        , SUM(T.consumption) AS total_consumption
        , RANK() OVER(ORDER BY SUM(T.consumption) desc) as rnk
FROM (-- UNION ALL CENTERS
SELECT *
    , CASE WHEN consumption > 0 THEN 'Europe' ELSE NULL END AS data_center_location
FROM fb_eu_energy 
UNION ALL
SELECT * 
    , CASE WHEN consumption > 0 THEN 'Asia' ELSE NULL END AS data_center_location
FROM fb_asia_energy 
UNION ALL
SELECT * 
    , CASE WHEN consumption > 0 THEN 'North America' ELSE NULL END AS data_center_location
FROM fb_na_energy 
    ) AS T
GROUP BY T.date
LIMIT 2


-- method2
WITH data AS ( -- UNION ALL CENTERS
            SELECT *
                 , CASE WHEN consumption > 0 THEN 'Europe' ELSE NULL END AS data_center_location
            FROM fb_eu_energy 
            UNION ALL
            SELECT * 
                 , CASE WHEN consumption > 0 THEN 'Asia' ELSE NULL END AS data_center_location
            FROM fb_asia_energy 
            UNION ALL
            SELECT * 
                 , CASE WHEN consumption > 0 THEN 'North America' ELSE NULL END AS data_center_location
            FROM fb_na_energy )

SELECT date
        , total_consumption 
FROM (-- rank consumption
        SELECT date
              , SUM(consumption) as total_consumption 
              , dense_rank() over (order by SUM(consumption) DESC) AS rnk
        FROM data
        GROUP BY date) AS data1
WHERE data1.rnk =1
```

# Finding User Purchases (Amazon)
Write a query that'll identify returning active users. A returning active user is a user that has made a second purchase within 7 days of any other of their purchases. Output a list of user_ids of these returning active users.
```sql
-- PLAN
-- 1. find previous purchase to every row
-- 2. find difference between the current order
-- 3. filter diff within 7 days

-- method1: using lag
WITH temp AS (
SELECT user_id
        , created_at
        , lag(created_at) OVER (PARTITION BY user_id ORDER BY created_at) AS previous_purchase_date
FROM amazon_transactions 
)

SELECT DISTINCT(user_id)
FROM temp
WHERE DATEDIFF(created_at, previous_purchase_date) <= 7

-- method 2: joining 
SELECT DISTINCT t1.user_id
-- SELECT *
FROM amazon_transactions t1, amazon_transactions t2
WHERE t1.user_id = t2.user_id 
AND ABS(DATEDIFF(t1.created_at, t2.created_at)) <= 7 
AND t1.id != t2.id -- but can't be same id
```

# Lowest Priced Orders (Amazon)
Find the lowest order cost of each customer.
Output the customer id along with the first name and the lowest order price.

**MyNOTE**
* There can be more than one minimum price per customer > use distinct on customer id 

```sql
-- method1: using rank 
-- PLAN
-- 1. rank orders by customers
-- 2. remove duplicates 

WITH tmp AS (
SELECT C.id 
        , C.first_name
        , O.total_order_cost
        , DENSE_RANK() OVER(PARTITION BY C.id ORDER BY O.total_order_cost ASC) AS rnk
FROM customers C
JOIN orders O
ON C.id = O.cust_id
ORDER BY C.id
)

SELECT  DISTINCT id
        , first_name
        , total_order_cost
FROM tmp
WHERE rnk = 1
order by id


-- method2: using groupby and min
-- PLAN
-- 1. Find min price per customers
-- 2. remove duplicates 

SELECT C.id 
        , C.first_name
        , MIN(O.total_order_cost) AS min_order_cost
FROM customers C
JOIN orders O ON C.id = O.cust_id
GROUP BY C.id
ORDER BY C.id
```

# Second Highest Salary (Amazon)
Find the second highest salary of employees.
```sql
SELECT *
FROM ( -- rank by salary
SELECT * 
    , RANK() OVER (ORDER BY salary DESC) AS rnk
FROM employee
) AS E
-- ORDER BY salary DESC
WHERE rnk = 2
```

# Favorite Customer (Amazon)
Find "favorite" customers based on the order count and the total cost of orders.
A customer is considered as a favorite if he or she has placed more than 3 orders and with the total cost of orders more than $100.

Output the customer's first name, city, number of orders, and total cost of orders.
```sql
-- PLAN
-- favorite defn: more than 3 orders and with the total cost of orders more than $100
-- 1. count number of orders per customer
-- 2. sum total cost per customer
-- 3. filter for more than 3 orders and more than 100 dollars

WITH tmp AS (
SELECT C.first_name 
       , C.city
        , COUNT(O.id) AS order_count
        , SUM(O.total_order_cost) AS total_cost
FROM customers C
JOIN orders O ON C.id = O.cust_id
GROUP BY C.id, C.city
)

SELECT *
FROM tmp
WHERE order_count > 3
AND total_cost > 100
;


/*
SELECT *
FROM tmp
WHERE order_count > 3
AND total_cost > 100
*/

```

# Highest Cost Orders (Amazon)
Find the customer with the highest daily total order cost between 2019-02-01 to 2019-05-01. If customer had more than one order on a certain day, sum the order costs on daily basis. Output their first name, total cost of their items, and the date. 
For simplicity, you can assume that every first name in the dataset is unique. 
```sql
-- PLAN
-- 1. filter data for specific date
-- 2. check if customer had more than one order
WITH tmp AS (
SELECT C.first_name
        , O.total_order_cost
        , O.order_date
FROM customers C
JOIN orders O on C.id = O.cust_id
WHERE O.order_date BETWEEN '2019-02-01' AND '2019-05-01'
-- ORDER BY 1, 3
)

SELECT first_name
        , order_date
        , RANK() OVER (ORDER BY SUM(total_order_cost) DESC ) AS rnk
        , SUM(total_order_cost) AS total_cost
FROM tmp
GROUP BY first_name, order_date
ORDER BY total_cost DESC
LIMIT 1


-- method 2
WITH daily_cost AS(
SELECT  C.id
        , C.first_name
        , SUM(O.total_order_cost) as total_cost
        , O.order_date
FROM customers AS C
JOIN orders O ON C.id = O.cust_id
WHERE DATE(O.order_date) BETWEEN '2019-02-01' and '2019-05-01'
GROUP BY C.id, O.order_date
),

ranking AS(
SELECT 
    first_name,
    total_cost,
    order_date,
    RANK() OVER (ORDER BY total_cost desc) AS rnk
FROM daily_cost
)

SELECT first_name
        , total_cost
        , order_date
FROM ranking 
WHERE rnk = 1
```

# Customer Revenue In March (Facebook)
Calculate the total revenue from each customer in March 2019. Include only customers who were active in March 2019. 
```sql
WITH tmp AS (
SELECT * 
FROM orders
WHERE MONTH(order_date) = '03'
)

SELECT cust_id
        , SUM(total_order_cost) AS total_cost
FROM tmp
GROUP BY cust_id
ORDER BY total_cost DESC

method2:
SELECT cust_id
        , SUM(total_order_cost) AS total_cost
FROM orders
WHERE MONTH(order_date) = '03'
GROUP BY cust_id
ORDER BY total_cost DESC
```

# Users By Average Session Time (Facebook, July 2021) 
Calculate each user's average session time. A session is defined as the time difference between a page_load and page_exit. For simplicity, assume a user has only 1 session per day and if there are multiple of the same events on that day, consider only the latest page_load and earliest page_exit. Output the user_id and their average session time.
```sql
-- PLAN
-- 1. join load table to exit table
-- 2. for specific date use the latest page_load and earliest page exit
-- 3. Average time difference

WITH tmp AS (
SELECT T1.user_id
        , DATE(T1.timestamp) AS 'load_date'
        , T1.action AS 'action_load'
        , T1.timestamp AS 'load_timestamp'
        , DATE(T2.timestamp) AS 'exit_date'
        , T2.action AS 'action_exit'
        , T2.timestamp AS 'exit_timestamp'
FROM facebook_web_log as T1
JOIN facebook_web_log AS T2
ON T1.user_id = T2.user_id
AND DATE(T1.timestamp) = DATE(T2.timestamp)
AND T2.action = "page_exit"
WHERE T1.action = "page_load"
),

tmp2 AS (
SELECT user_id
        , load_date
        , MAX(load_timestamp) AS load_timestamp
        , MIN(exit_timestamp) AS exit_timestamp
FROM tmp
GROUP BY user_id, load_date
)

SELECT 
user_id
, AVG(TIMESTAMPDIFF(second, load_timestamp, exit_timestamp)) AS timediff
FROM tmp2
GROUP BY user_id

```

# Monthly Percentage Difference
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

# Popularity Percentage
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

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```

```sql
```
