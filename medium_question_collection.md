# Acceptance Rate By Date (facebook)
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

# Highest Energy Consumption (facebook)
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

# Finding User Purchases (amazon)
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

# Lowest Priced Orders (amazon)
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

# Second Highest Salary
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

# Favorite Customer
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
