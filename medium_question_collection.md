# Acceptance Rate By Date (Facebook, Nov 2020)
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

# Highest Energy Consumption (Facebook, March 2020)
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

# Finding User Purchases (Amazon, Dec 2020)
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

# Highest Cost Orders (Amazon, May 2019)
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

# Customer Revenue In March (Facebook, July 2018)
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

# Monthly Churn Rate (Natera, Nov 2021)
Calculate the churn rate of September 2021 in percentages. The churn rate is the difference between the number of customers on the first day of the month and on the last day of the month, divided by the number of customers on the first day of a month. 
Assume that if customer's contract_end is NULL, their contract is still active. Additionally, if a customer started or finished their contract on a certain day, they should still be counted as a customer on that day.  
GIVEN TABLE: natera_subscriptions 
|columns | dtype |
|--|--|
|user_id | int |
|contract_start | datetime |
|contract_end | datetime |

```sql
-- PLAN 
-- 1. filter to september
-- 1. find the number of customers on the first day
-- 2. find the number of customers on the last day

WITH tmp AS ( -- filter customers
SELECT * 
FROM natera_subscriptions
WHERE contract_start <= '2021-09-01'
AND (contract_end >= '2021-09-01' OR contract_end IS NULL)
),

tmp2 AS ( -- Flag 1 for customers who cancelled in sept
SELECT *
        , CASE WHEN contract_end < '2021-09-30' THEN 1 ELSE 0 END AS canceled_in_sept
FROM tmp
)

SELECT (SUM(canceled_in_sept) / COUNT(*))*100 AS churn_rate
FROM tmp2
```

#
I only looked briefly at question 3 but I think it asked to write a query to calculate the proper compensation amount per customer.
* compensation was 1000won(each product) for 10 or more days, and 3000won 15 or more days.
```sql
-- if I was given the following table:
-- Table1: orders
--  order_date   | date (YYYY-MM-DD)
--  order_id     | INT
--  customer_id  | INT
--  product_id   | INT
--  order_price  | INT
--  delivery_id  | INT

-- Table2: customers
--  cutomer_id    | INT
--  customer_name | INT

-- Table3: delivery
--  id          | INT
--  delivery_id | INT
--  expected_delivery_day | date (YYYY-MM-DD)
--  arrived_delivery_day  | date

-- PLAN
-- 1. Join tables
-- 2. take care of null in arrived_delivery_day with current date
-- 3. calculate late delivery day
-- 4. calculate compensation num
-- 5. sum by customer

WITH tmp AS (
SELECT O.customer_id
			, O.order_id 
			, O.product_id
			, O.delivery_id
			, D.expected_delivery_day
			, CASE WHEN D.arrived_delivery_day IS NULL THEN CURRENT_DATE() END AS arrived_delivery_day
FROM ORDERS O
JOIN customers C ON O.customer_id = C.customer_id
JOIN delivery D on O.order_id = D.id
AND O.deliver_id = D.delivery_id
),

tmp2 AS (
SELECT *
			, DATEDIFF(expected_delivery_day, arrived_delivery_day) AS diff_day
			, CASE WHEN diff_day >= 10 THEN 1000 
             WHEN diff_day >= 15 THEN 3000 
						 ELSE NULL END AS compensation_price
FROM tmp
)

SELECT customer_id
			, SUM(compensation_price) as total
FROM tmp2
GROUP BY customer_id
ORDER BY SUM(compensation_price) DESC

# Show the number ofemployees in each age grouping from oldest to youngest, given table names "Employyes” with the fields -
• employee_first_name (e.g. Adam)
• employee_last_name (e.g. Smith)
• date_of_birth (e.g. 17/09/1990)
• employeeid (e.g. 123456)

```sql
with tmp as( 
-- find age
select empfirstname
       , emplastname
       , employeeid
       , empbirthdate
       , cast(left(cast(current_date as varchar), 4) as int) - cast(right(cast(empbirthdate as varchar), 4) as int) as employee_age
--  , (age(CURRENT_DATE,date(empbirthdate)), 2)
from employees
), 

tmp2 as (--select * from tmp;
select *, case when employee_age < 18 then 'Under 18'
            when employee_age between 18 and 29 then '20s'
            when employee_age between 30 and 39 then '30s'
            when employee_age between 40 and 49 then '40s'
            when employee_age between 50 and 59 then '50s'
            when employee_age > 60 then 'Seniors'
end as age_group
from tmp
)

select age_group, count(*) as cnt
from tmp2
group by age_group
order by age_group desc;
```

