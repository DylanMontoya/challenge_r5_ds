-- Se realiza por ventanas para evitar las subconsultas
SELECT 
  monthh,
  weekofmonth,
  dayofweek,
  100.0 * SUM(CASE WHEN fraudfound_p = 1 THEN 1 ELSE 0 END) / COUNT(*) OVER (PARTITION BY monthh) AS percentage_fraud_month,
  100.0 * SUM(CASE WHEN fraudfound_p = 1 THEN 1 ELSE 0 END) / COUNT(*) OVER (PARTITION BY monthh, weekofmonth) AS percentage_fraud_month_week,
  100.0 * SUM(CASE WHEN fraudfound_p = 1 THEN 1 ELSE 0 END) / COUNT(*) AS percentage_fraud_month_week_day
FROM 
  fraudes
GROUP BY 
  monthh, weekofmonth, dayofweek
ORDER BY 
  monthh, weekofmonth, dayofweek;