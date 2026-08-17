# Eval sample log: expected vs. trained model vs. plain filter vs. no rerank

- 50 row(s) sampled

## Row 4: how many cars were produced in 1980?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM CARS_DATA WHERE YEAR  =  1980;
```
Expected rows:
```
(29,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(Id) FROM cars_data WHERE Year = 1980
```
Rows:
```
(29,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(T2.Maker) FROM cars_data AS T1 INNER JOIN car_makers AS T2 ON T1.Id = T2.Country WHERE STRFTIME('%Y', T1.Year) = '1980'
```
Rows:
```
(0,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT COUNT(T2.Maker) FROM cars_data AS T1 INNER JOIN car_makers AS T2 ON T1.Id = T2.Country WHERE STRFTIME('%Y', T1.Year) = '1980'
```
Rows:
```
(0,)
```
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 13: What is the number of cars with a horsepower greater than 150?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM CARS_DATA WHERE horsepower  >  150;
```
Expected rows:
```
(281,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(*) FROM cars_data WHERE Horsepower > '150'
```
Rows:
```
(281,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(*) FROM cars_data WHERE Horsepower > '150'
```
Rows:
```
(281,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(*) FROM cars_data WHERE Horsepower > '150'
```
Rows:
```
(281,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 14: What are the different years in which there were cars produced that weighed less than 4000 and also cars that weighted more than 3000?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT DISTINCT T1.Year FROM CARS_DATA AS T1 WHERE T1.Weight  >  3000 AND T1.weight  <  4000;
```
Expected rows:
```
(1970,)
(1971,)
(1972,)
(1973,)
(1974,)
(1975,)
(1976,)
(1977,)
(1978,)
(1979,)
(1980,)
(1981,)
(1982,)
```

**Trained model + schema-filter:**
```sql
SELECT DISTINCT YEAR FROM cars_data WHERE Weight < 4000 OR Weight > 3000
```
Rows:
```
(1970,)
(1971,)
(1972,)
(1973,)
(1974,)
(1975,)
(1976,)
(1977,)
(1978,)
(1979,)
(1980,)
(1981,)
(1982,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT DISTINCT YEAR FROM cars_data WHERE Weight < 4000 OR Weight > 3000
```
Rows:
```
(1970,)
(1971,)
(1972,)
(1973,)
(1974,)
(1975,)
(1976,)
(1977,)
(1978,)
(1979,)
(1980,)
(1981,)
(1982,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT DISTINCT YEAR FROM cars_data WHERE Weight < 4000 OR Weight > 3000
```
Rows:
```
(1970,)
(1971,)
(1972,)
(1973,)
(1974,)
(1975,)
(1976,)
(1977,)
(1978,)
(1979,)
(1980,)
(1981,)
(1982,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 16: What are the names of all European countries with at least 3 manufacturers?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.CountryName FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country WHERE T2.Continent  =  'europe' GROUP BY T1.CountryName HAVING count(*)  >=  3;
```
Expected rows:
```
('france',)
('germany',)
```

**Trained model + schema-filter:**
```sql
SELECT T1.CountryName 
FROM countries AS T1 
JOIN car_makers AS T2 ON T1.CountryId = T2.Country 
GROUP BY T1.CountryName 
HAVING COUNT(T2.Maker) >= 3
```
Rows:
```
('france',)
('germany',)
('japan',)
('usa',)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.CountryName 
FROM countries AS T1 
JOIN car_makers AS T2 ON T1.CountryId = T2.Country 
GROUP BY T1.CountryName 
HAVING COUNT(T2.Maker) >= 3
```
Rows:
```
('france',)
('germany',)
('japan',)
('usa',)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T1.CountryName 
FROM countries AS T1 
JOIN countries AS T2 ON T1.Continent = T2.Continent 
WHERE T1.Continent IN ( SELECT Continent FROM car_makers GROUP BY Continent HAVING COUNT(Maker) >= 3 )
```
❌ execution error: `ambiguous column name: Continent`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 17: Among the cars that do not have the minimum horsepower, what are the make ids and names of al those with less than 4 cylinders?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T2.MakeId ,  T2.Make FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Horsepower  >  (SELECT min(Horsepower) FROM CARS_DATA) AND T1.Cylinders  <=  3;
```
Expected rows:
```
(79, 'mazda rx2 coupe')
(119, 'mazda rx3')
(251, 'mazda rx-4')
```

**Trained model + schema-filter:**
```sql
SELECT T2.MakeId, T3.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId INNER JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Horsepower < ( SELECT MIN(Horsepower) FROM cars_data ) AND T1.Cylinders < 4
```
_(0 rows)_
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.MakeId, T3.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId INNER JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Horsepower < ( SELECT MIN(Horsepower) FROM cars_data ) AND T1.Cylinders < 4
```
_(0 rows)_
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T2.MakeId, T3.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId INNER JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Horsepower < ( SELECT MIN(Horsepower) FROM cars_data ) AND T1.Cylinders < 4
```
_(0 rows)_
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 23: Which makers designed more than 3 car models? List full name and the id.

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.FullName ,  T1.Id FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id HAVING count(*)  >  3;
```
Expected rows:
```
('General Motors', 4)
('Chrysler', 6)
```

**Trained model + schema-filter:**
```sql
SELECT T2.FullName, T2.Id FROM model_list AS T1 INNER JOIN car_makers AS T2 ON T1.Maker = T2.Id GROUP BY T2.Id HAVING COUNT(T1.ModelId) > 3
```
Rows:
```
('General Motors', 4)
('Chrysler', 6)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.FullName, T1.Maker FROM model_list AS T1 INNER JOIN car_makers AS T2 ON T1.Maker = T2.Id GROUP BY T1.Maker HAVING COUNT(T1.Model) > 3
```
Rows:
```
('General Motors', 4)
('Chrysler', 6)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T2.FullName, T1.Maker FROM model_list AS T1 INNER JOIN car_makers AS T2 ON T1.Maker = T2.Id GROUP BY T1.Maker HAVING COUNT(T1.Model) > 3
```
Rows:
```
('General Motors', 4)
('Chrysler', 6)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 24: What is the average weight of cars each year?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT avg(Weight) ,  YEAR FROM CARS_DATA GROUP BY YEAR;
```
Expected rows:
```
(3441.3142857142857, 1970)
(2960.344827586207, 1971)
(3237.714285714286, 1972)
(3419.025, 1973)
(2877.925925925926, 1974)
(3176.8, 1975)
(3078.735294117647, 1976)
(2997.3571428571427, 1977)
(2861.8055555555557, 1978)
(3055.344827586207, 1979)
(2436.655172413793, 1980)
(2532.1666666666665, 1981)
(2453.548387096774, 1982)
```

**Trained model + schema-filter:**
```sql
SELECT AVG("weight") FROM `cars_data` GROUP BY "year"
```
Rows:
```
(3441.3142857142857,)
(2960.344827586207,)
(3237.714285714286,)
(3419.025,)
(2877.925925925926,)
(3176.8,)
(3078.735294117647,)
(2997.3571428571427,)
(2861.8055555555557,)
(3055.344827586207,)
(2436.655172413793,)
(2532.1666666666665,)
(2453.548387096774,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT AVG("weight") FROM `cars_data` GROUP BY "year"
```
Rows:
```
(3441.3142857142857,)
(2960.344827586207,)
(3237.714285714286,)
(3419.025,)
(2877.925925925926,)
(3176.8,)
(3078.735294117647,)
(2997.3571428571427,)
(2861.8055555555557,)
(3055.344827586207,)
(2436.655172413793,)
(2532.1666666666665,)
(2453.548387096774,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT AVG(T2.Weight) AS avg_weight, T1.Year FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId GROUP BY T1.Year
```
❌ execution error: `no such column: T2.Weight`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 36: Which distinct car models are the produced after 1980?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.model  =  T2.model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.id WHERE T3.year  >  1980;
```
Expected rows:
```
('plymouth',)
('buick',)
('dodge',)
('chevrolet',)
('toyota',)
('honda',)
('subaru',)
('datsun',)
('mazda',)
('ford',)
('volkswagen',)
('renault',)
('peugeot',)
('saab',)
('volvo',)
('oldsmobile',)
('chrysler',)
('pontiac',)
('amc',)
('mercury',)
... and 1 more row(s)
```

**Trained model + schema-filter:**
```sql
SELECT DISTINCT T2.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId WHERE T1.Year > 1980
```
Rows:
```
('plymouth',)
('buick',)
('dodge',)
('chevrolet',)
('toyota',)
('honda',)
('subaru',)
('datsun',)
('mazda',)
('ford',)
('volkswagen',)
('renault',)
('peugeot',)
('saab',)
('volvo',)
('oldsmobile',)
('chrysler',)
('pontiac',)
('amc',)
('mercury',)
... and 1 more row(s)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT DISTINCT T3.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId INNER JOIN model_list AS T3 ON T3.Model = T2.Model WHERE T1.Year > 1980
```
Rows:
```
('plymouth',)
('buick',)
('dodge',)
('chevrolet',)
('toyota',)
('honda',)
('subaru',)
('datsun',)
('mazda',)
('ford',)
('volkswagen',)
('renault',)
('peugeot',)
('saab',)
('volvo',)
('oldsmobile',)
('chrysler',)
('pontiac',)
('amc',)
('mercury',)
... and 1 more row(s)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT DISTINCT Model FROM cars_data WHERE YEAR > 1980
```
❌ execution error: `no such column: Model`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 41: What is the horsepower of the car with the largest accelerate?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.horsepower FROM CARS_DATA AS T1 ORDER BY T1.accelerate DESC LIMIT 1;
```
Expected rows:
```
('71',)
```

**Trained model + schema-filter:**
```sql
SELECT Horsepower FROM cars_data ORDER BY Accelerate DESC LIMIT 1
```
Rows:
```
('71',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Horsepower FROM cars_data ORDER BY Accelerate DESC LIMIT 1
```
Rows:
```
('71',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Horsepower FROM cars_data ORDER BY Accelerate DESC LIMIT 1
```
Rows:
```
('71',)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 45: In 1980, how many cars were made?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM CARS_DATA WHERE YEAR  =  1980;
```
Expected rows:
```
(29,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(*) FROM cars_data WHERE YEAR = 1980
```
Rows:
```
(29,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(*) FROM cars_data WHERE Year = 1980
```
Rows:
```
(29,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(*) FROM cars_data WHERE Year = 1980
```
Rows:
```
(29,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 48: What is the average horsepower of the cars before 1980?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT avg(horsepower) FROM CARS_DATA WHERE YEAR  <  1980;
```
Expected rows:
```
(111.13291139240506,)
```

**Trained model + schema-filter:**
```sql
SELECT AVG(Horsepower) FROM cars_data WHERE YEAR < 1980
```
Rows:
```
(111.13291139240506,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT AVG(Horsepower) FROM cars_data WHERE YEAR < 1980
```
Rows:
```
(111.13291139240506,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT AVG(T2.Horsepower) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId WHERE T1.Year < 1980
```
❌ execution error: `no such column: T2.Horsepower`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 50: What is the number of continents?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM CONTINENTS;
```
Expected rows:
```
(5,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(*) FROM continents
```
Rows:
```
(5,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(ContId) FROM continents
```
Rows:
```
(5,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(ContId) FROM continents
```
Rows:
```
(5,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 52: What is the smallest weight of the car produced with 8 cylinders on 1974?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT Weight FROM CARS_DATA WHERE Cylinders  =  4 AND YEAR  =  1974 ORDER BY Weight ASC LIMIT 1;
```
Expected rows:
```
(1649,)
```

**Trained model + schema-filter:**
```sql
SELECT MIN(Weight) FROM cars_data WHERE Cylinders = 8 AND Year = 1974
```
Rows:
```
(4141,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT MIN(Weight) FROM cars_data WHERE Cylinders = 8 AND Year = 1974
```
Rows:
```
(4141,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT MIN(Weight) FROM cars_data WHERE Cylinders = 8 AND Year = 1974
```
Rows:
```
(4141,)
```
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 53: What is the name of the country with the most car makers?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T2.CountryName FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId GROUP BY T1.Country ORDER BY Count(*) DESC LIMIT 1;
```
Expected rows:
```
('japan',)
```

**Trained model + schema-filter:**
```sql
SELECT CountryName FROM countries GROUP BY Continent ORDER BY COUNT(CountryId) DESC LIMIT 1
```
Rows:
```
('germany',)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.CountryName 
FROM countries AS T1 
JOIN car_makers AS T2 ON T1.CountryId = T2.Country 
GROUP BY T1.CountryName 
ORDER BY COUNT(T2.Maker) DESC LIMIT 1
```
Rows:
```
('japan',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T1.CountryName FROM countries AS T1 INNER JOIN car_makers AS T2 ON T1.CountryId = T2.Country ORDER BY COUNT(T2.Id) DESC LIMIT 1
```
❌ execution error: `misuse of aggregate: COUNT()`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 58: What are the different models for the cards produced after 1980?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT DISTINCT T1.model FROM MODEL_LIST AS T1 JOIN CAR_NAMES AS T2 ON T1.model  =  T2.model JOIN CARS_DATA AS T3 ON T2.MakeId  =  T3.id WHERE T3.year  >  1980;
```
Expected rows:
```
('plymouth',)
('buick',)
('dodge',)
('chevrolet',)
('toyota',)
('honda',)
('subaru',)
('datsun',)
('mazda',)
('ford',)
('volkswagen',)
('renault',)
('peugeot',)
('saab',)
('volvo',)
('oldsmobile',)
('chrysler',)
('pontiac',)
('amc',)
('mercury',)
... and 1 more row(s)
```

**Trained model + schema-filter:**
```sql
SELECT T2.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId WHERE T1.Year > 1980 GROUP BY T2.Model
```
Rows:
```
('amc',)
('buick',)
('chevrolet',)
('chrysler',)
('datsun',)
('dodge',)
('ford',)
('honda',)
('mazda',)
('mercury',)
('nissan',)
('oldsmobile',)
('peugeot',)
('plymouth',)
('pontiac',)
('renault',)
('saab',)
('subaru',)
('toyota',)
('volkswagen',)
... and 1 more row(s)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T3.Model FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId INNER JOIN model_list AS T3 ON T2.Model = T3.Model WHERE T1.Year > 1980
```
Rows:
```
('plymouth',)
('buick',)
('dodge',)
('chevrolet',)
('plymouth',)
('toyota',)
('plymouth',)
('honda',)
('subaru',)
('datsun',)
('toyota',)
('mazda',)
('plymouth',)
('ford',)
('ford',)
('volkswagen',)
('renault',)
('honda',)
('toyota',)
('datsun',)
... and 41 more row(s)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT DISTINCT Model FROM cars_data WHERE YEAR > 1980
```
❌ execution error: `no such column: Model`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 64: What is the number of countries with more than 2 car makers?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT COUNT(*) FROM ( SELECT T1.CountryId ,  COUNT(*) FROM COUNTRIES AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country GROUP BY T1.CountryId HAVING count(*)  >  2 );
```
Expected rows:
```
(4,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(CountryName) FROM countries WHERE CountryId IN ( SELECT Country FROM car_makers GROUP BY Country HAVING COUNT(Id) > 2 )
```
Rows:
```
(4,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(DISTINCT T3.CountryName) FROM car_makers AS T1 INNER JOIN cars_data AS T2 ON T1.Id = T2.Id INNER JOIN countries AS T3 ON T1.Country = T3.Continent WHERE T1.Id IN ( SELECT Id FROM car_makers GROUP BY Id HAVING COUNT(*) > 2 )
```
Rows:
```
(0,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT COUNT(DISTINCT T3.CountryName) FROM car_makers AS T1 INNER JOIN cars_data AS T2 ON T1.Id = T2.Id INNER JOIN countries AS T3 ON T1.Country = T3.Continent WHERE T1.Id IN ( SELECT Id FROM car_makers GROUP BY Id HAVING COUNT(*) > 2 )
```
Rows:
```
(0,)
```
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 72: What is the average miles per gallon of all the cards with 4 cylinders?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT avg(mpg) FROM CARS_DATA WHERE Cylinders  =  4;
```
Expected rows:
```
(28.86231884057971,)
```

**Trained model + schema-filter:**
```sql
SELECT AVG(CAST(REPLACE(trim(Edispl, 'k'), 'k', '') AS REAL)) FROM cars_data WHERE Cylinders = 4
```
Rows:
```
(109.90096618357488,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT AVG(CAST(REPLACE(REPLACE(T1.MPG, ',', ''), '.', '') AS REAL)) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId WHERE T1.Cylinders = 4
```
Rows:
```
(136.40579710144928,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT AVG(T2.MPG) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId WHERE T1.Cylinders = 4
```
❌ execution error: `no such column: T2.MPG`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 80: What is the maker of the carr produced in the earliest year and what year was it?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T2.Make ,  T1.Year FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T1.Year  =  (SELECT min(YEAR) FROM CARS_DATA);
```
Expected rows:
```
('chevrolet chevelle malibu', 1970)
('buick skylark 320', 1970)
('plymouth satellite', 1970)
('amc rebel sst', 1970)
('ford torino', 1970)
('ford galaxie 500', 1970)
('chevrolet impala', 1970)
('plymouth fury iii', 1970)
('pontiac catalina', 1970)
('amc ambassador dpl', 1970)
('citroen ds-21 pallas', 1970)
('chevrolet chevelle concours (sw)', 1970)
('ford torino (sw)', 1970)
('plymouth satellite (sw)', 1970)
('amc rebel sst (sw)', 1970)
('dodge challenger se', 1970)
('plymouth cuda 340', 1970)
('ford mustang boss 302', 1970)
('chevrolet monte carlo', 1970)
('buick estate wagon (sw)', 1970)
... and 15 more row(s)
```

**Trained model + schema-filter:**
```sql
SELECT T2.Maker, MIN(T1.Year) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId GROUP BY T2.Maker
```
❌ execution error: `no such column: T2.Maker`
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.Maker, MIN(T1.Year) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId GROUP BY T2.Maker
```
❌ execution error: `no such column: T2.Maker`
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T2.Maker, MIN(T1.Year) FROM cars_data AS T1 INNER JOIN car_names AS T2 ON T1.Id = T2.MakeId GROUP BY T2.Maker
```
❌ execution error: `no such column: T2.Maker`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 82: What is the full name of each car maker, along with its id and how many models it produces?

**Schema:**
```sql
CREATE TABLE "continents" (
  "ContId" INTEGER PRIMARY KEY,
  "Continent" TEXT /* one of: 'africa', 'america', 'asia', 'australia', 'europe' */
)
CREATE TABLE "countries" (
  "CountryId" INTEGER PRIMARY KEY,
  "CountryName" TEXT,
  "Continent" INTEGER,
  FOREIGN KEY (Continent) REFERENCES continents (
    ContId
  )
)
CREATE TABLE "car_makers" (
  "Id" INTEGER PRIMARY KEY,
  "Maker" TEXT,
  "FullName" TEXT,
  "Country" TEXT,
  FOREIGN KEY (Country) REFERENCES countries (
    CountryId
  )
)
CREATE TABLE "model_list" (
  "ModelId" INTEGER PRIMARY KEY,
  "Maker" INTEGER,
  "Model" TEXT UNIQUE,
  FOREIGN KEY (Maker) REFERENCES car_makers (
    Id
  )
)
CREATE TABLE "car_names" (
  "MakeId" INTEGER PRIMARY KEY,
  "Model" TEXT,
  "Make" TEXT,
  FOREIGN KEY (Model) REFERENCES model_list (
    Model
  )
)
CREATE TABLE "cars_data" (
  "Id" INTEGER PRIMARY KEY,
  "MPG" TEXT,
  "Cylinders" INTEGER,
  "Edispl" REAL,
  "Horsepower" TEXT,
  "Weight" INTEGER,
  "Accelerate" REAL,
  "Year" INTEGER,
  FOREIGN KEY (Id) REFERENCES car_names (
    MakeId
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.FullName ,  T1.Id ,  count(*) FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker GROUP BY T1.Id;
```
Expected rows:
```
('American Motor Company', 1, 1)
('Volkswagen', 2, 2)
('BMW', 3, 1)
('General Motors', 4, 5)
('Ford Motor Company', 5, 3)
('Chrysler', 6, 4)
('Citroen', 7, 1)
('Nissan Motors', 8, 2)
('Fiat', 9, 1)
('Honda', 11, 1)
('Mazda', 12, 1)
('Daimler Benz', 13, 2)
('Opel', 14, 1)
('Peugeaut', 15, 1)
('Renault', 16, 1)
('Saab', 17, 1)
('Subaru', 18, 1)
('Toyota', 19, 2)
('Triumph', 20, 1)
('Volvo', 21, 1)
... and 2 more row(s)
```

**Trained model + schema-filter:**
```sql
SELECT T1.Id, T1.FullName, COUNT(T2.ModelId) FROM car_makers AS T1 INNER JOIN model_list AS T2 ON T1.Id = T2.Maker GROUP BY T1.Id
```
Rows:
```
(1, 'American Motor Company', 1)
(2, 'Volkswagen', 2)
(3, 'BMW', 1)
(4, 'General Motors', 5)
(5, 'Ford Motor Company', 3)
(6, 'Chrysler', 4)
(7, 'Citroen', 1)
(8, 'Nissan Motors', 2)
(9, 'Fiat', 1)
(11, 'Honda', 1)
(12, 'Mazda', 1)
(13, 'Daimler Benz', 2)
(14, 'Opel', 1)
(15, 'Peugeaut', 1)
(16, 'Renault', 1)
(17, 'Saab', 1)
(18, 'Subaru', 1)
(19, 'Toyota', 2)
(20, 'Triumph', 1)
(21, 'Volvo', 1)
... and 2 more row(s)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.Id, T1.FullName, COUNT(T2.ModelId) FROM car_makers AS T1 INNER JOIN model_list AS T2 ON T1.Id = T2.Maker GROUP BY T1.Id
```
Rows:
```
(1, 'American Motor Company', 1)
(2, 'Volkswagen', 2)
(3, 'BMW', 1)
(4, 'General Motors', 5)
(5, 'Ford Motor Company', 3)
(6, 'Chrysler', 4)
(7, 'Citroen', 1)
(8, 'Nissan Motors', 2)
(9, 'Fiat', 1)
(11, 'Honda', 1)
(12, 'Mazda', 1)
(13, 'Daimler Benz', 2)
(14, 'Opel', 1)
(15, 'Peugeaut', 1)
(16, 'Renault', 1)
(17, 'Saab', 1)
(18, 'Subaru', 1)
(19, 'Toyota', 2)
(20, 'Triumph', 1)
(21, 'Volvo', 1)
... and 2 more row(s)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T1.Id, T1.FullName, COUNT(T2.ModelId) FROM car_makers AS T1 INNER JOIN model_list AS T2 ON T1.Id = T2.Maker GROUP BY T1.Id
```
Rows:
```
(1, 'American Motor Company', 1)
(2, 'Volkswagen', 2)
(3, 'BMW', 1)
(4, 'General Motors', 5)
(5, 'Ford Motor Company', 3)
(6, 'Chrysler', 4)
(7, 'Citroen', 1)
(8, 'Nissan Motors', 2)
(9, 'Fiat', 1)
(11, 'Honda', 1)
(12, 'Mazda', 1)
(13, 'Daimler Benz', 2)
(14, 'Opel', 1)
(15, 'Peugeaut', 1)
(16, 'Renault', 1)
(17, 'Saab', 1)
(18, 'Subaru', 1)
(19, 'Toyota', 2)
(20, 'Triumph', 1)
(21, 'Volvo', 1)
... and 2 more row(s)
```
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 99: Show all distinct template type codes for all templates.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT DISTINCT template_type_code FROM Templates
```
Expected rows:
```
('PP',)
('BK',)
('PPT',)
('AD',)
('CV',)
```

**Trained model + schema-filter:**
```sql
SELECT DISTINCT Template_Type_Code FROM Templates
```
Rows:
```
('PP',)
('BK',)
('PPT',)
('AD',)
('CV',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT DISTINCT Template_Type_Code FROM Templates
```
Rows:
```
('PP',)
('BK',)
('PPT',)
('AD',)
('CV',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT DISTINCT Template_Type_Code FROM Templates
```
Rows:
```
('PP',)
('BK',)
('PPT',)
('AD',)
('CV',)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 102: What is the template type descriptions for template type code "AD".

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT template_type_description FROM Ref_template_types WHERE template_type_code  =  "AD"
```
Expected rows:
```
('Advertisement',)
```

**Trained model + schema-filter:**
```sql
SELECT Template_Type_Description FROM Ref_Template_Types WHERE Template_Type_Code = 'AD'
```
Rows:
```
('Advertisement',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Template_Type_Description FROM Ref_Template_Types WHERE Template_Type_Code = 'AD'
```
Rows:
```
('Advertisement',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Template_Type_Description FROM Ref_Template_Types WHERE Template_Type_Code = 'AD'
```
Rows:
```
('Advertisement',)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 111: Show all template type codes with less than three templates.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT template_type_code FROM Templates GROUP BY template_type_code HAVING count(*)  <  3
```
Expected rows:
```
('CV',)
```

**Trained model + schema-filter:**
```sql
SELECT Template_Type_Code FROM Templates GROUP BY Template_Type_Code HAVING COUNT(Template_Type_Code) < 3
```
Rows:
```
('CV',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Template_Type_Code FROM Templates GROUP BY Template_Type_Code HAVING COUNT(Template_ID) < 3
```
Rows:
```
('CV',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Template_Type_Code FROM Templates GROUP BY Template_Type_Code HAVING COUNT(Template_ID) < 3
```
Rows:
```
('CV',)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 112: What are the ids of documents that have 2 or more paragraphs?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT document_id FROM Paragraphs GROUP BY document_id HAVING count(*)  >=  2
```
Expected rows:
```
(80,)
(2394,)
(651512,)
(16514113,)
```

**Trained model + schema-filter:**
```sql
SELECT Document_ID FROM Paragraphs GROUP BY Document_ID HAVING COUNT(Paragraph_ID) >= 2
```
Rows:
```
(80,)
(2394,)
(651512,)
(16514113,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Document_ID FROM Paragraphs GROUP BY Document_ID HAVING COUNT(Paragraph_ID) >= 2
```
Rows:
```
(80,)
(2394,)
(651512,)
(16514113,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Document_ID FROM Paragraphs GROUP BY Document_ID HAVING COUNT(Paragraph_ID) >= 2
```
Rows:
```
(80,)
(2394,)
(651512,)
(16514113,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 113: Which template type code is used by most number of documents?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.template_type_code FROM Templates AS T1 JOIN Documents AS T2 ON T1.template_id  =  T2.template_id GROUP BY T1.template_type_code ORDER BY count(*) DESC LIMIT 1
```
Expected rows:
```
('BK',)
```

**Trained model + schema-filter:**
```sql
SELECT T1.Template_Type_Code FROM Templates AS T1 INNER JOIN Documents AS T2 ON T1.Template_ID = T2.Template_ID GROUP BY T1.Template_Type_Code ORDER BY COUNT(T2.Document_ID) DESC LIMIT 1
```
Rows:
```
('BK',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.Template_Type_Code FROM Templates AS T1 INNER JOIN Documents AS T2 ON T1.Template_ID = T2.Template_ID GROUP BY T1.Template_Type_Code ORDER BY COUNT(T2.Document_ID) DESC LIMIT 1
```
Rows:
```
('BK',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Template_Type_Code FROM Documents GROUP BY Template_Type_Code ORDER BY COUNT(Document_ID) DESC LIMIT 1
```
❌ execution error: `no such column: Template_Type_Code`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 115: How many templates do we have?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM Templates
```
Expected rows:
```
(20,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(Template_ID) FROM Templates
```
Rows:
```
(20,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(Template_ID) FROM Templates
```
Rows:
```
(20,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(Template_ID) FROM Templates
```
Rows:
```
(20,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 117: What are the template ids of any templates used in more than a single document?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT template_id FROM Documents GROUP BY template_id HAVING count(*)  >  1
```
Expected rows:
```
(11,)
(14,)
(25,)
```

**Trained model + schema-filter:**
```sql
SELECT DISTINCT Template_ID FROM Documents
```
Rows:
```
(7,)
(25,)
(6,)
(20,)
(14,)
(11,)
(10,)
(1,)
(22,)
(21,)
(4,)
(8,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT DISTINCT Template_ID FROM Documents
```
Rows:
```
(7,)
(25,)
(6,)
(20,)
(14,)
(11,)
(10,)
(1,)
(22,)
(21,)
(4,)
(8,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT Template_ID FROM Templates GROUP BY Template_ID HAVING COUNT(Document_ID) > 1
```
❌ execution error: `no such column: Document_ID`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 120: Return the ids of templates that have the code PP or PPT.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT template_id FROM Templates WHERE template_type_code  =  "PP" OR template_type_code  =  "PPT"
```
Expected rows:
```
(0,)
(1,)
(6,)
(7,)
(10,)
(18,)
(22,)
(24,)
(25,)
```

**Trained model + schema-filter:**
```sql
SELECT Template_ID FROM Templates WHERE Template_Type_Code = 'PP' OR Template_Type_Code = 'PPT'
```
Rows:
```
(0,)
(1,)
(6,)
(7,)
(10,)
(18,)
(22,)
(24,)
(25,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Template_ID FROM Templates WHERE Template_Type_Code IN ('PP', 'PPT')
```
Rows:
```
(0,)
(1,)
(6,)
(7,)
(10,)
(18,)
(22,)
(24,)
(25,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Template_ID FROM Templates WHERE Template_Type_Code IN ('PP', 'PPT')
```
Rows:
```
(0,)
(1,)
(6,)
(7,)
(10,)
(18,)
(22,)
(24,)
(25,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 126: Count the number of documents that use the PPT template type.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM Documents AS T1 JOIN Templates AS T2 ON T1.Template_ID  =  T2.Template_ID WHERE T2.Template_Type_Code  =  'PPT'
```
Expected rows:
```
(3,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(DISTINCT Document_ID) FROM Documents WHERE Template_ID IN ( SELECT Template_ID FROM Templates WHERE Template_Type_Code = 'PPT' )
```
Rows:
```
(3,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(Template_ID) FROM Templates WHERE Template_Type_Code = 'PPT'
```
Rows:
```
(3,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(DISTINCT T1.Document_ID) FROM Templates AS T1 INNER JOIN Ref_Template_Types AS T2 ON T1.Template_Type_Code = T2.Template_Type_Code WHERE T2.Template_Type_Description = 'Presentation'
```
❌ execution error: `no such column: T1.Document_ID`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 136: Count the number of documents.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM Documents
```
Expected rows:
```
(15,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(Document_ID) FROM Documents
```
Rows:
```
(15,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(Document_ID) FROM Documents
```
Rows:
```
(15,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(Document_ID) FROM Documents
```
Rows:
```
(15,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 141: How many different templates do all document use?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(DISTINCT template_id) FROM Documents
```
Expected rows:
```
(12,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(T2.Template_Type_Code) FROM Documents AS T1 INNER JOIN Templates AS T2 ON T1.Template_ID = T2.Template_ID
```
Rows:
```
(15,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(DISTINCT Template_Type_Code) FROM Templates
```
Rows:
```
(5,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT COUNT(DISTINCT Template_Type_Code) FROM Documents
```
❌ execution error: `no such column: Template_Type_Code`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 143: What is the document id and name with greatest number of paragraphs?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.document_id ,  T2.document_name FROM Paragraphs AS T1 JOIN Documents AS T2 ON T1.document_id  =  T2.document_id GROUP BY T1.document_id ORDER BY count(*) DESC LIMIT 1
```
Expected rows:
```
(2394, 'Customer reviews')
```

**Trained model + schema-filter:**
```sql
SELECT Document_ID, Document_Name FROM Documents WHERE Document_ID = ( SELECT Document_ID FROM Paragraphs GROUP BY Document_ID ORDER BY COUNT(Paragraph_ID) DESC LIMIT 1 )
```
Rows:
```
(2394, 'Customer reviews')
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT D.Document_ID, D.Document_Name FROM Documents D INNER JOIN Paragraphs P ON D.Document_ID = P.Document_ID GROUP BY D.Document_ID ORDER BY COUNT(P.Paragraph_ID) DESC LIMIT 1
```
Rows:
```
(2394, 'Customer reviews')
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT D.Document_ID, D.Document_Name FROM Documents D INNER JOIN Paragraphs P ON D.Document_ID = P.Document_ID GROUP BY D.Document_ID ORDER BY COUNT(P.Paragraph_ID) DESC LIMIT 1
```
Rows:
```
(2394, 'Customer reviews')
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 149: What are the ids, names, and descriptions for all documents?

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT document_id ,  document_name ,  document_description FROM Documents
```
Expected rows:
```
(0, 'Introduction of OS', 'n')
(1, 'Understanding DB', 'y')
(3, 'Summer Show', 'u')
(76, 'Robbin CV', 'y')
(80, 'Welcome to NY', 'h')
(82, 'Data base', 'w')
(2394, 'Customer reviews', 'y')
(3830, 'Do not panic', 'k')
(33930, 'How Google people work', 'z')
(50123, 'Learning French', 'r')
(651512, 'How to write a CV', 'f')
(801801, 'How to read a book', 'w')
(3540024, 'Palm reading', 'y')
(16514113, 'A history of Arts', 'h')
(385906526, 'About Korea', 'b')
```

**Trained model + schema-filter:**
```sql
SELECT Document_ID, Document_Name, Document_Description FROM Documents
```
Rows:
```
(0, 'Introduction of OS', 'n')
(1, 'Understanding DB', 'y')
(3, 'Summer Show', 'u')
(76, 'Robbin CV', 'y')
(80, 'Welcome to NY', 'h')
(82, 'Data base', 'w')
(2394, 'Customer reviews', 'y')
(3830, 'Do not panic', 'k')
(33930, 'How Google people work', 'z')
(50123, 'Learning French', 'r')
(651512, 'How to write a CV', 'f')
(801801, 'How to read a book', 'w')
(3540024, 'Palm reading', 'y')
(16514113, 'A history of Arts', 'h')
(385906526, 'About Korea', 'b')
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT Document_ID, Document_Name, Document_Description FROM Documents
```
Rows:
```
(0, 'Introduction of OS', 'n')
(1, 'Understanding DB', 'y')
(3, 'Summer Show', 'u')
(76, 'Robbin CV', 'y')
(80, 'Welcome to NY', 'h')
(82, 'Data base', 'w')
(2394, 'Customer reviews', 'y')
(3830, 'Do not panic', 'k')
(33930, 'How Google people work', 'z')
(50123, 'Learning French', 'r')
(651512, 'How to write a CV', 'f')
(801801, 'How to read a book', 'w')
(3540024, 'Palm reading', 'y')
(16514113, 'A history of Arts', 'h')
(385906526, 'About Korea', 'b')
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT Document_ID, Document_Name, Document_Description FROM Documents
```
Rows:
```
(0, 'Introduction of OS', 'n')
(1, 'Understanding DB', 'y')
(3, 'Summer Show', 'u')
(76, 'Robbin CV', 'y')
(80, 'Welcome to NY', 'h')
(82, 'Data base', 'w')
(2394, 'Customer reviews', 'y')
(3830, 'Do not panic', 'k')
(33930, 'How Google people work', 'z')
(50123, 'Learning French', 'r')
(651512, 'How to write a CV', 'f')
(801801, 'How to read a book', 'w')
(3540024, 'Palm reading', 'y')
(16514113, 'A history of Arts', 'h')
(385906526, 'About Korea', 'b')
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 151: Return the id of the document with the fewest paragraphs.

**Schema:**
```sql
CREATE TABLE Ref_Template_Types (
  Template_Type_Code TEXT(15) NOT NULL PRIMARY KEY, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Template_Type_Description TEXT(255) NOT NULL /* one of: 'Advertisement', 'Book', 'CV', 'Paper', 'Presentation' */
)
CREATE TABLE Templates (
  Template_ID INTEGER NOT NULL PRIMARY KEY,
  Version_Number INTEGER NOT NULL,
  Template_Type_Code TEXT(15) NOT NULL, /* one of: 'AD', 'BK', 'CV', 'PP', 'PPT' */
  Date_Effective_From DATETIME,
  Date_Effective_To DATETIME,
  Template_Details TEXT(255) NOT NULL, /* one of: '' */
  FOREIGN KEY (Template_Type_Code) REFERENCES Ref_Template_Types (
    Template_Type_Code
  )
)
CREATE TABLE Documents (
  Document_ID INTEGER NOT NULL PRIMARY KEY,
  Template_ID INTEGER,
  Document_Name TEXT(255),
  Document_Description TEXT(255), /* one of: 'b', 'f', 'h', 'k', 'n', 'r', 'u', 'w', 'y', 'z' */
  Other_Details TEXT(255),
  FOREIGN KEY (Template_ID) REFERENCES Templates (
    Template_ID
  )
)
CREATE TABLE Paragraphs (
  Paragraph_ID INTEGER NOT NULL PRIMARY KEY,
  Document_ID INTEGER NOT NULL,
  Paragraph_Text TEXT(255),
  Other_Details TEXT(255),
  FOREIGN KEY (Document_ID) REFERENCES Documents (
    Document_ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT document_id FROM Paragraphs GROUP BY document_id ORDER BY count(*) ASC LIMIT 1
```
Expected rows:
```
(3,)
```

**Trained model + schema-filter:**
```sql
SELECT Document_ID FROM Paragraphs GROUP BY Document_ID ORDER BY COUNT(*) LIMIT 1
```
Rows:
```
(3,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT MIN(Document_ID) FROM Paragraphs
```
Rows:
```
(3,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT MIN(Document_ID) FROM Paragraphs
```
Rows:
```
(3,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 173: How much does the most recent treatment cost?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT cost_of_treatment FROM Treatments ORDER BY date_of_treatment DESC LIMIT 1
```
Expected rows:
```
(407,)
```

**Trained model + schema-filter:**
```sql
SELECT cost_of_treatment FROM Treatments ORDER BY date_of_treatment DESC LIMIT 1
```
Rows:
```
(407,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.cost_of_treatment FROM Treatments AS T1 INNER JOIN Dogs AS T2 ON T1.dog_id = T2.dog_id ORDER BY T1.date_of_treatment DESC LIMIT 1
```
Rows:
```
(407,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T1.cost_of_treatment FROM Treatments AS T1 INNER JOIN Dogs AS T2 ON T1.dog_id = T2.dog_id ORDER BY T1.date_of_treatment DESC LIMIT 1
```
Rows:
```
(407,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 175: Find the id and cell phone of the professionals who operate two or more types of treatments.

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.professional_id ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >=  2
```
Expected rows:
```
(4, '00230569697')
(6, '139-321-7313')
(8, '328.842.3792')
(9, '962-983-8109x3509')
(10, '461-801-2600')
(14, '1-185-137-1945x409')
```

**Trained model + schema-filter:**
```sql
SELECT T2.professional_id, T1.cell_number FROM Professionals AS T1 INNER JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T2.professional_id HAVING COUNT(DISTINCT T2.treatment_type_code) > 1
```
Rows:
```
(6, '139-321-7313')
(9, '962-983-8109x3509')
(10, '461-801-2600')
(14, '1-185-137-1945x409')
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.professional_id, T1.cell_number FROM Professionals AS T1 INNER JOIN Treatments AS T2 ON T1.professional_id = T2.professional_id GROUP BY T2.professional_id HAVING COUNT(DISTINCT T2.treatment_type_code) > 1
```
Rows:
```
(6, '139-321-7313')
(9, '962-983-8109x3509')
(10, '461-801-2600')
(14, '1-185-137-1945x409')
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT professional_id, cell_number FROM Professionals INNER JOIN Treatments ON Professionals.professional_id = Treatments.professional_id GROUP BY professional_id HAVING COUNT(DISTINCT treatment_type_code) > 1
```
❌ execution error: `ambiguous column name: professional_id`
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 177: How many dogs have an age below the average?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT count(*) FROM Dogs WHERE age  <  ( SELECT avg(age) FROM Dogs )
```
Expected rows:
```
(9,)
```

**Trained model + schema-filter:**
```sql
SELECT COUNT(*), AVG(age) 
FROM Dogs 
GROUP BY age HAVING age < AVG(age)
```
Rows:
```
(1, 1.0)
(3, 2.0)
(1, 3.0)
(3, 4.0)
(1, 5.0)
(1, 6.0)
(2, 8.0)
(3, 9.0)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT COUNT(*) FROM Dogs WHERE CAST(SUBSTR(Age, INSTR(Age, '-') + 1) AS REAL) < (SELECT AVG(age) FROM Dogs)
```
Rows:
```
(9,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT COUNT(*) FROM Dogs WHERE CAST(SUBSTR(Age, INSTR(Age, '-') + 1) AS REAL) < (SELECT AVG(age) FROM Dogs)
```
Rows:
```
(9,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 184: How much does the most expensive charge type costs?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT max(charge_amount) FROM Charges
```
Expected rows:
```
(640,)
```

**Trained model + schema-filter:**
```sql
SELECT MAX(charge_amount) FROM Charges
```
Rows:
```
(640,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT MAX(charge_amount) FROM Charges
```
Rows:
```
(640,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT MAX(charge_amount) FROM Charges
```
Rows:
```
(640,)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 186: What are each owner's first name and their dogs's name?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.first_name ,  T2.name FROM Owners AS T1 JOIN Dogs AS T2 ON T1.owner_id  =  T2.owner_id
```
Expected rows:
```
('Jaclyn', 'Kacey')
('Gay', 'Hipolito')
('Nora', 'Mavis')
('Rachelle', 'Houston')
('Emelie', 'Jeffrey')
('Johann', 'Merritt')
('Jaclyn', 'Narciso')
('Rachelle', 'George')
('Melisa', 'Bessie')
('Kade', 'Troy')
('Cindy', 'Betty')
('Orlando', 'Holden')
('Rolando', 'Jesus')
('Rachelle', 'Lyric')
('Lorenz', 'Evangeline')
```

**Trained model + schema-filter:**
```sql
SELECT T1.first_name, T2.name FROM Owners AS T1 INNER JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id
```
Rows:
```
('Jaclyn', 'Kacey')
('Gay', 'Hipolito')
('Nora', 'Mavis')
('Rachelle', 'Houston')
('Emelie', 'Jeffrey')
('Johann', 'Merritt')
('Jaclyn', 'Narciso')
('Rachelle', 'George')
('Melisa', 'Bessie')
('Kade', 'Troy')
('Cindy', 'Betty')
('Orlando', 'Holden')
('Rolando', 'Jesus')
('Rachelle', 'Lyric')
('Lorenz', 'Evangeline')
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.first_name, T2.name FROM Owners AS T1 INNER JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id
```
Rows:
```
('Jaclyn', 'Kacey')
('Gay', 'Hipolito')
('Nora', 'Mavis')
('Rachelle', 'Houston')
('Emelie', 'Jeffrey')
('Johann', 'Merritt')
('Jaclyn', 'Narciso')
('Rachelle', 'George')
('Melisa', 'Bessie')
('Kade', 'Troy')
('Cindy', 'Betty')
('Orlando', 'Holden')
('Rolando', 'Jesus')
('Rachelle', 'Lyric')
('Lorenz', 'Evangeline')
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T1.first_name, T2.name FROM Owners AS T1 INNER JOIN Dogs AS T2 ON T1.owner_id = T2.owner_id
```
Rows:
```
('Jaclyn', 'Kacey')
('Gay', 'Hipolito')
('Nora', 'Mavis')
('Rachelle', 'Houston')
('Emelie', 'Jeffrey')
('Johann', 'Merritt')
('Jaclyn', 'Narciso')
('Rachelle', 'George')
('Melisa', 'Bessie')
('Kade', 'Troy')
('Cindy', 'Betty')
('Orlando', 'Holden')
('Rolando', 'Jesus')
('Rachelle', 'Lyric')
('Lorenz', 'Evangeline')
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 194: List the emails of the professionals who live in the state of Hawaii or the state of Wisconsin.

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT email_address FROM Professionals WHERE state  =  'Hawaii' OR state  =  'Wisconsin'
```
Expected rows:
```
('uboehm@example.org',)
('mekhi.little@example.org',)
```

**Trained model + schema-filter:**
```sql
SELECT email_address FROM Professionals WHERE state = 'Hawaii' OR state = 'Wisconsin'
```
Rows:
```
('uboehm@example.org',)
('mekhi.little@example.org',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.email_address FROM Professionals AS T1 INNER JOIN Owners AS T2 ON T1.role_code = T2.owner_id WHERE T2.state IN ('HI', 'WI')
```
_(0 rows)_
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T2.email_address FROM Profesionals AS T1 INNER JOIN Owners AS T2 ON T1.professional_id = T2.owner_id WHERE T2.state IN ('HI', 'WI')
```
❌ execution error: `no such table: Profesionals`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 195: Compute the average age of all the dogs.

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT avg(age) FROM Dogs
```
Expected rows:
```
(5.066666666666666,)
```

**Trained model + schema-filter:**
```sql
SELECT AVG(age) FROM Dogs
```
Rows:
```
(5.066666666666666,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT AVG(age) FROM Dogs
```
Rows:
```
(5.066666666666666,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT AVG(age) FROM Dogs
```
Rows:
```
(5.066666666666666,)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 215: Tell me the age of the oldest dog.

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT max(age) FROM Dogs
```
Expected rows:
```
('9',)
```

**Trained model + schema-filter:**
```sql
SELECT MAX(T1.age) FROM Dogs AS T1
```
Rows:
```
('9',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT MAX(age) FROM Dogs
```
Rows:
```
('9',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT MAX(age) FROM Dogs
```
Rows:
```
('9',)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 217: What are the emails of the professionals living in either the state of Hawaii or the state of Wisconsin?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT email_address FROM Professionals WHERE state  =  'Hawaii' OR state  =  'Wisconsin'
```
Expected rows:
```
('uboehm@example.org',)
('mekhi.little@example.org',)
```

**Trained model + schema-filter:**
```sql
SELECT email_address FROM Professionals WHERE state = 'Hawaii' OR state = 'Wisconsin'
```
Rows:
```
('uboehm@example.org',)
('mekhi.little@example.org',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T2.email_address FROM Professionals AS T1 INNER JOIN Owners AS T2 ON T1.role_code = T2.owner_id WHERE T2.state IN ('HI', 'WI')
```
_(0 rows)_
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT T2.email_address FROM Professions AS T1 INNER JOIN Owners AS T2 ON T1.professional_id = T2.owner_id WHERE T2.state IN ('HI', 'WI')
```
❌ execution error: `no such table: Professions`
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 230: What is the description of the treatment type that costs the least money in total?

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T1.treatment_type_description FROM Treatment_types AS T1 JOIN Treatments AS T2 ON T1.treatment_type_code  =  T2.treatment_type_code GROUP BY T1.treatment_type_code ORDER BY sum(cost_of_treatment) ASC LIMIT 1
```
Expected rows:
```
('Take for a Walk',)
```

**Trained model + schema-filter:**
```sql
SELECT T1.treatment_type_description FROM Treatment_Types AS T1 INNER JOIN ( SELECT treatment_type_code, SUM(cost_of_treatment) AS cost FROM Treatments GROUP BY treatment_type_code ) AS T2 ON T1.treatment_type_code = T2.treatment_type_code ORDER BY T2.cost LIMIT 1
```
Rows:
```
('Take for a Walk',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.treatment_type_description FROM Treatment_Types AS T1 INNER JOIN ( SELECT treatment_type_code, SUM(cost_of_treatment) AS cost FROM Treatments GROUP BY treatment_type_code ) AS T2 ON T1.treatment_type_code = T2.treatment_type_code ORDER BY T2.cost LIMIT 1
```
Rows:
```
('Take for a Walk',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T1.treatment_type_description FROM Treatment_Types AS T1 INNER JOIN ( SELECT treatment_type_code, SUM(cost_of_treatment) AS cost FROM Treatments GROUP BY treatment_type_code ) AS T2 ON T1.treatment_type_code = T2.treatment_type_code ORDER BY T2.cost LIMIT 1
```
Rows:
```
('Take for a Walk',)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 236: Give me the id, role and email of the professionals who did not perform any treatment on dogs.

**Schema:**
```sql
CREATE TABLE "Breeds" (
  "breed_code" TEXT(10) PRIMARY KEY, /* one of: 'BUL', 'ESK', 'HUS' */
  "breed_name" TEXT(80) /* one of: 'Bulldog', 'Eskimo', 'Husky' */
)
CREATE TABLE "Charges" (
  "charge_id" INTEGER PRIMARY KEY,
  "charge_type" TEXT(10), /* one of: 'Daily Accommodation', 'Drugs', 'Health Check' */
  "charge_amount" REAL(19, 4)
)
CREATE TABLE "Sizes" (
  "size_code" TEXT(10) PRIMARY KEY, /* one of: 'LGE', 'MED', 'SML' */
  "size_description" TEXT(80) /* one of: 'Large', 'Medium', 'Small' */
)
CREATE TABLE "Treatment_Types" (
  "treatment_type_code" TEXT(10) PRIMARY KEY, /* one of: 'EXAM', 'VAC', 'WALK' */
  "treatment_type_description" TEXT(80) /* one of: 'Physical examination', 'Take for a Walk', 'Vaccination' */
)
CREATE TABLE "Owners" (
  "owner_id" INTEGER PRIMARY KEY,
  "first_name" TEXT(50),
  "last_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Dogs" (
  "dog_id" INTEGER PRIMARY KEY,
  "owner_id" INTEGER NOT NULL,
  "abandoned_yn" TEXT(1), /* one of: '0', '1' */
  "breed_code" TEXT(10) NOT NULL, /* one of: 'BUL', 'ESK', 'HUS' */
  "size_code" TEXT(10) NOT NULL, /* one of: 'LGE', 'MED', 'SML' */
  "name" TEXT(50),
  "age" TEXT(20), /* one of: '1', '2', '3', '4', '5', '6', '8', '9' */
  "date_of_birth" DATETIME,
  "gender" TEXT(1), /* one of: '0', '1' */
  "weight" TEXT(20),
  "date_arrived" DATETIME,
  "date_adopted" DATETIME,
  "date_departed" DATETIME,
  FOREIGN KEY ("breed_code") REFERENCES "Breeds" (
    "breed_code"
  ),
  FOREIGN KEY ("size_code") REFERENCES "Sizes" (
    "size_code"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  ),
  FOREIGN KEY ("owner_id") REFERENCES "Owners" (
    "owner_id"
  )
)
CREATE TABLE "Professionals" (
  "professional_id" INTEGER PRIMARY KEY,
  "role_code" TEXT(10) NOT NULL, /* one of: 'Employee', 'Veterenarian' */
  "first_name" TEXT(50),
  "street" TEXT(50),
  "city" TEXT(50),
  "state" TEXT(20),
  "zip_code" TEXT(20),
  "last_name" TEXT(50),
  "email_address" TEXT(50),
  "home_phone" TEXT(20),
  "cell_number" TEXT(20)
)
CREATE TABLE "Treatments" (
  "treatment_id" INTEGER PRIMARY KEY,
  "dog_id" INTEGER NOT NULL,
  "professional_id" INTEGER NOT NULL,
  "treatment_type_code" TEXT(10) NOT NULL, /* one of: 'EXAM', 'VAC', 'WALK' */
  "date_of_treatment" DATETIME,
  "cost_of_treatment" REAL(19, 4),
  FOREIGN KEY ("treatment_type_code") REFERENCES "Treatment_Types" (
    "treatment_type_code"
  ),
  FOREIGN KEY ("professional_id") REFERENCES "Professionals" (
    "professional_id"
  ),
  FOREIGN KEY ("dog_id") REFERENCES "Dogs" (
    "dog_id"
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT professional_id ,  role_code ,  email_address FROM Professionals EXCEPT SELECT T1.professional_id ,  T1.role_code ,  T1.email_address FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id
```
Expected rows:
```
(1, 'Employee', 'deanna.schuster@example.com')
(2, 'Employee', 'lucile.shanahan@example.org')
(3, 'Employee', 'uboehm@example.org')
(11, 'Employee', 'west.eula@example.net')
(12, 'Veterenarian', 'marquardt.furman@example.org')
(13, 'Veterenarian', 'delphine29@example.com')
(15, 'Employee', 'jeichmann@example.com')
```

**Trained model + schema-filter:**
```sql
SELECT professional_id, role_code, email_address FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )
```
Rows:
```
(1, 'Employee', 'deanna.schuster@example.com')
(2, 'Employee', 'lucile.shanahan@example.org')
(3, 'Employee', 'uboehm@example.org')
(11, 'Employee', 'west.eula@example.net')
(12, 'Veterenarian', 'marquardt.furman@example.org')
(13, 'Veterenarian', 'delphine29@example.com')
(15, 'Employee', 'jeichmann@example.com')
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT professional_id, role_code, email_address FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )
```
Rows:
```
(1, 'Employee', 'deanna.schuster@example.com')
(2, 'Employee', 'lucile.shanahan@example.org')
(3, 'Employee', 'uboehm@example.org')
(11, 'Employee', 'west.eula@example.net')
(12, 'Veterenarian', 'marquardt.furman@example.org')
(13, 'Veterenarian', 'delphine29@example.com')
(15, 'Employee', 'jeichmann@example.com')
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT professional_id, role_code, email_address FROM Professionals WHERE professional_id NOT IN ( SELECT professional_id FROM Treatments )
```
Rows:
```
(1, 'Employee', 'deanna.schuster@example.com')
(2, 'Employee', 'lucile.shanahan@example.org')
(3, 'Employee', 'uboehm@example.org')
(11, 'Employee', 'west.eula@example.net')
(12, 'Veterenarian', 'marquardt.furman@example.org')
(13, 'Veterenarian', 'delphine29@example.com')
(15, 'Employee', 'jeichmann@example.com')
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 259: What is the name of the high schooler who has the greatest number of friends?

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id GROUP BY T1.student_id ORDER BY count(*) DESC LIMIT 1
```
Expected rows:
```
('Andrew',)
```

**Trained model + schema-filter:**
```sql
SELECT T1.name FROM Highschooler AS T1 JOIN Friend AS T2 ON T1.ID = T2.student_id GROUP BY T2.student_id ORDER BY COUNT(T2.student_id) DESC LIMIT 1
```
Rows:
```
('Andrew',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT T1.name FROM Highschooler AS T1 JOIN Friend AS T2 ON T1.ID = T2.student_id GROUP BY T2.student_id ORDER BY COUNT(*) DESC LIMIT 1
```
Rows:
```
('Andrew',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT T1.name FROM Highschooler AS T1 JOIN Friend AS T2 ON T1.ID = T2.student_id GROUP BY T2.student_id ORDER BY COUNT(*) DESC LIMIT 1
```
Rows:
```
('Andrew',)
```
QA correct: ✅ yes

Oracle bucket: `mixed`

---

## Row 275: Show the names and grades of each high schooler.

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT name ,  grade FROM Highschooler
```
Expected rows:
```
('Jordan', 9)
('Gabriel', 9)
('Tiffany', 9)
('Cassandra', 9)
('Haley', 10)
('Andrew', 10)
('Kris', 10)
('Brittany', 10)
('Alexis', 11)
('Austin', 11)
('Gabriel', 11)
('Jessica', 11)
('Jordan', 12)
('John', 12)
('Kyle', 12)
('Logan', 12)
```

**Trained model + schema-filter:**
```sql
SELECT name, grade FROM Highschooler
```
Rows:
```
('Jordan', 9)
('Gabriel', 9)
('Tiffany', 9)
('Cassandra', 9)
('Haley', 10)
('Andrew', 10)
('Kris', 10)
('Brittany', 10)
('Alexis', 11)
('Austin', 11)
('Gabriel', 11)
('Jessica', 11)
('Jordan', 12)
('John', 12)
('Kyle', 12)
('Logan', 12)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT name, grade FROM Highschooler
```
Rows:
```
('Jordan', 9)
('Gabriel', 9)
('Tiffany', 9)
('Cassandra', 9)
('Haley', 10)
('Andrew', 10)
('Kris', 10)
('Brittany', 10)
('Alexis', 11)
('Austin', 11)
('Gabriel', 11)
('Jessica', 11)
('Jordan', 12)
('John', 12)
('Kyle', 12)
('Logan', 12)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT name, grade FROM Highschooler
```
Rows:
```
('Jordan', 9)
('Gabriel', 9)
('Tiffany', 9)
('Cassandra', 9)
('Haley', 10)
('Andrew', 10)
('Kris', 10)
('Brittany', 10)
('Alexis', 11)
('Austin', 11)
('Gabriel', 11)
('Jessica', 11)
('Jordan', 12)
('John', 12)
('Kyle', 12)
('Logan', 12)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 280: Show ids of all students who do not have any friends.

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT id FROM Highschooler EXCEPT SELECT student_id FROM Friend
```
Expected rows:
```
(1025,)
(1641,)
```

**Trained model + schema-filter:**
```sql
SELECT DISTINCT student_id FROM Friend WHERE student_id NOT IN (SELECT friend_id FROM Friend)
```
Rows:
```
(1510,)
```
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT DISTINCT student_id FROM Friend WHERE student_id NOT IN (SELECT friend_id FROM Friend)
```
Rows:
```
(1510,)
```
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT DISTINCT student_id FROM Friend WHERE student_id NOT IN (SELECT friend_id FROM Friend)
```
Rows:
```
(1510,)
```
QA correct: ❌ no

Oracle bucket: `zero_correct`

---

## Row 283: Show the names of students who have a grade higher than 5 and have at least 2 friends.

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT T2.name FROM Friend AS T1 JOIN Highschooler AS T2 ON T1.student_id  =  T2.id WHERE T2.grade  >  5 GROUP BY T1.student_id HAVING count(*)  >=  2
```
Expected rows:
```
('Alexis',)
('Kris',)
('Jordan',)
('Gabriel',)
('Andrew',)
```

**Trained model + schema-filter:**
```sql
SELECT name FROM Highschooler WHERE grade > 5 AND (SELECT COUNT(*) FROM Friend WHERE friend_id = student_id) >= 2
```
_(0 rows)_
QA correct: ❌ no

**Plain filter (constant + schema-filter):**
```sql
SELECT name FROM Highschooler WHERE grade > 5 AND (SELECT COUNT(*) FROM Friend WHERE friend_id = student_id) >= 2
```
_(0 rows)_
QA correct: ❌ no

**No rerank (first candidate):**
```sql
SELECT name FROM Highschooler WHERE grade > 5 AND (SELECT COUNT(*) FROM Friend WHERE friend_id = student_id) >= 2
```
_(0 rows)_
QA correct: ❌ no

Oracle bucket: `mixed`

---

## Row 288: What are the names of all high schoolers in grade 10?

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT name FROM Highschooler WHERE grade  =  10
```
Expected rows:
```
('Haley',)
('Andrew',)
('Kris',)
('Brittany',)
```

**Trained model + schema-filter:**
```sql
SELECT name FROM Highschooler WHERE grade = 10
```
Rows:
```
('Haley',)
('Andrew',)
('Kris',)
('Brittany',)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT name FROM Highschooler WHERE grade = 10
```
Rows:
```
('Haley',)
('Andrew',)
('Kris',)
('Brittany',)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT name FROM Highschooler WHERE grade = 10
```
Rows:
```
('Haley',)
('Andrew',)
('Kris',)
('Brittany',)
```
QA correct: ✅ yes

Oracle bucket: `all_correct`

---

## Row 296: Show me all grades that have at least 4 students.

**Schema:**
```sql
CREATE TABLE Highschooler (
  ID INTEGER PRIMARY KEY,
  name TEXT,
  grade INTEGER
)
CREATE TABLE Friend (
  student_id INTEGER,
  friend_id INTEGER,
  PRIMARY KEY (student_id, friend_id),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (friend_id) REFERENCES Highschooler (
    ID
  )
)
CREATE TABLE Likes (
  student_id INTEGER,
  liked_id INTEGER,
  PRIMARY KEY (student_id, liked_id),
  FOREIGN KEY (liked_id) REFERENCES Highschooler (
    ID
  ),
  FOREIGN KEY (student_id) REFERENCES Highschooler (
    ID
  )
)
```

**Expected (`sql_good`):**
```sql
SELECT grade FROM Highschooler GROUP BY grade HAVING count(*)  >=  4
```
Expected rows:
```
(9,)
(10,)
(11,)
(12,)
```

**Trained model + schema-filter:**
```sql
SELECT grade FROM Highschooler GROUP BY grade HAVING COUNT(ID) >= 4
```
Rows:
```
(9,)
(10,)
(11,)
(12,)
```
QA correct: ✅ yes

**Plain filter (constant + schema-filter):**
```sql
SELECT grade FROM Highschooler GROUP BY grade HAVING COUNT(ID) >= 4
```
Rows:
```
(9,)
(10,)
(11,)
(12,)
```
QA correct: ✅ yes

**No rerank (first candidate):**
```sql
SELECT grade FROM Highschooler GROUP BY grade HAVING COUNT(DISTINCT student_id) >= 4
```
❌ execution error: `no such column: student_id`
QA correct: ❌ no

Oracle bucket: `mixed`

---
