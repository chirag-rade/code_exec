-- Create the SalesData table
CREATE TABLE SalesData (
    CustomerID INT,
    PurchaseDate DATE,
    PurchaseAmount DECIMAL(10, 2)
);

-- Execute the provided SQL query
-- Calculate the average purchase amount for both new and returning customers
SELECT
  AVG(CASE
        WHEN CustomerFirstPurchaseDate.PurchaseDate = SalesData.PurchaseDate THEN SalesData.PurchaseAmount
      END) AS AveragePurchaseAmountNewCustomers,
  
  AVG(CASE
        WHEN CustomerFirstPurchaseDate.PurchaseDate <> SalesData.PurchaseDate THEN SalesData.PurchaseAmount
      END) AS AveragePurchaseAmountReturningCustomers
FROM
  SalesData
JOIN
  (SELECT CustomerID, MIN(PurchaseDate) AS PurchaseDate
   FROM SalesData
   GROUP BY CustomerID) AS CustomerFirstPurchaseDate
ON
  SalesData.CustomerID = CustomerFirstPurchaseDate.CustomerID;

-- Median calculation query
WITH RankedPurchases AS (
    SELECT
        SalesData.CustomerID,
        SalesData.PurchaseAmount,
        -- Determine if the transaction is from a new customer (first purchase)
        CustomerFirstPurchaseDate.PurchaseDate = SalesData.PurchaseDate AS IsNewCustomer,
        ROW_NUMBER() OVER(
            PARTITION BY SalesData.CustomerID, CustomerFirstPurchaseDate.PurchaseDate
            ORDER BY SalesData.PurchaseAmount
        ) AS RowAsc,
        ROW_NUMBER() OVER(
            PARTITION BY SalesData.CustomerID, CustomerFirstPurchaseDate.PurchaseDate
            ORDER BY SalesData.PurchaseAmount DESC
        ) AS RowDesc
    FROM
        SalesData
    JOIN
        -- Subquery to determine each customer's first PurchaseDate
        (SELECT CustomerID, MIN(PurchaseDate) AS PurchaseDate
        FROM SalesData
        GROUP BY CustomerID) AS CustomerFirstPurchaseDate
        ON SalesData.CustomerID = CustomerFirstPurchaseDate.CustomerID
),
PurchaseCounts AS (
    SELECT
        IsNewCustomer,
        COUNT(*) AS Total
    FROM RankedPurchases
    GROUP BY IsNewCustomer
),
Medians AS (
    SELECT
        RP.IsNewCustomer,
        -- Median computed as the average PurchaseAmount around the center of the dataset
        AVG(RP.PurchaseAmount) AS MedianPurchaseAmount
    FROM RankedPurchases RP
    JOIN PurchaseCounts PC ON RP.IsNewCustomer = PC.IsNewCustomer
    WHERE RP.RowAsc BETWEEN PC.Total/2.0 AND PC.Total/2.0 + 1
       OR RP.RowDesc BETWEEN PC.Total/2.0 AND PC.Total/2.0 + 1
    GROUP BY RP.IsNewCustomer
)
SELECT
    CASE
        WHEN IsNewCustomer = 1 THEN 'New Customers'
        ELSE 'Returning Customers'
    END AS CustomerType,
    MedianPurchaseAmount
FROM Medians;
