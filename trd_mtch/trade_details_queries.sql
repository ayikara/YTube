-- Trade Details Self-Join Queries
-- For derivatives futures trading system where each trade has buyers and sellers

-- Sample data structure:
-- Trade_Details table:
-- Firm_Name (VARCHAR) - Unique key
-- Volume (DECIMAL/NUMERIC) 
-- Trade_ID (VARCHAR/INT) - Same for buyer and seller in a transaction

-- =============================================================================
-- METHOD 1: Basic Self-Join with Aliases
-- =============================================================================
-- This assumes we can distinguish buyers from sellers by some logic
-- (e.g., positive volume = buy, negative volume = sell, or separate indicator)

SELECT 
    b.Trade_ID,
    b.Firm_Name AS Firm_Name_Buy,
    b.Volume AS Buy_Volume,
    s.Firm_Name AS Firm_Name_Sell,
    s.Volume AS Sell_Volume
FROM Trade_Details b
INNER JOIN Trade_Details s 
    ON b.Trade_ID = s.Trade_ID 
    AND b.Firm_Name != s.Firm_Name  -- Avoid matching same firm
WHERE b.Volume > 0  -- Assuming positive volume indicates buy
  AND s.Volume < 0; -- Assuming negative volume indicates sell

-- =============================================================================
-- METHOD 2: If you have a separate Transaction_Type field
-- =============================================================================
-- Assuming there's a Transaction_Type field ('BUY' or 'SELL')

/*
SELECT 
    b.Trade_ID,
    b.Firm_Name AS Firm_Name_Buy,
    b.Volume AS Buy_Volume,
    s.Firm_Name AS Firm_Name_Sell,
    s.Volume AS Sell_Volume
FROM Trade_Details b
INNER JOIN Trade_Details s 
    ON b.Trade_ID = s.Trade_ID 
    AND b.Firm_Name != s.Firm_Name
WHERE b.Transaction_Type = 'BUY'
  AND s.Transaction_Type = 'SELL';
*/

-- =============================================================================
-- METHOD 3: Cross Join Approach (All Combinations)
-- =============================================================================
-- This creates all possible buyer-seller combinations for each Trade_ID

SELECT 
    b.Trade_ID,
    b.Firm_Name AS Firm_Name_Buy,
    b.Volume AS Buy_Volume,
    s.Firm_Name AS Firm_Name_Sell,
    s.Volume AS Sell_Volume
FROM Trade_Details b
CROSS JOIN Trade_Details s
WHERE b.Trade_ID = s.Trade_ID
  AND b.Firm_Name != s.Firm_Name
  AND b.Firm_Name < s.Firm_Name  -- Avoid duplicate pairs (A-B and B-A)
ORDER BY b.Trade_ID, b.Firm_Name, s.Firm_Name;

-- =============================================================================
-- METHOD 4: Window Function Approach (More Advanced)
-- =============================================================================
-- This pairs firms systematically within each Trade_ID

WITH RankedTrades AS (
    SELECT 
        Trade_ID,
        Firm_Name,
        Volume,
        ROW_NUMBER() OVER (PARTITION BY Trade_ID ORDER BY Firm_Name) as rn,
        COUNT(*) OVER (PARTITION BY Trade_ID) as trade_count
    FROM Trade_Details
)
SELECT 
    b.Trade_ID,
    b.Firm_Name AS Firm_Name_Buy,
    b.Volume AS Buy_Volume,
    s.Firm_Name AS Firm_Name_Sell,
    s.Volume AS Sell_Volume
FROM RankedTrades b
JOIN RankedTrades s 
    ON b.Trade_ID = s.Trade_ID 
    AND b.rn != s.rn  -- Different row numbers (different firms)
WHERE b.rn = 1  -- Take first firm as buyer
  AND s.rn = 2  -- Take second firm as seller
ORDER BY b.Trade_ID;

-- =============================================================================
-- METHOD 5: Handle Multiple Buyers/Sellers per Trade
-- =============================================================================
-- This handles the case where one seller can have multiple buyers (or vice versa)

SELECT 
    t1.Trade_ID,
    t1.Firm_Name AS Firm_Name_Buy,
    t1.Volume AS Buy_Volume,
    t2.Firm_Name AS Firm_Name_Sell,
    t2.Volume AS Sell_Volume,
    -- Additional info
    COUNT(*) OVER (PARTITION BY t1.Trade_ID) as Total_Participants
FROM Trade_Details t1
JOIN Trade_Details t2 
    ON t1.Trade_ID = t2.Trade_ID 
    AND t1.Firm_Name != t2.Firm_Name
-- Add your business logic here to determine buyer vs seller
-- Example: assuming first alphabetically is buyer, second is seller
WHERE t1.Firm_Name < t2.Firm_Name
ORDER BY t1.Trade_ID, t1.Firm_Name, t2.Firm_Name;

-- =============================================================================
-- METHOD 6: Comprehensive Solution with All Combinations
-- =============================================================================
-- This shows all possible buyer-seller pairs for each trade

SELECT 
    buyer.Trade_ID,
    buyer.Firm_Name AS Firm_Name_Buy,
    buyer.Volume AS Buy_Volume,
    seller.Firm_Name AS Firm_Name_Sell,
    seller.Volume AS Sell_Volume,
    -- Additional calculated fields
    ABS(buyer.Volume) + ABS(seller.Volume) AS Total_Volume,
    CASE 
        WHEN buyer.Volume = ABS(seller.Volume) THEN 'Matched'
        ELSE 'Unmatched'
    END AS Volume_Match_Status
FROM Trade_Details buyer
JOIN Trade_Details seller 
    ON buyer.Trade_ID = seller.Trade_ID 
    AND buyer.Firm_Name != seller.Firm_Name
-- Optional: Add conditions based on your business rules
-- WHERE buyer.Volume > 0 AND seller.Volume < 0  -- If using signed volumes
ORDER BY buyer.Trade_ID, buyer.Firm_Name, seller.Firm_Name;

-- =============================================================================
-- SAMPLE TEST DATA
-- =============================================================================
/*
-- Create sample table for testing
CREATE TABLE Trade_Details (
    Firm_Name VARCHAR(100) PRIMARY KEY,
    Volume DECIMAL(15,2),
    Trade_ID VARCHAR(50)
);

-- Insert sample data
INSERT INTO Trade_Details VALUES
('Goldman Sachs', 1000.00, 'TXN001'),
('JP Morgan', -1000.00, 'TXN001'),
('Morgan Stanley', 500.00, 'TXN002'),
('Bank of America', 300.00, 'TXN002'),
('Citigroup', -800.00, 'TXN002'),
('Wells Fargo', 2000.00, 'TXN003'),
('Deutsche Bank', -2000.00, 'TXN003');
*/

-- =============================================================================
-- PERFORMANCE OPTIMIZATION TIPS
-- =============================================================================
/*
1. Create index on Trade_ID for faster joins:
   CREATE INDEX idx_trade_id ON Trade_Details(Trade_ID);

2. Create composite index if querying frequently:
   CREATE INDEX idx_trade_firm ON Trade_Details(Trade_ID, Firm_Name);

3. For large datasets, consider partitioning by Trade_ID or date ranges

4. Use EXPLAIN PLAN to analyze query performance
*/
