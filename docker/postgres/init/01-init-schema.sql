-- This file will be automatically executed when the PostgreSQL container is first created
-- You can define your database schema here
-- Files in this directory are executed in alphabetical order

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    date TEXT NOT NULL,
    category TEXT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    description TEXT,
    expense BOOLEAN NOT NULL
);

-- Create an index on date for better query performance
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);

-- Create an index on category for filtering
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
