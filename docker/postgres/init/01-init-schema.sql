-- This file will be automatically executed when the PostgreSQL container is first created
-- You can define your database schema here
-- Files in this directory are executed in alphabetical order

-- Create categories table first (needed for foreign key reference)
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    icon_name VARCHAR(255) DEFAULT '',
    colour VARCHAR(8) DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index on category name for faster lookups
CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);

-- Create payment_methods table (needed for foreign key reference)
CREATE TABLE IF NOT EXISTS payment_methods (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create an index on payment method name for faster lookups
CREATE INDEX IF NOT EXISTS idx_payment_methods_name ON payment_methods(name);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    date TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    amount DECIMAL(10, 2) NOT NULL,
    description TEXT,
    merchant VARCHAR(255),
    expense BOOLEAN NOT NULL,
    payment_method_id INTEGER REFERENCES payment_methods(id),
    ai_comment TEXT
);

-- Create an index on date for better query performance
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);

-- Create an index on category_id for filtering
CREATE INDEX IF NOT EXISTS idx_transactions_category_id ON transactions(category_id);

-- Create an index on payment_method_id for filtering
CREATE INDEX IF NOT EXISTS idx_transactions_payment_method_id ON transactions(payment_method_id);
