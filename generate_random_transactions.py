#!/usr/bin/env python3
"""
Script to generate random transaction data for the transactions table.
"""

import os
import random
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5436")
DB_NAME = os.getenv("DB_NAME", "spendy-db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")


def get_db_connection():
    """Get a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        raise


def ensure_categories_and_payment_methods(conn):
    """Ensure categories and payment methods exist in the database."""
    cursor = conn.cursor()
    
    # Default categories
    categories = [
        ("Food & Dining", "Restaurants, groceries, and food delivery"),
        ("Transportation", "Public transport, rideshare, gas, parking"),
        ("Shopping", "Retail purchases, online shopping"),
        ("Bills & Utilities", "Electricity, water, internet, phone bills"),
        ("Entertainment", "Movies, concerts, games, subscriptions"),
        ("Healthcare", "Medical expenses, pharmacy, insurance"),
        ("Travel", "Hotels, flights, vacation expenses"),
        ("Education", "Courses, books, tuition"),
        ("Personal Care", "Haircuts, cosmetics, gym membership"),
        ("Other", "Miscellaneous expenses"),
    ]
    
    # Default payment methods
    payment_methods = [
        ("Credit Card", "Credit card payments"),
        ("Debit Card", "Debit card payments"),
        ("Cash", "Cash transactions"),
        ("Bank Transfer", "Direct bank transfers"),
        ("PayPal", "PayPal payments"),
        ("GrabPay", "GrabPay wallet"),
        ("PayNow", "PayNow transfers"),
    ]
    
    # Insert categories if they don't exist
    for name, description in categories:
        cursor.execute(
            "INSERT INTO categories (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            (name, description)
        )
    
    # Insert payment methods if they don't exist
    for name, description in payment_methods:
        cursor.execute(
            "INSERT INTO payment_methods (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            (name, description)
        )
    
    conn.commit()
    cursor.close()
    print("✓ Categories and payment methods ensured")


def get_category_ids(conn):
    """Get all category IDs from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM categories")
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return ids


def get_payment_method_ids(conn):
    """Get all payment method IDs from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM payment_methods")
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return ids


def generate_random_transactions(num_transactions=100):
    """Generate random transaction data."""
    
    # Merchant names by category
    merchants_by_category = {
        "Food & Dining": [
            "McDonald's", "Starbucks", "KFC", "Pizza Hut", "Subway",
            "NTUC FairPrice", "Cold Storage", "Giant", "Foodpanda", "GrabFood",
            "Restaurant ABC", "Cafe XYZ", "Hawker Center", "Food Court"
        ],
        "Transportation": [
            "Grab", "Gojek", "ComfortDelGro", "SMRT", "SBS Transit",
            "Shell", "Esso", "Caltex", "SPC", "Parking.sg"
        ],
        "Shopping": [
            "Amazon", "Shopee", "Lazada", "Qoo10", "IKEA",
            "Uniqlo", "H&M", "Zara", "NTUC", "FairPrice"
        ],
        "Bills & Utilities": [
            "SP Group", "PUB", "Singtel", "StarHub", "M1",
            "Netflix", "Spotify", "Disney+", "YouTube Premium"
        ],
        "Entertainment": [
            "Cinema", "KTV", "Arcade", "Escape Room", "Bowling",
            "Netflix", "Spotify", "Disney+", "YouTube Premium"
        ],
        "Healthcare": [
            "Guardian", "Watsons", "Clinic ABC", "Hospital XYZ", "Pharmacy"
        ],
        "Travel": [
            "Booking.com", "Agoda", "Expedia", "AirAsia", "Scoot",
            "Changi Airport", "Hotel ABC", "Resort XYZ"
        ],
        "Education": [
            "Coursera", "Udemy", "Bookstore", "School Supplies", "Library"
        ],
        "Personal Care": [
            "Hair Salon", "Spa", "Gym", "Fitness First", "Pure Fitness"
        ],
        "Other": [
            "Miscellaneous", "Unknown", "Various"
        ],
    }
    
    # Transaction descriptions
    descriptions = [
        "Lunch purchase", "Dinner with friends", "Grocery shopping",
        "Morning coffee", "Taxi ride", "Bus fare", "Train ticket",
        "Online purchase", "Bill payment", "Subscription renewal",
        "Medical consultation", "Pharmacy purchase", "Shopping spree",
        "Entertainment ticket", "Food delivery", "Gas refill",
        "Parking fee", "Hotel booking", "Flight ticket", "Course fee",
        "Book purchase", "Gym membership", "Haircut", "Spa treatment"
    ]
    
    conn = get_db_connection()
    
    try:
        # Ensure categories and payment methods exist
        ensure_categories_and_payment_methods(conn)
        
        # Get IDs
        category_ids = get_category_ids(conn)
        payment_method_ids = get_payment_method_ids(conn)
        
        if not category_ids:
            print("Error: No categories found in database")
            return
        
        if not payment_method_ids:
            print("Error: No payment methods found in database")
            return
        
        # Get category names for merchant selection
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM categories")
        category_map = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        
        # Generate transactions
        cursor = conn.cursor()
        
        # Date range: last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        transactions = []
        for i in range(num_transactions):
            # Random date within range
            random_days = random.randint(0, 180)
            transaction_date = start_date + timedelta(days=random_days)
            
            # Random category
            category_id = random.choice(category_ids)
            category_name = category_map[category_id]
            
            # Random payment method (90% chance of having one)
            payment_method_id = random.choice(payment_method_ids) if random.random() < 0.9 else None
            
            # Random amount (most transactions are small, some are large)
            if random.random() < 0.7:  # 70% small transactions
                amount = round(random.uniform(5.0, 50.0), 2)
            elif random.random() < 0.9:  # 20% medium transactions
                amount = round(random.uniform(50.0, 200.0), 2)
            else:  # 10% large transactions
                amount = round(random.uniform(200.0, 1000.0), 2)
            
            # Random merchant based on category
            if category_name in merchants_by_category:
                merchant = random.choice(merchants_by_category[category_name])
            else:
                merchant = "Unknown Merchant"
            
            # Random description
            description = random.choice(descriptions)
            
            # All transactions are expenses (expense = True)
            expense = True
            
            # AI comment
            ai_comment = f"Auto-generated transaction: {description} at {merchant}"
            
            # Format date as ISO string
            date_str = transaction_date.isoformat()
            
            transactions.append((
                date_str,
                category_id,
                amount,
                description,
                merchant,
                expense,
                payment_method_id,
                ai_comment
            ))
        
        # Bulk insert
        insert_query = """
            INSERT INTO transactions 
            (date, category_id, amount, description, merchant, expense, payment_method_id, ai_comment)
            VALUES %s
        """
        
        psycopg2.extras.execute_values(
            cursor,
            insert_query,
            transactions,
            template=None,
            page_size=100
        )
        
        conn.commit()
        cursor.close()
        
        print(f"✓ Successfully generated {num_transactions} random transactions")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        
    except Exception as e:
        print(f"Error generating transactions: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    
    # Number of transactions to generate (default: 100)
    num_transactions = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"Generating {num_transactions} random transactions...")
    print(f"Connecting to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
    
    generate_random_transactions(num_transactions)
    
    print("Done!")
