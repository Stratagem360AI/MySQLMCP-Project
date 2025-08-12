# ==================================================
# init.sql - Sample Database Setup
# ==================================================

"""
-- init.sql - Sample database schema for testing
-- This file can be mounted in Docker for automatic setup

CREATE DATABASE IF NOT EXISTS test_db;
USE test_db;

-- Users table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Products table
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(50),
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    total_amount DECIMAL(10, 2),
    status ENUM('pending', 'completed', 'cancelled') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Order items table
CREATE TABLE order_items (
    id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    product_id INT,
    quantity INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Insert sample data
INSERT INTO users (username, email, first_name, last_name) VALUES
('john_doe', 'john@example.com', 'John', 'Doe'),
('jane_smith', 'jane@example.com', 'Jane', 'Smith'),
('bob_wilson', 'bob@example.com', 'Bob', 'Wilson');

INSERT INTO products (name, description, price, category, stock_quantity) VALUES
('Laptop Computer', 'High-performance laptop for work', 999.99, 'Electronics', 10),
('Office Chair', 'Ergonomic office chair', 299.99, 'Furniture', 25),
('Coffee Maker', 'Automatic coffee brewing machine', 149.99, 'Appliances', 15),
('Wireless Mouse', 'Bluetooth wireless mouse', 29.99, 'Electronics', 50);

INSERT INTO orders (user_id, total_amount, status) VALUES
(1, 1299.98, 'completed'),
(2, 149.99, 'pending'),
(3, 329.98, 'completed');

INSERT INTO order_items (order_id, product_id, quantity, price) VALUES
(1, 1, 1, 999.99),
(1, 2, 1, 299.99),
(2, 3, 1, 149.99),
(3, 2, 1, 299.99),
(3, 4, 1, 29.99);
"""