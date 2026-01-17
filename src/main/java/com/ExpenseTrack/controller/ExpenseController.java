package com.ExpenseTrack.controller;

import com.ExpenseTrack.model.ExpenseRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.sql.DataSource;
import javax.transaction.Transactional;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/transactions")
public class ExpenseController {
    @PersistenceContext
    private EntityManager entityManager;
    @Autowired
    private DataSource dataSource;

    @PostMapping
    @Transactional
    public ResponseEntity<?> createExpense(@RequestBody ExpenseRequest request) {
        try {
            Connection conn = dataSource.getConnection();
            System.out.println("Received expense:");
            System.out.println("   Amount: $" + request.getAmount() + " " + request.getCurrency());
            System.out.println("   Note: " + request.getNote());
            String categoryName = request.getCategory() != null ?
                    request.getCategory() : "Uncategorized";
            Long categoryId = findOrCreateCategoryId(conn, categoryName);

            String methodName = request.getMethod() != null ?
                    request.getMethod() : "Unknown";
            Long paymentMethodId = findOrCreatePaymentMethod(methodName);

            String merchant = request.getMerchant() != null ?
                    request.getMerchant() : "Unknown";
            String note = request.getNote() != null ?
                    request.getNote() : "No description";
            Date occurredAt = request.getOccurredAt() != null ?
                    request.getOccurredAt() : new Date();

            entityManager.createNativeQuery(
                            "INSERT INTO transactions (date, amount, description, expense, category_id, payment_method_id, ai_comment) " +
                                    "VALUES (?, ?, ?, ?, ?, ?, ?)")
                    .setParameter(1, occurredAt)
                    .setParameter(2, request.getAmount())
                    .setParameter(3, note)
                    //.setParameter(4, merchant)
                    .setParameter(4, categoryId)
                    .setParameter(5, paymentMethodId)
                    .setParameter(6, "Pending AI analysis")
                    .executeUpdate();

            Object lastId = entityManager.createNativeQuery(
                    "SELECT MAX(id) FROM transactions").getSingleResult();

            Map<String, Object> response = new HashMap<>();
            response.put("message", "Transaction saved successfully");
            response.put("transactionId", lastId);
            response.put("status", "saved_to_db");

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();

            Map<String, String> error = new HashMap<>();
            error.put("error", "Database insert failed");
            error.put("details", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
}
    private Long findOrCreateCategoryId(Connection conn, String categoryName) throws SQLException {
        // Try to find existing category using JDBC
        String findSql = "SELECT id FROM categories WHERE name = ?";
        try (PreparedStatement stmt = conn.prepareStatement(findSql)) {
            stmt.setString(1, categoryName);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                return rs.getLong("id");
            }
        }

        // Create new category using JDBC
        System.out.println("   Creating new category: " + categoryName);
        String insertSql = "INSERT INTO categories (name, description) VALUES (?, ?) RETURNING id";
        try (PreparedStatement stmt = conn.prepareStatement(insertSql)) {
            stmt.setString(1, categoryName);
            stmt.setString(2, "User created category");
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                return rs.getLong("id");
            }
        }

        throw new SQLException("Failed to create category");
    }

    private Long findOrCreatePaymentMethod(String methodName) {
        try {
            Object existing = entityManager.createNativeQuery(
                            "SELECT id FROM payment_methods WHERE name = ?")
                    .setParameter(1, methodName)
                    .getSingleResult();
            return ((Number) existing).longValue();
        } catch (Exception e) {
            System.out.println("   Creating new payment method: " + methodName);
            entityManager.createNativeQuery(
                            "INSERT INTO payment_methods (name, description, created_at) VALUES (?, ?, ?)")
                    .setParameter(1, methodName)
                    .setParameter(2, "User created payment method")
                    .setParameter(3, new Date())
                    .executeUpdate();

            Object newId = entityManager.createNativeQuery(
                    "SELECT MAX(id) FROM payment_methods").getSingleResult();
            return ((Number) newId).longValue();
        }
    }
    @GetMapping("/health")
    public String health() {
        return "Simple Expense API. Test Database operations";
    }
    @GetMapping("/test-db")
    @Transactional
    public ResponseEntity<?> testDatabase() {
        try {
            Object result = entityManager.createNativeQuery(
                    "SELECT COUNT(*) FROM transactions").getSingleResult();

            Map<String, Object> response = new HashMap<>();
            response.put("status", "database_connected");
            response.put("transaction_count", result);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, String> error = new HashMap<>();
            error.put("status", "database_error");
            error.put("message", e.getMessage());
            return ResponseEntity.status(500).body(error);
        }
    }
    @PostMapping("/direct")
    public ResponseEntity<?> directInsert(@RequestBody Map<String, Object> request) {
        System.out.println("🔧 DIRECT INSERT WITH EXPLICIT COMMIT");

        Connection conn = null;
        try {
            // Get connection WITHOUT transaction management
            conn = dataSource.getConnection();

            // CRITICAL: Set auto-commit to TRUE
            conn.setAutoCommit(true);

            System.out.println("Auto-commit: " + conn.getAutoCommit());
            System.out.println("Database: " + conn.getMetaData().getDatabaseProductName());
            System.out.println("URL: " + conn.getMetaData().getURL());

            double amount = request.get("amount") != null ?
                    Double.parseDouble(request.get("amount").toString()) : 0.0;
            String note = request.get("note") != null ?
                    request.get("note").toString() : "";
            /*String merchant = request.get("merchant") != null ?
                    request.get("merchant").toString() : "Unknown";*/
            String currency = request.get("currency") != null ?
                    request.get("currency").toString():"SGD";
            //String category = request
            //Long categoryId = findOrCreateCategory();
            boolean isExpense = true; // Default to expense
            if (request.containsKey("isExpense")) {
                isExpense = Boolean.parseBoolean(request.get("isExpense").toString());
            } else if (request.containsKey("type")) {
                String type = request.get("type").toString().toLowerCase();
                isExpense = !type.contains("income");
            }
            String categoryName = request.get("category") != null ?
                    request.get("category").toString() : "Uncategorized";
            Long categoryId = findOrCreateCategoryId(conn, categoryName);
            //Long categoryId = findOrCreateCategory(request.get("category"));
            // PostgreSQL INSERT with RETURNING
//
            String sql = "INSERT INTO transactions (date, amount, description, expense, category_id," +
                    "currency, ai_comment) " +
                    "VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?) " +
                    "RETURNING id, date, amount, expense";
            try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.setDouble(1, amount);
                stmt.setString(2, note);
                stmt.setBoolean(3, isExpense);
                stmt.setLong(4, categoryId);
                //stmt.setString(5, merchant);
                stmt.setString(5, currency);
                stmt.setString(6, "Quick insert");
                //stmt.setString(4, "Direct insert with commit");

                ResultSet rs = stmt.executeQuery();
                if (rs.next()) {
                    Long id = rs.getLong("id");
                    Timestamp date = rs.getTimestamp("date");
                    Double insertedAmount = rs.getDouble("amount");
                    Boolean expenseFlag = rs.getBoolean("expense");

                    System.out.println("   INSERTED IN DATABASE:");
                    System.out.println("   ID: " + id);
                    System.out.println("   Date: " + date);
                    System.out.println("   Amount: " + insertedAmount);
                    System.out.println("   Database: " + conn.getMetaData().getDatabaseProductName());
                    System.out.println("   Is Expense: " + expenseFlag + " (1=true/expense, 0=false/income)");
                    //System.out.println("   Merchant Note: " + merchant);

                    // VERIFY it's actually there
                    String verifySql = "SELECT COUNT(*) as count FROM transactions WHERE id = ?";
                    try (PreparedStatement verifyStmt = conn.prepareStatement(verifySql)) {
                        verifyStmt.setLong(1, id);
                        ResultSet verifyRs = verifyStmt.executeQuery();
                        if (verifyRs.next()) {
                            Long count = verifyRs.getLong("count");
                            System.out.println("   VERIFICATION: Found " + count + " record(s)");
                        }
                    }

                    return ResponseEntity.ok(Map.of(
                            "success", true,
                            "id", id,
                            "date", date.toString(),
                            "amount", insertedAmount,
                            "is_expense", expenseFlag,
                            //"database", conn.getMetaData().getDatabaseProductName(),
                            "message", "Committed to database"
                    ));
                }
            }

        } catch (Exception e) {
            System.err.println("Insert error: " + e.getMessage());
            e.printStackTrace();

            // Check if it's H2 error
            if (e.getMessage().contains("H2")) {
                return ResponseEntity.status(500).body(Map.of(
                        "error", "STILL CONNECTED TO H2!",
                        "message", "Check application.properties"
                ));
            }

        } finally {
            if (conn != null) {
                try { conn.close(); } catch (Exception e) {}
            }
        }

        return ResponseEntity.status(500).body(Map.of(
                "success", false,
                "message", "Insert failed"
        ));
    }
}
