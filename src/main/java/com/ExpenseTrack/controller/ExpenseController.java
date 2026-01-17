package com.ExpenseTrack.controller;

import com.ExpenseTrack.model.ExpenseRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.transaction.Transactional;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
@RestController
@CrossOrigin(origins = "*")
@RequestMapping("/api/transactions")
public class ExpenseController {
    @PersistenceContext
    private EntityManager entityManager;

    @PostMapping
    @Transactional
    public ResponseEntity<?> createExpense(@RequestBody ExpenseRequest request) {
        try {
            System.out.println("Received expense:");
            System.out.println("   Amount: $" + request.getAmount() + " " + request.getCurrency());
            System.out.println("   Note: " + request.getNote());
            String categoryName = request.getCategory() != null ?
                    request.getCategory() : "Uncategorized";
            Long categoryId = findOrCreateCategory(categoryName);

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
                    .setParameter(4, merchant)
                    .setParameter(5, categoryId)
                    .setParameter(6, paymentMethodId)
                    .setParameter(7, "Pending AI analysis")
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

    private Long findOrCreateCategory(String categoryName) {
        try {
            Object existing = entityManager.createNativeQuery(
                            "SELECT id FROM categories WHERE name = ?")
                    .setParameter(1, categoryName)
                    .getSingleResult();
            return ((Number) existing).longValue();
        } catch (Exception e) {
            System.out.println("   Creating new category: " + categoryName);
            entityManager.createNativeQuery(
                            "INSERT INTO categories (name, description, created_at) VALUES (?, ?, ?)")
                    .setParameter(1, categoryName)
                    .setParameter(2, "User created category")
                    .setParameter(3, new Date())
                    .executeUpdate();

            Object newId = entityManager.createNativeQuery(
                    "SELECT MAX(id) FROM categories").getSingleResult();
            return ((Number) newId).longValue();
        }
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

            // Return the new ID
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
}
