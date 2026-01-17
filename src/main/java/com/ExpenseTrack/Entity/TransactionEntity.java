package com.ExpenseTrack.Entity;
import java.util.Date;

import javax.persistence.*;

import com.ExpenseTrack.model.ExpenseRequest;

@Entity
@Table(name = "transactions")
public class TransactionEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Temporal(TemporalType.TIMESTAMP)
    @Column(name = "date")
    private Date date;

    private double amount;

    @Column(name = "description")
    private String description;

    @Column(name = "expense")
    private String expense;

    @Column(name = "currency")
    private String currency;


    @ManyToOne
    @JoinColumn(name = "category_id")
    private CategoryEntity category;

    @ManyToOne
    @JoinColumn(name = "payment_method_id")
    private PaymentMethodEntity paymentMethod;

    @Column(name = "ai_comment")
    private String aiComment;

    public TransactionEntity() {
        this.date = new Date();
        this.aiComment = "Pending AI analysis";
    }

    public TransactionEntity(ExpenseRequest request) {
        this.date = request.getOccurredAt() != null ? request.getOccurredAt() : new Date();
        this.amount = request.getAmount();
        this.description = request.getNote();
        this.expense = request.getMerchant() != null ? request.getMerchant() : "Unknown";
        this.currency = request.getCurrency();
        this.aiComment = "Transaction processed: " + (request.getNote() != null ? request.getNote() : "New expense");
    }

    public Long getId() {
        return id;
    }
    public void setId(Long id) {
        this.id = id;
    }

    public Date getDate() {
        return date;
    }
    public void setDate(Date date) {
        this.date = date;
    }

    public double getAmount() {
        return amount;
    }
    public void setAmount(double amount) {
        this.amount = amount;
    }

    public String getDescription() {
        return description;
    }
    public void setDescription(String description) {
        this.description = description;
    }

    public String getExpense() {
        return expense;
    }
    public void setExpense(String expense) {
        this.expense = expense;
    }

    public CategoryEntity getCategory() {
        return category;
    }
    public void setCategory(CategoryEntity category) {
        this.category = category;
    }

    public PaymentMethodEntity getPaymentMethod() {
        return paymentMethod;
    }
    public void setPaymentMethod(PaymentMethodEntity paymentMethod) {
        this.paymentMethod = paymentMethod;
    }

    public String getAiComment() {
        return aiComment;
    }
    public void setAiComment(String aiComment) {
        this.aiComment = aiComment;
    }
    public String getCurrency() {
        return currency;
    }
    public void setCurrency(String currency) {
        this.currency = currency;
    }

}
