package com.ExpenseTrack.model;

import java.util.Date;

public class ExpenseRequest {
    //class for receiving manual user input from frontend through API endpoint, feed the response in the db accordingly

    //private String id;
    private double amount;
    private String currency;
    private String category;
    private String method;
    private Date occurredAt;
    private String merchant;
    private String note;
    public double getAmount() {
        return amount;
    }
    public void setAmount(double amount) {
        this.amount = amount;
    }
    public String getCurrency() {
        return currency;
    }
    public void setCurrency(String currency) {
        this.currency = currency;
    }

    public String getCategory() {
        return category;
    }
    public void setCategory(String category) {
        this.category = category;
    }

    public String getMethod() {
        return method;
    }
    public void setMethod(String method) {
        this.method = method;
    }

    public Date getOccurredAt() {
        return occurredAt;
    }
    public void setOccurredAt(Date occurredAt) {
        this.occurredAt = occurredAt;
    }

    public String getMerchant() {
        return merchant;
    }
    public void setMerchant(String merchant) {
        this.merchant = merchant;
    }

    public String getNote() {
        return note;
    }
    public void setNote(String note) {
        this.note = note;
    }

}
