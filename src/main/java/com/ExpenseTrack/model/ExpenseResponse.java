package com.ExpenseTrack.model;

import java.util.Date;
import java.util.UUID;

public class ExpenseResponse {
    private String id;
    private double amount;
    private String currency;
    private String category;
    private String method;
    private Date occurredAt;
    private String merchant;
    private String note;
    private String judgment;
    private String avatarReaction;
    private double lifeHours;
    private int guiltScore;
    private String verdict;
    public ExpenseResponse(ExpenseRequest request, String judgment,
                           //String avatarReaction,
                           double lifeHours,
                           int guiltScore, String verdict) {
        this.id = UUID.randomUUID().toString();
        this.amount = request.getAmount();
        this.currency = request.getCurrency();
        this.category = request.getCategory();
        this.method = request.getMethod();
        this.occurredAt = request.getOccurredAt();
        this.merchant = request.getMerchant();
        this.note = request.getNote();
        this.judgment = judgment;
        this.lifeHours = lifeHours;
        this.guiltScore = guiltScore;
        this.verdict = verdict;
    }
    public String getId() {
        return id;
    }
    public double getAmount() {
        return amount;
    }
    public String getCurrency() {
        return currency;
    }
    public String getCategory() {
        return category;
    }
    public String getMethod() {
        return method;
    }
    public Date getOccurredAt() { return occurredAt; }
    public String getMerchant() { return merchant; }
    public String getNote() { return note; }
    public String getJudgment() { return judgment; }
    public String getAvatarReaction() { return avatarReaction; }
    public double getLifeHours() { return lifeHours; }
    public int getGuiltScore() { return guiltScore; }
    public String getVerdict() { return verdict; }
}
