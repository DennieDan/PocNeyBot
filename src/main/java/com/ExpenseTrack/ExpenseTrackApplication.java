package com.ExpenseTrack;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ExpenseTrackApplication {
    public static void main(String[] args) {
        SpringApplication.run(ExpenseTrackApplication.class, args);
        System.out.println("Expense Track Application running on http://localhost:8080");
    }
}