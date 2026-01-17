package com.ExpenseTrack;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ExpenseTrackApplication {
    public static void main(String[] args) {
        System.setProperty("spring.datasource.url", "jdbc:postgresql://10.94.6.169:5436/spendy-db");
        System.setProperty("spring.datasource.username", "postgres");
        System.setProperty("spring.datasource.password", "postgres");
        System.setProperty("spring.datasource.driver-class-name", "org.postgresql.Driver");
        SpringApplication.run(ExpenseTrackApplication.class, args);
        System.out.println("Expense Track Application running on http://10.94.6.242:8000");
    }
}
