package com.sap.ai.bestpractices.rpt1;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class Rpt1Application {

    public static void main(String[] args) {
        SpringApplication.run(Rpt1Application.class, args);
    }

    @Bean
    ApplicationRunner applicationRunner(Rpt1Runner runner) {
        return new ApplicationRunner() {
            @Override
            public void run(ApplicationArguments args) {
                runner.run(args.getNonOptionArgs());
            }
        };
    }
}
