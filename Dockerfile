# Stage 1: Build stage
FROM maven:3.8.6-openjdk-11-slim AS build

WORKDIR /app

# Copy pom.xml first to leverage Docker cache
COPY pom.xml .
COPY src ./src

# Build the application
RUN mvn clean package -DskipTests

# Stage 2: Runtime stage
FROM openjdk:11-jre-slim

WORKDIR /app

# Copy the built JAR from build stage
COPY --from=build /app/target/*.jar app.jar

# Run the application
# Render provides PORT env var - use it directly and bind to all interfaces
ENTRYPOINT ["sh", "-c", "java -jar -Dserver.port=${PORT} -Dserver.address=0.0.0.0 app.jar"]
