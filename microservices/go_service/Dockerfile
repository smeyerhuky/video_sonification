# Go DataService Implementation Dockerfile
# Uses multi-stage build for smaller final image

# Build stage
FROM golang:1.19-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy source code first (including go.mod)
COPY . .

# Initialize Modules
RUN if [ ! -f go.mod ]; then go mod init video_sonification/data_service; fi

# Update modules and download dependencies
RUN go mod tidy
RUN go mod download

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o data_service .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS connections
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from the builder stage
COPY --from=builder /app/data_service .

# Expose the service port
EXPOSE 5001

# Run the binary
CMD ["./data_service"]