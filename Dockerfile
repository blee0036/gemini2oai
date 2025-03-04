FROM golang:1.20-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o gemini2oai .

FROM alpine:latest
WORKDIR /root/
COPY --from=builder /app/gemini2oai .
EXPOSE 3000

CMD ["./gemini2oai"]