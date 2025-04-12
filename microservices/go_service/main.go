package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

// DataItem represents a data item as defined in the Thrift IDL
type DataItem struct {
	ID          int               `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Value       float64           `json:"value"`
	Metadata    map[string]string `json:"metadata"`
}

// DataResponse represents a response as defined in the Thrift IDL
type DataResponse struct {
	Success bool       `json:"success"`
	Message string     `json:"message"`
	Item    *DataItem  `json:"item,omitempty"`
	Items   []DataItem `json:"items,omitempty"`
}

// Sample data (in a real implementation, this would come from a database)
var dataItems = []DataItem{
	{
		ID:          1,
		Name:        "Item 1",
		Description: "This is the first item from the Go service",
		Value:       10.5,
		Metadata:    map[string]string{"source": "go", "category": "sample"},
	},
	{
		ID:          2,
		Name:        "Item 2",
		Description: "This is the second item from the Go service",
		Value:       20.75,
		Metadata:    map[string]string{"source": "go", "category": "sample"},
	},
	{
		ID:          3,
		Name:        "Item 3",
		Description: "This is the third item from the Go service",
		Value:       30.0,
		Metadata:    map[string]string{"source": "go", "category": "test"},
	},
}

func main() {
	// Create a new router
	r := mux.NewRouter()

	// Define routes
	r.HandleFunc("/metadata", getMetadata).Methods("GET")
	r.HandleFunc("/data", getAllData).Methods("GET")
	r.HandleFunc("/data/{id:[0-9]+}", getDataByID).Methods("GET")
	r.HandleFunc("/health", healthCheck).Methods("GET")

	// Set up CORS
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
	})
	handler := c.Handler(r)

	// Get port from environment variable or use default
	port := os.Getenv("SERVICE_PORT")
	if port == "" {
		port = "5001"
	}

	// Start the server
	fmt.Printf("Starting Go service on port %s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, handler))
}

// getMetadata returns metadata about the service
func getMetadata(w http.ResponseWriter, r *http.Request) {
	metadata := map[string]string{
		"service_name": os.Getenv("SERVICE_NAME"),
		"version":      "1.0.0",
		"language":     "Go",
		"framework":    "gorilla/mux",
		"timestamp":    "2025-04-05",
	}

	if metadata["service_name"] == "" {
		metadata["service_name"] = "go-data-service"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metadata)
}

// getAllData returns all data items
func getAllData(w http.ResponseWriter, r *http.Request) {
	response := DataResponse{
		Success: true,
		Message: "Data retrieved successfully",
		Items:   dataItems,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// getDataByID returns a specific data item by ID
func getDataByID(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}

	for _, item := range dataItems {
		if item.ID == id {
			response := DataResponse{
				Success: true,
				Message: fmt.Sprintf("Item %d retrieved successfully", id),
				Item:    &item,
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}

	// Item not found
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	json.NewEncoder(w).Encode(DataResponse{
		Success: false,
		Message: fmt.Sprintf("Item %d not found", id),
	})
}

// healthCheck returns the health status of the service
func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}