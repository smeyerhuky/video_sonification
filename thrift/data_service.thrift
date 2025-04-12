// data_service.thrift
namespace py video_sonification.data
namespace go video_sonification.data

// Simple struct representing a data item
struct DataItem {
  1: i32 id,
  2: string name,
  3: string description,
  4: double value,
  5: map<string, string> metadata
}

// Response structure for service operations
struct DataResponse {
  1: bool success,
  2: string message,
  3: optional DataItem item,
  4: optional list<DataItem> items
}

// Service definition
service DataService {
  // Get all available data items
  DataResponse getAllData(),

  // Get a specific data item by ID
  DataResponse getDataById(1: i32 id),
  
  // Get metadata about the service
  map<string, string> getMetadata()
}