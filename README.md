# Video Sonification Project

This project combines a microservices architecture with advanced natural language processing for data visualization and sonification. It features a React frontend with NLP-based controls and multiple backend services implemented in different languages (Python and Go), all communicating through a common Thrift interface definition.

## System Architecture

```
┌─────────────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                         │      │                 │      │                 │
│  React Frontend         │──────│  Python Service │      │   Go Service    │
│  - NLP Visualization    │      │                 │      │                 │
│  - Data Sonification    │      └─────────────────┘      └─────────────────┘
│                         │               │                       │
└─────────────────────────┘               │                       │
            │                             ▼                       │
            │                    ┌─────────────────┐              │
            └────────────────────│  Thrift IDL     │──────────────┘
                                 │  Definition     │
                                 └─────────────────┘
```

The system consists of the following components:

- **Frontend**: A React/Vite application with Tailwind CSS that includes:
  - Natural language UI control system for data visualization
  - Data sonification capabilities
  - Responsive design with accessibility features
  - State management using Zustand

- **Python Service**: A Flask-based implementation of the DataService interface
- **Go Service**: A Go implementation of the same DataService interface
- **Thrift Definition**: A common interface definition that both services implement

## Directory Structure

- `frontend/`: React/Vite application with Tailwind CSS
  - `src/components/`: UI components including dashboards and NLP interface
  - `src/store/`: State management using Zustand
  - `src/hooks/`: Custom React hooks
- `microservices/python_service/`: Python implementation of the DataService
- `microservices/go_service/`: Go implementation of the DataService
- `thrift/`: Thrift interface definitions
- `guidelines/`: Project guidelines and documentation

## Quick Start

### Using the Setup Script

We provide a setup script that automates the installation and configuration process:

```bash
# Make the script executable (if needed)
chmod +x devSetup.sh

# Run the setup script
./devSetup.sh
```

The script will:
1. Check for required tools (Docker, Docker Compose)
2. Set up the frontend (install dependencies if Yarn is available)
3. Set up the Python service (create virtual environment if Python is available)
4. Set up the Go service (initialize Go module and download dependencies if Go is available)
5. Build the Docker images

### Running the System

After setup, start the system using Docker Compose:

```bash
docker-compose up
```

This will start all services:
- Frontend: http://localhost:5173/video_sonification/
- Python Service: http://localhost:5000
- Go Service: http://localhost:5001

## Manual Setup

If you prefer to set up the system manually:

### Prerequisites

- Docker and Docker Compose
- Node.js and Yarn (optional, for local frontend development)
- Python 3.9+ (optional, for local Python service development)
- Go 1.19+ (optional, for local Go service development)

### Frontend Setup

```bash
cd frontend
yarn install
```

### Python Service Setup

```bash
cd microservices/python_service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Go Service Setup

```bash
cd microservices/go_service
# Initialize Go module if it doesn't exist
if [ ! -f go.mod ]; then
    go mod init video_sonification/data_service
fi
go mod tidy
go mod download
```

### Building Docker Images

```bash
docker-compose build
```

## Testing the Services

You can test the services directly using curl:

```bash
# Get all data from Python service
curl http://localhost:5000/data

# Get all data from Go service
curl http://localhost:5001/data

# Get metadata from Python service
curl http://localhost:5000/metadata

# Get metadata from Go service
curl http://localhost:5001/metadata
```

## Features

### Frontend

- React/Vite application with hot-reloading
- Tailwind CSS for styling
- Natural language processing for UI control
- Data visualization with recharts
- Service switching between Python and Go implementations
- Responsive design for mobile and desktop
- Accessible UI with proper ARIA attributes

### Natural Language UI Control

The frontend includes a sophisticated natural language processing system that allows users to control data visualizations using plain English commands. Examples include:

- "Show bar chart"
- "Group by month"
- "Use blue colors"
- "Show monthly revenue as bar chart"

### Microservices

- Python service using Flask
- Go service using gorilla/mux
- Both implementing the same Thrift interface
- RESTful API endpoints
- CORS support for cross-origin requests

## How It Works

### Thrift Interface Definition

The core of this architecture is the Thrift IDL definition that specifies the interface both Python and Go services implement:

```thrift
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
```

This interface is implemented by both Python and Go services, demonstrating Thrift's language neutrality.

## Development

### Rebuilding After Changes

If you make changes to the code, you'll need to rebuild the Docker images:

```bash
docker-compose build
docker-compose up
```

### Adding New Services

To add a new service implementation in another language:

1. Create a new directory under `microservices/`
2. Implement the DataService interface defined in `thrift/data_service.thrift`
3. Add the service to `docker-compose.yml`
4. Update the frontend to communicate with the new service

## Troubleshooting

### Go Service Build Issues

If you encounter issues with the Go service build related to missing go.sum entries:

1. The Go service Dockerfile is configured to automatically initialize the Go module if it doesn't exist and run `go mod tidy` to generate the go.sum file. This should resolve most dependency issues automatically.

2. If you still encounter issues, you can manually fix them by running:
   ```bash
   cd microservices/go_service
   # Initialize Go module if it doesn't exist
   if [ ! -f go.mod ]; then
       go mod init video_sonification/data_service
   fi
   go mod tidy
   go mod download
   ```

3. Then rebuild the Docker images:
   ```bash
   docker-compose build
   ```

### Python Service Issues

If you encounter issues with the Python service related to Werkzeug or other dependencies:

1. The Python service requires specific versions of dependencies to work correctly. In particular, Flask 2.0.1 requires Werkzeug 2.0.x. The requirements.txt file includes these pinned versions.

2. If you encounter an error like `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`, it means there's a version mismatch. Ensure the requirements.txt file includes:
   ```
   flask==2.0.1
   flask-cors==3.0.10
   thrift==0.15.0
   gunicorn==20.1.0
   werkzeug==2.0.3
   ```

3. Then rebuild the Docker images:
   ```bash
   docker-compose build
   ```

### Frontend Issues

If you encounter issues with the frontend:

1. Make sure Tailwind CSS is properly configured:
   - Check that `tailwind.config.js` and `postcss.config.js` exist
   - Verify that the CSS file includes the Tailwind directives

2. Check for dependency issues:
   ```bash
   cd frontend
   yarn install
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.