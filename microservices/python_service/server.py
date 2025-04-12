#!/usr/bin/env python3
import os
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample data (in a real implementation, this would come from a database)
data_items = [
    {
        "id": 1,
        "name": "Item 1",
        "description": "This is the first item from the Python service",
        "value": 10.5,
        "metadata": {"source": "python", "category": "sample"}
    },
    {
        "id": 2,
        "name": "Item 2",
        "description": "This is the second item from the Python service",
        "value": 20.75,
        "metadata": {"source": "python", "category": "sample"}
    },
    {
        "id": 3,
        "name": "Item 3",
        "description": "This is the third item from the Python service",
        "value": 30.0,
        "metadata": {"source": "python", "category": "test"}
    }
]

# Get service metadata
@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify({
        "service_name": os.environ.get('SERVICE_NAME', 'python-data-service'),
        "version": "1.0.0",
        "language": "Python",
        "framework": "Flask",
        "timestamp": "2025-04-05"
    })

# Get all data items
@app.route('/data', methods=['GET'])
def get_all_data():
    return jsonify({
        "success": True,
        "message": "Data retrieved successfully",
        "items": data_items
    })

# Get data item by ID
@app.route('/data/<int:item_id>', methods=['GET'])
def get_data_by_id(item_id):
    for item in data_items:
        if item['id'] == item_id:
            return jsonify({
                "success": True,
                "message": f"Item {item_id} retrieved successfully",
                "item": item
            })
    
    return jsonify({
        "success": False,
        "message": f"Item {item_id} not found"
    }), 404

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('SERVICE_PORT', 5000))
    app.run(host='0.0.0.0', port=port)