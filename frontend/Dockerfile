# Development container for React/Vite application
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and yarn.lock
COPY package.json yarn.lock ./

# Install dependencies
RUN yarn install

# Copy source code
COPY . .

# Expose port
EXPOSE 5173

# Set environment variable
ENV NODE_ENV=development

# Command to run the application in development mode
# Explicitly set the base path for development
CMD ["yarn", "dev", "--host", "0.0.0.0", "--base=/video_sonification/"]