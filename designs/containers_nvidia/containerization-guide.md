# React/Vite Application Containerization Guide - Development Environment

This guide explains the containerization setup for the React/Vite application's development environment and provides instructions on how to use it.

## Containerization Approach

We've implemented a Docker-based development environment with the following features:

- **Hot-reloading** for rapid development
- **Volume mounting** for real-time code changes
- **Proper base path handling** for the `/playground/` path
- **Consistent environment** across all development machines

## Docker Configuration Files

The containerization setup consists of three main files:

1. **Dockerfile**: Contains the build instructions for the development container
2. **docker-compose.yml**: Makes it easy to run the application in development
3. **.dockerignore**: Optimizes build performance by excluding unnecessary files

## Development Workflow

To run the application in development mode with hot-reloading:

```bash
docker compose up dev
```

This will:
- Build the development image if it doesn't exist
- Mount your local source code as a volume for hot-reloading
- Start the development server on port 5173
- Enable hot-reloading so your changes appear instantly

### After Dependency Changes

If you've added or updated dependencies in package.json, you'll need to rebuild the container:

```bash
docker compose build dev
docker compose up dev
```

## Common Docker Commands

Here are some common commands you might use during development:

### View Running Containers
```bash
docker ps
```

### View Container Logs
```bash
docker logs playground-dev
```

### Execute Commands in a Running Container
```bash
# Run a terminal in the development container
docker exec -it playground-dev sh

# Run a specific command (e.g., lint check)
docker exec playground-dev yarn lint
```

### Stop Containers
```bash
docker compose down
```

### Rebuild Images and Remove Volumes
```bash
docker compose down -v
docker compose build --no-cache
```

## Handling Base Paths in React/Vite Applications

React/Vite applications often use a base path configuration, which is specified in the `vite.config.js` file. This is particularly important when:
- The application is not served from the root of a domain
- The application is deployed to a subdirectory
- The application is part of a larger system with its own routing

In our project, we use the base path `/playground/` as specified in the `vite.config.js`:

```javascript
// vite.config.js
export default defineConfig({
  // ...
  base: '/playground/',
  // ...
});
```

### Base Path Considerations in Docker

When containerizing a React/Vite application with a custom base path, you must ensure that the Vite dev server is started with the same base path.

Failure to properly configure base paths often results in these errors:

```
Loading module was blocked because of a disallowed MIME type ("text/html")
Loading failed for the module with source "http://localhost:5173/playground/assets/index-XXXX.js"
The stylesheet was not loaded because its MIME type, "text/html", is not "text/css"
```

This happens because the server is returning the index.html file instead of the requested assets, as it doesn't recognize the base path.

### Development Environment Configuration

For the development environment, we pass the base path to the Vite dev server:

```dockerfile
# In Dockerfile
CMD ["yarn", "dev", "--host", "0.0.0.0", "--base=/playground/"]
```

Or in docker-compose.yml:

```yaml
command: yarn dev --host 0.0.0.0 --base=/playground/
```

### Testing Base Path Configuration

To verify your base path configuration is working:

1. Start your container: `docker compose up dev`
2. Navigate to: `http://localhost:5173/playground/`
3. Check browser developer tools for any 404 errors or MIME type errors

## Troubleshooting

### Container Exits Immediately

If the container exits immediately after starting:
- Check container logs: `docker logs playground-dev`
- Ensure ports aren't already in use on your machine
- Verify node_modules volume is properly mounted

### Hot Reloading Not Working

If changes aren't being detected:
- Ensure CHOKIDAR_USEPOLLING=true is set in the environment
- Check that the correct directories are mounted as volumes
- Restart the container

### Build Errors

If you encounter build errors:
- Check for conflicting dependencies
- Ensure your Docker version is up to date
- Try clearing Docker's build cache: `docker system prune`

## Environment Variables

You can pass environment variables to the development container through the docker-compose.yml file:

```yaml
environment:
  - NODE_ENV=development
  - CHOKIDAR_USEPOLLING=true
  - API_URL=https://api.example.com
```

Or by using a .env file (create it in the project root).