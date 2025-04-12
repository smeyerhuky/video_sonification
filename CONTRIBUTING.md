# Contributing to Video Sonification Project

Thank you for your interest in contributing to the Video Sonification project! This document provides guidelines and instructions for contributing to various aspects of the project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Contribution Workflow](#contribution-workflow)
- [Component-Specific Guidelines](#component-specific-guidelines)
  - [Frontend](#frontend)
  - [Microservices](#microservices)
  - [Thrift Definitions](#thrift-definitions)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Guidelines](#testing-guidelines)
- [Accessibility Requirements](#accessibility-requirements)
- [Documentation](#documentation)
- [Performance Considerations](#performance-considerations)
- [Communication Channels](#communication-channels)

## Development Environment Setup

1. **Prerequisites:**
   - Node.js (see frontend/.nvmrc for specific version)
   - Docker and docker-compose
   - Go (for microservices development)
   - Python 3.8+ (for microservices development)
   - Apache Thrift compiler (for interface development)

2. **Initial Setup:**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd video_sonification

   # Start all services using Docker
   docker-compose up
   
   # For frontend-only development (faster iterative development)
   cd frontend
   yarn install
   yarn dev
   ```

3. **Environment Configuration:**
   - Copy any example environment files (.env.example) to .env and configure as needed
   - Set up any required API keys or service credentials

## Contribution Workflow

1. **Find or Create an Issue:**
   - Check existing issues or create a new one describing the feature/bug
   - Get assignment or approval from maintainers before starting work

2. **Branching Strategy:**
   - Create branches from `main` using the format:
     - `feature/short-description` (for new features)
     - `fix/short-description` (for bug fixes)
     - `docs/short-description` (for documentation)
     - `refactor/short-description` (for code refactoring)

3. **Development Process:**
   - Make focused, related changes in each pull request
   - Keep commits atomic and use meaningful commit messages
   - Reference issue numbers in commit messages

4. **Pull Request Process:**
   - Create a pull request to `main` branch
   - Fill in the pull request template completely
   - Ensure all CI checks pass
   - Request review from appropriate team members
   - Address review comments and update PR

5. **Merge and Cleanup:**
   - Maintainers will merge approved PRs
   - Delete branches after merging
   - Close related issues

## Component-Specific Guidelines

### Frontend

1. **Architecture:**
   - Follow the established component structure
   - Use shadcn UI components when available
   - Place shared components in appropriate directories

2. **State Management:**
   - Use React hooks for local state
   - Consider context API for shared state
   - Document complex state interactions

3. **Performance:**
   - Use React.memo and useMemo/useCallback when appropriate
   - Consider chunking for large component files
   - Optimize render cycles for audio/visual processing

4. **Media Handling:**
   - Follow established patterns for video input handling
   - Consider memory usage when processing video frames
   - Use Web Workers for intensive processing

### Microservices

1. **Service Structure:**
   - Each service should have a clear, focused responsibility
   - Follow the established patterns in existing services
   - Document service APIs and dependencies

2. **Language Considerations:**
   - Go services: Follow Go best practices and idioms
   - Python services: Follow PEP 8 guidelines
   - Ensure proper error handling and logging

3. **Docker Integration:**
   - Update Dockerfiles and docker-compose.yml when adding dependencies
   - Optimize container size and build time
   - Document environment variables and configuration

4. **API Design:**
   - Ensure compatibility with Thrift interfaces
   - Design for performance with high-throughput data
   - Consider backward compatibility

### Thrift Definitions

1. **Interface Design:**
   - Keep interfaces focused and cohesive
   - Document all structs, services, and methods
   - Consider versioning for future compatibility

2. **Data Types:**
   - Use appropriate types for data efficiency
   - Consider binary formats for media data
   - Document constraints and validation requirements

3. **Development Process:**
   - Regenerate client/server code after interface changes
   - Update dependent services to use new interfaces
   - Test interface changes thoroughly

## Code Style and Standards

1. **General Guidelines:**
   - Use meaningful variable and function names
   - Add comments for complex logic
   - Keep functions and methods focused
   - Follow DRY (Don't Repeat Yourself) principles

2. **Language-Specific Standards:**
   - JavaScript/TypeScript: Follow ESLint configuration
   - Go: Use `go fmt` and `golint`
   - Python: Follow PEP 8 guidelines
   - Thrift: Follow documentation conventions

3. **Formatting:**
   - Use consistent indentation (2 spaces for JS/TS, standard for other languages)
   - Limit line length to 100 characters when practical
   - Use consistent bracket and bracing styles

## Testing Guidelines

1. **Testing Requirements:**
   - Write tests for new features and bug fixes
   - Maintain or improve code coverage
   - Test edge cases and failure scenarios

2. **Testing Types:**
   - Unit tests for functions and components
   - Integration tests for service interactions
   - Performance tests for processing-intensive operations

3. **Media Testing:**
   - Include test media files when appropriate
   - Document test scenarios for video analysis
   - Test with various input types and formats

## Accessibility Requirements

1. **Frontend Components:**
   - Ensure WCAG 2.1 AA compliance for all UI components
   - Use appropriate ARIA attributes
   - Support keyboard navigation
   - Ensure sufficient color contrast

2. **Media Controls:**
   - Provide accessible alternatives for media content
   - Ensure all controls can be operated via keyboard
   - Include appropriate labels and instructions

3. **Feedback and Notifications:**
   - Ensure screen reader compatibility for notifications
   - Provide multiple forms of feedback (visual, audio)
   - Document accessibility features

## Documentation

1. **Code Documentation:**
   - Document functions, components, and classes
   - Explain complex algorithms or business logic
   - Update documentation when changing code

2. **User Documentation:**
   - Update README.md with new features
   - Provide usage examples
   - Include screenshots or diagrams when helpful

3. **API Documentation:**
   - Document all API endpoints and services
   - Include request/response examples
   - Document error cases and handling

## Performance Considerations

1. **Video Processing:**
   - Consider memory usage when handling video frames
   - Use efficient algorithms for video analysis
   - Document performance characteristics and limitations

2. **Audio Generation:**
   - Optimize audio synthesis for real-time performance
   - Consider Web Audio API best practices
   - Test on various devices for performance issues

3. **Data Transfer:**
   - Minimize data transfer between services
   - Use appropriate serialization formats
   - Consider compression for media data

## Communication Channels

1. **Issue Tracker:**
   - Use GitHub Issues for bugs and feature requests
   - Follow issue templates and guidelines
   - Link issues to related pull requests

2. **Discussions:**
   - Use GitHub Discussions for architecture and design discussions
   - Participate in code reviews constructively
   - Share knowledge and help other contributors

3. **Community:**
   - Be respectful and inclusive in all communications
   - Help new contributors get started
   - Recognize and appreciate all types of contributions

---

Thank you for contributing to the Video Sonification project! Your efforts help create an innovative tool for transforming visual content into captivating audio experiences.