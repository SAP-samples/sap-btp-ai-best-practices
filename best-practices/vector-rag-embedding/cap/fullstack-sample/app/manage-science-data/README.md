# Manage Science Data - UI5 TypeScript Application

This application demonstrates how to build a UI5 TypeScript frontend for the CAP fullstack sample that integrates with generative AI models through the SAP Cloud SDK for AI. It provides a simple user interface for querying language models about capitals of countries.

## Description

This UI5 application connects to a CAP backend service that uses the SAP Cloud SDK for AI to communicate with generative AI models. Users can enter a country name and receive the name of its capital through the AI service.

## Project Structure

```
ask-capital/
├── webapp/                  # UI5 application source code
│   ├── controller/          # UI controllers
│   ├── i18n/                # Internationalization files
│   ├── test/                # Unit and integration tests
│   ├── view/                # UI views
│   ├── Component.ts         # Application component
│   ├── index.html           # Entry point
│   └── manifest.json        # Application configuration
├── package.json             # Frontend dependencies and scripts
├── tsconfig.json            # TypeScript configuration
├── ui5.yaml                 # UI5 tooling configuration
└── README.md                # This file
```

## Prerequisites

- Node.js LTS version
- UI5 CLI (`npm install -g @ui5/cli`)
- The CAP backend service running locally or deployed

## Setup

1. Install dependencies:

```sh
npm install
```

2. Make sure the CAP backend service is running (see main README.md for instructions).

## Run the Application

For development with automatic reload on code changes:

```sh
npm start
```

This will start the UI5 application at http://localhost:8080/index.html.

## Build for Deployment

### Standard Build

For a standard build:

```sh
npm run build
```

The output will be in the `dist` folder.

### Optimized Build

For an optimized self-contained build including all UI5 resources:

```sh
npm run build:opt
```

This creates a fully deployable package in the `dist` folder.

## Debug the Application

The application includes source maps, allowing you to debug the original TypeScript code in your browser's developer tools. If breakpoints don't automatically map to TypeScript files, use `Ctrl`/`Cmd` + `P` in Chrome to open the desired `.ts` file.

## Integration with CAP Backend

This UI5 application connects to the CAP backend by calling the `/odata/v4/orchestration/askCapitalOfCountry` endpoint. The connection is configured in the `manifest.json` file.

## Contributing

If you want to contribute to this project, please follow the standard pull request process.
