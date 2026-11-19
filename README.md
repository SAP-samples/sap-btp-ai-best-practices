# SAP Business AI Platform Best Practices Website

This project contains the SAP Business AI Platform Best Practices website (Docusaurus frontend) and its supporting CAP backend service.

## Prerequisites

- Node.js (version `>=18.0` recommended, as specified in `app/package.json`)
- npm (usually comes with Node.js)
- Cloud Foundry CLI (required for deploying the backend service)
- Access to a HANA database (if the backend service `srv` is configured to use one)

## Project Structure

- `app/`: Contains the Docusaurus frontend application.
- `srv/`: Contains the CAP (Cloud Application Programming Model) backend service.

See `app/README.md` and `srv/README.md` for more details specific to each component.

## Development

### 1. Backend Service (`srv`)

Navigate to the `srv` directory for backend development:

```bash
cd srv
npm install
cf login -a https://api.cf.eu10-005.hana.ondemand.com/
npm run watch
```

The service will usually be available at `http://localhost:4004`.
For deployment-related commands (build, deploy), refer to `srv/README.md`.

### 2. Frontend Application (`app`)

Navigate to the `app` directory for frontend development:

```bash
cd app
npm install
# Start the development server:
npm start
```

The application will usually be available at `http://localhost:3000`.
For build and deployment instructions, refer to `app/README.md`.

## Building and Deployment

Detailed build and deployment instructions for each component can be found in their respective README files:

- **Backend Service (`srv`):** See `srv/README.md` (includes MTA build and CF deployment)
- **Frontend Application (`app`):** See `app/README.md` (includes static site build and Docusaurus deployment options)

## Further Information

- For detailed information about the CAP service, see `srv/README.md`.
- For detailed information about the Docusaurus application, see `app/README.md`.
