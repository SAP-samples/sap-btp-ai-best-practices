# Purchase Order Extractor

A SAP Cloud Application Programming Model (CAP) application that extracts and processes purchase order data from PDF documents using SAP Document AI (DOX) and AI Core services.

## Overview

This application automates the extraction of purchase order information from PDF documents and provides intelligent material number mapping between customer and supplier catalogs. It features two Fiori applications for managing purchase order extraction and material mapping workflows.

## Key Features

- **Automated PO Extraction**: Extract purchase order header and line item data from PDF documents using SAP Document AI - Premium Edition
- **AI-Powered Material Mapping**: Leverage SAP Generative AI Hub to intelligently map customer material numbers to supplier material numbers
- **Sales Order Integration**: Match purchase order line items with sales order items
- **Customer & Material Management**: Maintain customer information and customer-material-internal-reference (CMIR) mappings
- **Dual Fiori Applications**:
  - Purchase Order Extraction - Main interface for PO document processing
  - PO-SO Material Mapping - Interface for managing material and sales order mappings
- **Document Preview**: View PDF purchase orders directly within the application

## Architecture

### Technology Stack

- **Backend**: SAP CAP with TypeScript
- **Database**: SAP HANA (HDI Container)
- **Frontend**: SAP Fiori Elements (UI5)
- **AI Services**: 
  - SAP Document AI (Premium Edition)
  - SAP Generative AI Hub
- **Security**: SAP XSUAA (User Authentication & Authorization)
- **Deployment**: Multi-Target Application (MTA) on SAP Business Technology Platform

### Core Components

- `db/` - Data model and database schema definitions
- `srv/` - Service layer with TypeScript implementation
- `app/` - Fiori applications
  - `purchaseorderextraction/` - Main PO extraction UI
  - `po-so-material-mapping/` - Material mapping UI

## Prerequisites

- Node.js LTS version (see [nodejs.org](https://nodejs.org))
- SAP Business Application Studio or VS Code with SAP extensions
- SAP Cloud Foundry CLI (`cf`) - [Installation Guide](https://docs.cloudfoundry.org/cf-cli/install-go-cli.html)
- Cloud MTA Build Tool (`mbt`) - [Installation Guide](https://sap.github.io/cloud-mta-build-tool/)
- Access to SAP BTP with the following services:
  - SAP HANA Cloud
  - SAP Document AI (Premium Edition)
  - SAP AI Core with Generative AI Hub
  - Destination service
  - XSUAA service

## Installation & Setup

1. **Clone the repository** (if applicable) or open the project in SAP Business Application Studio

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure destinations**: Set up the following destinations in your BTP subaccount:
   - `GENERATIVE_AI_HUB` - For AI Core integration
   - `DOX-PREMIUM` - For Document AI

4. **Update configuration** in `package.json`:
   - Update the `DEPLOYMENT_ID`, `API_VERSION` in the GENERATIVE_AI_HUB configuration for the model you want to use
   - Update the `CLIENT` in the DOX-PREMIUM configuration

## Project Structure

```
purchase-order-extractor/
├── app/                                  # Fiori applications
│   ├── purchaseorderextraction/          # Main PO extraction app
│   └── po-so-material-mapping/           # Material mapping app
├── db/                                   # Database artifacts
│   ├── schema.cds                        # Data model definitions
│   └── src/                              # HDI configuration
├── srv/                                  # Service layer
│   ├── service.cds                       # Service definitions
│   ├── service.ts                        # Service implementation
│   └── utils/                            # Utility modules
│       └── AICore.ts                     # AI service integration
├── mta.yaml                              # MTA deployment descriptor
├── package.json                          # Project dependencies and scripts
├── xs-security.json                      # Security configuration
└── README.md                             # This file
```

## Development

### Local Development with CDS Watch

The **CDS Development Kit** provides the `cds watch` command for local development with automatic live reload:

```bash
cds watch
```

**What `cds watch` does:**
- Starts the CAP server in development mode
- Automatically restarts on file changes
- Provides an index page with links to all services and apps
- Uses SQLite for local database (no HANA required)
- Enables mock authentication for testing

Access the applications:
- Service index: http://localhost:4004
- Purchase Order Extraction: http://localhost:4004/purchaseorderextraction/webapp/index.html
- PO-SO Material Mapping: http://localhost:4004/po-so-material-mapping/webapp/index.html

### Alternative: Watch Specific Applications

You can also use the npm scripts to open specific apps directly:

```bash
# Watch and open Purchase Order Extraction app
npm run watch-purchaseorderextraction

# Watch and open PO-SO Material Mapping app
npm run watch-po-so-material-mapping
```

## Deployment

### Build the MTA Archive with MBT

The **Cloud MTA Build Tool (mbt)** is the recommended tool for building Multi-Target Applications:

```bash
mbt build
```

**What `mbt build` does:**
- Validates the `mta.yaml` descriptor
- Executes build scripts for all modules (Node.js, UI5 apps, database)
- Generates optimized production artifacts
- Packages everything into an `.mtar` archive in `mta_archives/`
- Handles dependencies and build order automatically

The build process creates `mta_archives/archive.mtar` which contains all deployable artifacts.

### Deploy to Cloud Foundry with CF CLI

Use the **Cloud Foundry CLI** to deploy the MTA archive:

```bash
cf deploy mta_archives/archive.mtar --retries 1
```

**What `cf deploy` does:**
- Authenticates with Cloud Foundry (run `cf login` first if needed)
- Creates/updates all required service instances (HANA, XSUAA, Destination, etc.)
- Deploys all application modules
- Configures routes and bindings
- Performs blue-green deployment for zero-downtime updates

**Before deploying:**
1. Login to Cloud Foundry:
   ```bash
   cf login -a <api-endpoint> -o <org> -s <space>
   ```

2. Verify you're in the correct space:
   ```bash
   cf target
   ```

### Undeploy

To remove the application and optionally delete services:

```bash
cf undeploy purchase-order-extractor --delete-services --delete-service-keys --delete-service-brokers
```

## Configuration

### Required Service Instances

The application requires the following BTP services (automatically created during deployment):

- **XSUAA** (xsuaa/application) - User authentication
- **HANA HDI Container** (hana/hdi-shared) - Database
- **Destination** (destination/lite) - External service connections
- **HTML5 Application Repository** (html5-apps-repo) - App hosting

### Destination Configuration

Configure these destinations in your BTP cockpit:

1. **GENERATIVE_AI_HUB**
   - Type: HTTP
   - URL: Your AI Core service URL
   - Authentication: OAuth2ClientCredentials
   - Additional Properties: Resource Group, Deployment ID

2. **DOX-PREMIUM**
   - Type: HTTP
   - URL: Your Document AI service URL
   - Authentication: OAuth2ClientCredentials
   - Additional Properties: Client ID

## Applications

### Purchase Order Extraction

The main application for extracting and managing purchase order data from PDF documents.

**Features:**
- Upload and process PO PDF documents
- Review extracted header and line item data
- Validate extraction accuracy
- Associate POs with customers
- Map line items to SAP material numbers
- Track payment and review status

### PO-SO Material Mapping

Application for managing the relationship between purchase orders, sales orders, and material mappings.

**Features:**
- View purchase order line items
- Match line items with sales order items
- Manage customer material mappings (CMIR)
- View and select material mapping candidates
- Automated AI-powered material suggestions

## Services

### DocumentService

Main OData V4 service exposing the following entities:

- **PurchaseOrders** - PO header information with draft support
- **LineItems** - PO line item details
- **Customers** - Customer master data
- **CMIRMappings** - Customer material to internal reference mappings
- **SalesOrders** - Sales order headers
- **SalesOrderItems** - Sales order line items
- **LineItemSalesOrderItemCandidates** - PO-SO line item matching candidates

**AI-Powered Actions:**
- `SyncDOX()` - Sync and extract data from Document AI service
- `generateMapping()` - Generate material mappings using AI

## Data Model

### Core Entities

**PurchaseOrders**
- Header fields: document number, dates, amounts, sender/ship-to information
- Status tracking: extraction review status, payment status
- Relationships: line items, customer, sales orders

**LineItems**
- Line details: description, quantity, price, material numbers
- Relationships: parent PO, material mapping candidates, SO mappings

**Customers**
- Customer master data with address information
- Relationships to POs and material mappings

**CMIRMappings**
- Maps customer material numbers to supplier material numbers
- Enables consistent material identification across systems

**SalesOrders & SalesOrderItems**
- Sales order information linked to purchase orders
- Enables PO-SO reconciliation and fulfillment tracking

## Development Tools Reference

### CDS Command Line Interface
The `@sap/cds-dk` package provides the CDS development tools:
- `cds watch` - Development server with live reload
- `cds build` - Build for production
- `cds deploy` - Deploy database artifacts
- `cds compile` - Compile CDS models

### Cloud MTA Build Tool (mbt)
Multi-Target Application build tool that:
- Orchestrates complex multi-module builds
- Handles different technology stacks (Node.js, Java, UI5, etc.)
- Creates deployment-ready `.mtar` archives
- Optimizes artifacts for cloud deployment

### Cloud Foundry CLI (cf)
Command-line interface for Cloud Foundry:
- `cf login` - Authenticate to Cloud Foundry
- `cf target` - View/set target org and space
- `cf deploy` - Deploy MTA archives
- `cf apps` - List deployed applications
- `cf services` - List service instances

## Security Configuration

This section explains how to configure security settings for production deployments. The current configuration is optimized for development with relaxed security settings.

### Restricting iframe Loading in index.html Files
We load our demo applications in iFrames on our AI4U Demo page, so we have enabled this setting for non-productive use.

**Current State (Development):**
 Both UI applications currently allow iframe embedding with the following configuration in their `index.html` files:

```html
<script
    id="sap-ui-bootstrap"
    src="https://sapui5.hana.ondemand.com/1.141.1/resources/sap-ui-core.js"
    ...
    data-sap-ui-frame-options="allow"
></script>
```

**Security Risk:** The `data-sap-ui-frame-options="allow"` setting permits the application to be loaded in iframes from any origin, which can expose the application to clickjacking attacks. 

**Production Configuration:**

To restrict iframe loading and improve security:

1. **Option 1: Remove the attribute completely** (Recommended for most cases)
   
   Remove the `data-sap-ui-frame-options="allow"` line entirely from both files:
   - `app/purchaseorderextraction/webapp/index.html`
   - `app/po-so-material-mapping/webapp/index.html`
   
   This will use the default UI5 behavior which blocks iframe loading unless from trusted origins.

2. **Option 2: Use trusted origins**
   
   If you need iframe support from specific origins, use:
   ```html
   data-sap-ui-frame-options="trusted"
   ```
   
   Then configure trusted origins in the UI5 application's `manifest.json` or through CSP headers.

3. **Option 3: Explicitly deny all iframes**
   
   To completely block iframe loading:
   ```html
   data-sap-ui-frame-options="deny"
   ```

**When to use each option:**
- **Development:** Use `"allow"` for maximum flexibility
- **Production:** Remove the attribute or use `"trusted"` with explicit origin configuration
- **High Security:** Use `"deny"` if iframe embedding is never required

### Enforcing Authentication on Services

#### 1. Enable Authentication in CDS Service

**Current State (Development):**
The `srv/service.cds` file has authentication disabled:

```cds
// annotate DocumentService with @requires :
// [
//     'authenticated-user'
// ];
```

**Production Configuration:**

Uncomment and enable the authentication requirement:

```cds
annotate DocumentService with @requires :
[
    'authenticated-user'
];
```

**What this does:**
- Forces all service requests to require an authenticated user
- Integrates with XSUAA for user authentication
- Returns 401 Unauthorized for unauthenticated requests
- Enforces role-based access control when configured

**Alternative configurations:**

For role-based access, specify required roles or scopes:
```cds
annotate DocumentService with @requires :
[
    'system-user',  // Specific role
    'Viewer'        // Custom role from xs-security.json
];
```

#### 2. Enable Authentication in xs-app.json Files

**Current State (Development):**
Both xs-app.json files have ALL routes configured without authentication, including the catch-all route for the UI:

```json
{
  "routes": [
    {
      "source": "^/?service/(.*)$",
      "authenticationType": "none",
      "csrfProtection": false
    },
    {
      "source": "^(.*)$",
      "service": "html5-apps-repo-rt",
      "authenticationType": "none"
    }
  ]
}
```

**Production Configuration:**

Update ALL routes in **both** xs-app.json files to require authentication:

```json
{
  "routes": [
    {
      "source": "^/?service/(.*)$",
      "target": "/service/$1",
      "destination": "srv-api",
      "authenticationType": "xsuaa",
      "csrfProtection": true
    },
    {
      "source": "^(.*)$",
      "target": "$1",
      "service": "html5-apps-repo-rt",
      "authenticationType": "xsuaa"
    }
  ]
}
```

**Changes explained:**
- Change `"authenticationType": "none"` to `"xsuaa"` on ALL routes
- Enable `"csrfProtection": true` for service routes
- This ensures both the UI and backend services require authentication

#### 3. Configure Authentication in package.json
We removed authentication to allow public access on our demo page. You should always configure proper authentication for production environments.

**Current State (Development):**
The `package.json` file has authentication configured with environment-specific profiles:

```json
"cds": {
  "requires": {
    "auth": {
      "[development]": {
        "kind": "dummy"
      },
      "[production]": {
        "kind": "xsuaa",
        "restrict_all_services": false
      }
    },
    "[production]": {
      "db": "hana"
    }
  }
}
```

**Production Configuration:**

Remove the `"restrict_all_services": false` setting:

```json
"cds": {
  "requires": {
    "auth": {
      "[development]": {
        "kind": "dummy"
      },
      "[production]": {
        "kind": "xsuaa"
      }
    },
    "[production]": {
      "db": "hana"
    }
  }
}
```

**What this does:**
- `[development]` profile uses `"kind": "dummy"` for mock authentication during local development
- `[production]` profile uses `"kind": "xsuaa"` for real authentication
- The `restrict_all_services` setting defaults to `true` when not specified
- Removing `"restrict_all_services": false` ensures that service-level security annotations (like `@requires: ['authenticated-user']`) are properly enforced


### Complete Security Checklist for Production

- [ ] Remove or change `data-sap-ui-frame-options="allow"` in both index.html files
- [ ] Uncomment the `@requires: ['authenticated-user']` annotation in `srv/service.cds`
- [ ] Remove `"restrict_all_services": false` from the auth `[production]` profile in `package.json`
- [ ] Change ALL routes from `"authenticationType": "none"` to `"xsuaa"` in both xs-app.json files
- [ ] Enable `"csrfProtection": true` for service routes in both xs-app.json files
- [ ] Test authentication flow with actual XSUAA service binding
- [ ] Configure appropriate roles and scopes in `xs-security.json`
- [ ] Verify role assignments for users in BTP Cockpit

### Impact on Development Workflow

**Important:** Enabling these security settings will require authentication during local development with `cds watch`. You can refer to CAP documentation for default test users, e.g. 'privileged', 'admin'.

To maintain development convenience:
1. Keep separate configurations for development vs. production
2. Use environment-specific profiles in `package.json`
3. Leverage CDS mock authentication during local development
4. Consider using feature toggles or build-time configuration

For local development, you can keep the current relaxed security settings and only apply the production configuration before deployment.

## Learn More

- [SAP Cloud Application Programming Model](https://cap.cloud.sap/docs/)
- [SAP Document AI](https://help.sap.com/docs/document-ai)
- [SAP AI Core](https://help.sap.com/docs/sap-ai-core)
- [SAP Fiori Elements](https://ui5.sap.com/test-resources/sap/fe/core/fpmExplorer/index.html)
- [Cloud MTA Build Tool](https://sap.github.io/cloud-mta-build-tool/)
- [Cloud Foundry Documentation](https://docs.cloudfoundry.org/)
- [SAP Business Technology Platform](https://help.sap.com/docs/btp)
- [UI5 Frame Options](https://ui5.sap.com/#/topic/62d9c4d8f5ad49aa914624af9551beb7)
- [CAP Security Guide](https://cap.cloud.sap/docs/guides/security/)
