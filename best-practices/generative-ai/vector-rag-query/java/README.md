# AI Model Access Demo

This project demonstrates best practices for retrieving documents in semantic search when implementing Retrieval Augmented Generation.
## Project Structure

```
vector-rag-query/java
├── sample
│   ├── src
|   |   └── main
|   |       └── java
|   |           └── com
|   |               └── sap
|   |                   └── ai
|   |                       └── core
|   |                           └── sample
|   |                               └── rag
|   |                                   └── App.java  # Main application
│   └── target
|       ├── classes
|       |   └── com
|       |       └── sap
|       |           └── ai
|       |               └── core
|       |                   └── sample
|       |                       └── rag
|       |                           └── App.class
|       └── com
|               └── sap
|                   └── ai
|                       └── core
|                           └── sample
|                               └── rag
|                                   └── App.class
├── .env                            # Template for environment variables
├── .gitignore                      # Specifies files to ignore in Git
├── pom.xml                         # mvn dependancies and project configuration
└── README.md                       # Project documentation

```

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SAP-samples/sap-btp-ai-best-practices.git
   cd sap-btp-ai-best-practices/best-practices/vector-rag-query/java
   ```

2. **Configure environment variables:**

   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Populate the `.env` file with the required values.

3. **Install dependencies and run:**
   ```
   mvn clean install
   ```

## Usage Example

The application will run the main function, which retrieves information from a file in local directory and uses it in a prompt.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
