# Setting up AI Core

In order to use AI Core in SAP Business Technology Platform (BTP), you need to create an instance of the service in your BTP subaccount.  This can be done using the BTP cockpit or the BTP CLI.  The following steps will guide you through the process of creating an AI Core service instance and binding it to your application.

## Create an AI Core Service Instance

- **Create an AI Core Service Instance**:
  - Go to the SAP BTP cockpit and navigate to the subaccount where you want to create the service instance.
  - Click on "Services" and search for "AI Core".
  - Click on "Create" and fill in the required details (e.g., name, plan, etc.).
  - Once created, note down the service instance name and credentials.  For the examples in this repository, the service instance name is `best-practices-aicore`, the type is `aicore` and the service plan is `extended`.  The service instance will be used to connect to the AI Core service.

    > **Note:** The `extended` plan is required for the generative AI hub, including the orchestration service, foundation models and SAP-RPT models used throughout this repository.

  This can be done using the cloud foundry CLI:

  ```bash
  cf create-service aicore extended best-practices-aicore
  ```

- **Create a Service Key**:
  - After creating the service instance, create a service key to obtain the credentials needed to connect to the AI Core service.
  - Go to the service instance details and click on "Create Service Key".
  - Note down the generated service key, which contains the credentials (client ID, client secret, etc.) needed to connect to the AI Core service.

  This can be done using the cloud foundry CLI:

  ```bash
  cf create-service-key best-practices-aicore best-practices-aicore-key
  ```

## Using a Destination for AI Core

There are several ways to establish a connection to AI Core.  Most of the samples included in this repository are using either an application binding in BTP, or a service key.  

A service key is a JSON object that contains the credentials needed to connect to a service.  This is typically used for local development, or when you want to use the service in a different environment than it was created in.

If you want to use a destination to connect to an AI Core instance in another environment (BTP subaccount or space), you can do so by creating a destination in the BTP cockpit.  This destination will contain the credentials needed to connect to the AI Core instance, and will be used by the application to connect to the service.

Once the destination is created, you can use it in your application by specifying the destination name in the AI Core SDK configuration.  The SDK will then use the destination to connect to the AI Core instance.

### Implementation Steps

1. **Create a destination in the BTP cockpit**:
   - Go to the BTP cockpit and navigate to the destination configuration under Connectivity for the subaccount where the application will be deployed.
   - Create a new destination with the following properties (replace the placeholders with the values from your service key):
        - **Name**: `my-aicore`
        - **Type**: `HTTP`
        - **URL**: `[serviceurls.AI_API_URL]`
        - **Proxy Type**: `Internet`
        - **Authentication**: `OAuth2ClientCredentials`
        - **Client ID**: `[clientid]`
        - **Client Secret**: `[clientsecret]`
        - **Token Service URL Type**: `Dedicated`
        - **Token Service URL**: `[url]/oauth/token`
   - Save the destination.

        **[More Info](https://sap.github.io/ai-sdk/docs/js/connecting-to-ai-core#using-a-destination)**
2. **Bind to the destination service, instead of AI Core**:
    - This allows your application to use the destination service.  It can use destinations defined either in the destination instance, or at the subaccount level.

    - If using manifest.yml:

        ```yaml
        applications:
          - name: <your-app-name>
            ...
            services:
              - best-practices-ai-core  --->> CHANGE TO -->> best-practices-destination
        ```

        You will need to manually create the destination service instance in the BTP cockpit or using the CLI.  The destination service instance will be used to connect to the AI Core instance.

        - If using the Cloud Foundry CLI, you can create the destination service instance with the following command:

            ```bash
            cf create-service destination lite best-practices-destination
            ```

    - If using mta.yaml:
        - Remove this section:

        ```yaml
        resources:
            - name: best-practices-aicore
              type: org.cloudfoundry.existing-service
              parameters:
                service-name: best-practices-aicore
        ```

        - Add this section:

        ```yaml
        resources:
            - name: best-practices-destination
              type: org.cloudfoundry.managed-service
              parameters:
                service: destination
                service-plan: lite
        ```

3. **Update your application to use the destination**:
    - In your application code, update the AI Core SDK configuration to use the destination name instead of the service key.

    For example:

    ```typescript
        private readonly getEmbeddingClient = (resourceGroupId: string) => {
            return new AzureOpenAiEmbeddingClient(
            {
                modelName: "text-embedding-3-small",
            },
            { destinationName: "my-aicore" }
            );
        };
    ```

    Another example for orchestration:

    ```typescript
        const client = new OrchestrationClient( 
            orchestrationConfig, langchainOptions, deploymentConfig, 
            { destinationName: 'my-aicore' }
        );
    ```
