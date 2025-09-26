# Getting Started

Welcome to the Log Analyzer project. This project helps to organize Large Amounts of Error logs and assess their priority. It can also generate next steps based on SAP Context. 

It contains these folders and files, following our recommended project layout:

File or Folder | Purpose
---------|----------
`app/` | content for UI frontends goes here
`db/` | your domain models and data go here
`srv/` | your service models and code go here
`package.json` | project metadata and configuration
`readme.md` | this getting started guide

## Integration

The service can be deployed to Cloud Foundry using the mta.yaml. The commands are `mbt build` and `cf deploy ./mta_archives/<your .mtar>`. 

## Security 

Remove the flag `"restrict_all_services": false` from the package.json to restrict access to authenticated users only and set up a authorization concept following the CAP framework best practices.git 

## Dependencies

You need to have a destination maintained in BTP that points to your AI Core instance. The destination name needs to be `GENERATIVE_AI_HUB`.

You need to update the LLM configuration(`DEPLOYMENT_ID`, `RESOURCE_GROUP_ID`, `API_VERSION`) in package.json:

```json
  "cds": {
    "requires": {
      "GENERATIVE_AI_HUB": {
        "kind": "rest",
        "credentials": {
          "destination": "GENERATIVE_AI_HUB",
          "requestTimeout": "300000"
        },
        "DEPLOYMENT_ID": "dc29d51f3a94d40a",
        "RESOURCE_GROUP_ID": "default",
        "API_VERSION": "2024-12-01-preview"
      },
```

## Next Steps

- Open a new terminal and run `cds watch`
- (in VS Code simply choose _**Terminal** > Run Task > cds watch_)


## Learn More

Learn more at https://cap.cloud.sap/docs/get-started/.
