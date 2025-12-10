# MCP Servers Usage Guide for Cline/Claude Dev

## What is MCP (Model Context Protocol)

MCP (Model Context Protocol) is a protocol that allows AI assistants to connect to external servers and use additional tools and resources. This extends AI capabilities, enabling interaction with external APIs, databases, services, and specialized tools.

## Where to Find MCP Settings

### In Cline (VS Code Extension)
Configuration file location:
```
%APPDATA%\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json
```

### In Claude Desktop App
Configuration file locations:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## MCP Configuration Structure

```json
{
  "mcpServers": {
    "server-name": {
      "command": "command-to-run",
      "args": ["arg1", "arg2"],
      "env": {
        "API_KEY": "your-api-key"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Configuration Parameters:
- **command**: Command to run the MCP server
- **args**: Command line arguments
- **env**: Environment variables (API keys, tokens)
- **disabled**: true/false - server enabled/disabled
- **autoApprove**: List of actions for automatic approval

## Popular MCP Server Configuration Examples

### 1. SAP Fiori MCP Server
```json
{
  "mcpServers": {
    "fiori-mcp": {
      "command": "fiori-mcp",
      "args": [],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**What it provides:**
- SAP Fiori Elements application generation
- List Report and Object Page creation
- CAP project integration
- SAP Fiori documentation search

### 2. Upstash Context7 MCP Server
```json
{
  "mcpServers": {
    "github.com/upstash/context7-mcp": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@upstash/context7-mcp@latest"],
      "env": {
        "CONTEXT7_API_KEY": "your-api-key-here"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**What it provides:**
- Vector search and semantic search
- Contextual data management
- Document indexing

### 3. Weather MCP Server (example)
```json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["C:/path/to/weather-server/build/index.js"],
      "env": {
        "OPENWEATHER_API_KEY": "your-openweather-api-key"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## How to Use MCP Servers in Code

### 1. Using MCP Tools
```javascript
// In Cline use the use_mcp_tool command
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "list_functionality",
  arguments: {
    "appPath": "/path/to/your/project"
  }
})
```

### 2. Accessing MCP Resources
```javascript
// Access resources via access_mcp_resource
access_mcp_resource({
  server_name: "weather",
  uri: "weather://London/current"
})
```

## Practical Usage Examples

### Creating a Fiori Application
```javascript
// 1. Check available functionality
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "list_functionality",
  arguments: {"appPath": "./my-project"}
})

// 2. Generate application
use_mcp_tool({
  server_name: "fiori-mcp", 
  tool_name: "execute_functionality",
  arguments: {
    "functionalityId": "generate-fiori-ui-application-cap",
    "parameters": {
      "projectPath": "./my-project",
      "entitySet": "Products"
    }
  }
})
```

### Documentation Search
```javascript
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "search_docs",
  arguments: {
    "query": "List Report table configuration",
    "maxResults": 5
  }
})
```

## Installing Popular MCP Servers

### SAP Fiori MCP Server
```bash
npm install -g @sap-ux/fiori-mcp-server
```

### Installing Windows Dependencies
```bash
npm install -g @lancedb/lancedb-win32-x64-msvc
```

### Using via NPX (alternative)
```json
{
  "command": "cmd",
  "args": ["/c", "npx", "-y", "@sap-ux/fiori-mcp-server@latest"]
}
```

## Debugging and Troubleshooting

### Checking Server Status
1. Ensure the server is added to configuration
2. Verify the command runs from command line
3. Check logs in VS Code Developer Tools (F12)

### Common Issues and Solutions

#### 1. "Not connected" error
- Check command and arguments correctness
- Ensure server is installed globally
- Verify environment variables

#### 2. Dependency issues
```bash
# Clear npm cache
npm cache clean --force

# Reinstall package
npm uninstall -g package-name
npm install -g package-name
```

#### 3. Windows-specific issues
- Install Microsoft C++ Redistributable
- Use PowerShell or Command Prompt as administrator
- Check PATH variable

### Testing Connectivity
```javascript
// Connection test (replace with your server)
use_mcp_tool({
  server_name: "your-server-name",
  tool_name: "ping", // or any available tool
  arguments: {}
})
```

## Creating Your Own MCP Server

### Basic Project Structure
```
my-mcp-server/
├── package.json
├── tsconfig.json
└── src/
    └── index.ts
```

### Simple MCP Server Example
```typescript
#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema, 
  ListToolsRequestSchema 
} from '@modelcontextprotocol/sdk/types.js';

class MyMCPServer {
  private server: Server;

  constructor() {
    this.server = new Server({
      name: 'my-mcp-server',
      version: '1.0.0'
    }, {
      capabilities: { tools: {} }
    });

    this.setupToolHandlers();
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [{
        name: 'hello',
        description: 'Say hello',
        inputSchema: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Name to greet' }
          }
        }
      }]
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      if (request.params.name === 'hello') {
        const name = request.params.arguments?.name || 'World';
        return {
          content: [{ type: 'text', text: `Hello, ${name}!` }]
        };
      }
      throw new Error(`Unknown tool: ${request.params.name}`);
    });
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

new MyMCPServer().run();
```

## Best Practices

### 1. Security
- Never store API keys in public repositories
- Use environment variables for sensitive data
- Regularly update MCP servers

### 2. Performance
- Use autoApprove for frequently used safe operations
- Disable unused servers (disabled: true)
- Monitor resource consumption

### 3. Debugging
- Enable detailed logging in development mode
- Use simple test commands to verify connectivity
- Check version compatibility

## Available MCP Servers

### Official Servers
- **Filesystem**: File operations and directory browsing
- **Postgres**: PostgreSQL database operations
- **Slack**: Slack messaging and channel management
- **GitHub**: Repository management and issue tracking
- **Google Drive**: File management in Google Drive

### Community Servers
- **Weather**: Weather information from OpenWeatherMap
- **Database**: Various database connectors
- **AWS**: AWS service integrations
- **Docker**: Container management

### Enterprise Servers
- **SAP Fiori MCP**: SAP Fiori Elements development
- **Microsoft Graph**: Microsoft 365 integration
- **Salesforce**: Salesforce CRM operations

## Advanced Configuration

### Environment Variables
```json
{
  "mcpServers": {
    "advanced-server": {
      "command": "node",
      "args": ["server.js"],
      "env": {
        "NODE_ENV": "production",
        "API_ENDPOINT": "https://api.example.com",
        "DEBUG": "mcp:*"
      }
    }
  }
}
```

### Auto-approval Configuration
```json
{
  "mcpServers": {
    "trusted-server": {
      "command": "trusted-server",
      "autoApprove": [
        "read_file",
        "list_directory",
        "search_documents"
      ]
    }
  }
}
```

## Integration Examples

### With VS Code Extensions
```javascript
// Use MCP tools in VS Code extension development
const result = await vscode.commands.executeCommand(
  'cline.useMcpTool',
  {
    serverName: 'fiori-mcp',
    toolName: 'generate_app',
    arguments: { projectPath: workspaceFolder.uri.fsPath }
  }
);
```

### With Node.js Applications
```javascript
const mcp = require('@modelcontextprotocol/sdk/client');

const client = new mcp.Client();
await client.connect('stdio', { command: 'my-mcp-server' });

const result = await client.request('tools/call', {
  name: 'my-tool',
  arguments: { param: 'value' }
});
```

## Useful Resources

- **MCP SDK Documentation**: https://modelcontextprotocol.io/
- **MCP Server Examples**: https://github.com/modelcontextprotocol/servers
- **SAP Fiori MCP**: https://www.npmjs.com/package/@sap-ux/fiori-mcp-server
- **Cline Documentation**: https://docs.cline.bot/
- **Claude Desktop**: https://claude.ai/

## Frequently Asked Questions

### Q: How do I know if an MCP server is working?
A: Check the "Connected MCP Servers" section in Cline or try calling a simple tool like `ping` or listing available tools.

### Q: Can I use multiple MCP servers simultaneously?
A: Yes, you can configure and use multiple MCP servers. Each server runs independently.

### Q: Are MCP servers secure?
A: MCP servers run locally and only have access to what you explicitly configure. Always review server code and use trusted sources.

### Q: Can I create custom tools for existing servers?
A: This depends on the server implementation. Some servers allow plugins, while others require modification of the server code.

### Q: How do I update an MCP server?
A: For npm-based servers, use `npm update -g package-name`. For other servers, follow their specific update instructions.

## Conclusion

MCP servers significantly extend AI assistant capabilities, enabling interaction with external systems and APIs. Proper configuration and usage of MCP servers can substantially improve development productivity, especially when working with specialized platforms like SAP.

The key to successful MCP integration is understanding your workflow requirements and choosing the right servers to support them. Start with simple servers and gradually add more complex integrations as needed.
