# MCP Servers Guide

## Quick Start

This repository contains comprehensive guides for using MCP (Model Context Protocol) servers with Cline and Claude Dev.

## 📚 Documentation

### 🇺🇸 English
- **[Complete MCP Usage Guide](MCP_Usage_Guide_EN.md)** - Comprehensive guide covering installation, configuration, and usage
- Includes examples for SAP Fiori MCP, weather servers, and custom server creation

### 🇷🇺 Russian
- **[Complete MCP Guide (Russian)](MCP_Usage_Guide_RU.md)** - Detailed guide on installation, configuration, and usage
- Includes examples for SAP Fiori MCP, weather servers, and custom server creation

## 🚀 Quick Setup

### 1. Find your configuration file

**Cline (VS Code)**:
```
%APPDATA%\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json
```

**Claude Desktop**:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

### 2. Basic configuration structure

```json
{
  "mcpServers": {
    "server-name": {
      "command": "server-command",
      "args": [],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## 🛠️ Currently Working Servers

### ✅ SAP Fiori MCP Server
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

**Features:**
- Generate SAP Fiori Elements applications
- Create List Report and Object Page
- Search Fiori documentation
- CAP project integration

### ⚠️ Upstash Context7 (Requires API Key)
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

## 📦 Installation

### SAP Fiori MCP Server
```bash
# Install server
npm install -g @sap-ux/fiori-mcp-server

# Install Windows dependencies
npm install -g @lancedb/lancedb-win32-x64-msvc
```

## 💡 Usage Examples

### List available tools
```javascript
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "list_functionality",
  arguments: {"appPath": "./my-project"}
})
```

### Search documentation
```javascript
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "search_docs",
  arguments: {
    "query": "List Report configuration",
    "maxResults": 5
  }
})
```

## 🔧 Troubleshooting

### Common issues:
1. **"Not connected"** - Check server installation and configuration
2. **Dependency errors** - Run `npm cache clean --force` and reinstall
3. **Windows issues** - Install Visual C++ Redistributable

### Check server status:
```javascript
use_mcp_tool({
  server_name: "your-server-name",
  tool_name: "ping", // or any available tool
  arguments: {}
})
```

## 🔗 Useful Links

- [MCP SDK Documentation](https://modelcontextprotocol.io/)
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)
- [SAP Fiori MCP Server](https://www.npmjs.com/package/@sap-ux/fiori-mcp-server)
- [Cline Documentation](https://docs.cline.bot/)

## 📝 Contributing

Feel free to contribute improvements to these guides or add examples for additional MCP servers.

## 📄 License

This documentation is provided as-is for educational and reference purposes.
