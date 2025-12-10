# Руководство по использованию MCP серверов в Cline/Claude Dev

## Что такое MCP (Model Context Protocol)

MCP (Model Context Protocol) — это протокол, который позволяет AI ассистентам подключаться к внешним серверам и использовать дополнительные инструменты и ресурсы. Это расширяет возможности AI, позволяя ему работать с внешними API, базами данных, сервисами и специализированными инструментами.

## Где найти настройки MCP

### В Cline (VS Code Extension)
Файл конфигурации находится по пути:
```
%APPDATA%\Roaming\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json
```

### В Claude Desktop App
Файл конфигурации находится по пути:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## Структура конфигурации MCP

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

### Параметры конфигурации:
- **command**: Команда для запуска MCP сервера
- **args**: Аргументы командной строки
- **env**: Переменные окружения (API ключи, токены)
- **disabled**: true/false - включен/выключен сервер
- **autoApprove**: Список действий для автоматического одобрения

## Примеры настройки популярных MCP серверов

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

**Что предоставляет:**
- Генерация SAP Fiori Elements приложений
- Создание List Report и Object Page
- Работа с CAP проектами
- Поиск по документации SAP Fiori

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

**Что предоставляет:**
- Векторный поиск и семантический поиск
- Работа с контекстными данными
- Индексация документов

### 3. Weather MCP Server (пример)
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

## Как использовать MCP серверы в коде

### 1. Использование инструментов MCP
```javascript
// В Cline используйте команду use_mcp_tool
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "list_functionality",
  arguments: {
    "appPath": "/path/to/your/project"
  }
})
```

### 2. Доступ к ресурсам MCP
```javascript
// Доступ к ресурсам через access_mcp_resource
access_mcp_resource({
  server_name: "weather",
  uri: "weather://London/current"
})
```

## Практические примеры использования

### Создание Fiori приложения
```javascript
// 1. Проверить доступные функции
use_mcp_tool({
  server_name: "fiori-mcp",
  tool_name: "list_functionality",
  arguments: {"appPath": "./my-project"}
})

// 2. Генерировать приложение
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

### Поиск по документации
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

## Установка популярных MCP серверов

### SAP Fiori MCP Server
```bash
npm install -g @sap-ux/fiori-mcp-server
```

### Установка зависимостей для Windows
```bash
npm install -g @lancedb/lancedb-win32-x64-msvc
```

### Использование через NPX (альтернатива)
```json
{
  "command": "cmd",
  "args": ["/c", "npx", "-y", "@sap-ux/fiori-mcp-server@latest"]
}
```

## Отладка и диагностика проблем

### Проверка статуса сервера
1. Убедитесь, что сервер добавлен в конфигурацию
2. Проверьте, что команда запускается из командной строки
3. Проверьте логи в VS Code Developer Tools (F12)

### Общие проблемы и решения

#### 1. "Not connected" ошибка
- Проверьте правильность команды и аргументов
- Убедитесь, что сервер установлен глобально
- Проверьте переменные окружения

#### 2. Проблемы с зависимостями
```bash
# Очистить кэш npm
npm cache clean --force

# Переустановить пакет
npm uninstall -g package-name
npm install -g package-name
```

#### 3. Windows-специфичные проблемы
- Установите Microsoft C++ Redistributable
- Используйте PowerShell или Command Prompt как администратор
- Проверьте переменную PATH

### Проверка работоспособности
```javascript
// Тест подключения (замените на ваш сервер)
use_mcp_tool({
  server_name: "your-server-name",
  tool_name: "ping", // или любой доступный инструмент
  arguments: {}
})
```

## Создание собственного MCP сервера

### Базовая структура проекта
```
my-mcp-server/
├── package.json
├── tsconfig.json
└── src/
    └── index.ts
```

### Пример простого MCP сервера
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

## Лучшие практики

### 1. Безопасность
- Никогда не храните API ключи в публичных репозиториях
- Используйте переменные окружения для чувствительных данных
- Регулярно обновляйте MCP серверы

### 2. Производительность
- Используйте autoApprove для часто используемых безопасных операций
- Отключайте неиспользуемые серверы (disabled: true)
- Мониторьте потребление ресурсов

### 3. Отладка
- Включите подробное логирование в development режиме
- Используйте простые тестовые команды для проверки подключения
- Проверяйте совместимость версий

## Полезные ресурсы

- **MCP SDK Documentation**: https://modelcontextprotocol.io/
- **Примеры MCP серверов**: https://github.com/modelcontextprotocol/servers
- **SAP Fiori MCP**: https://www.npmjs.com/package/@sap-ux/fiori-mcp-server
- **Cline Documentation**: https://docs.cline.bot/

## Заключение

MCP серверы значительно расширяют возможности AI ассистентов, позволяя им взаимодействовать с внешними системами и API. Правильная настройка и использование MCP серверов может существенно повысить продуктивность разработки, особенно при работе со специализированными платформами как SAP.
