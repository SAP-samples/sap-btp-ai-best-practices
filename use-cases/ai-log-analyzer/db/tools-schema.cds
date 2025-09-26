namespace sap.btp.ai.tools;

using { managed } from '@sap/cds/common';

type JSONSchemaProperties: LargeString;

type JSONSchemaType : String enum {
    object = 'object';
    array = 'array';
}

/**
 * JSON Schema for the tool's parameters
 */
type JSONSchema {

    // JSON Schema type, e.g. "object"
    type: JSONSchemaType not null;

    // Tool-specific parameters. Holds JSON Schema as String. Used for validation.
    properties: JSONSchemaProperties not null;

    required: Array of String not null;
}

/**
 * Optional hints about tool behavior
 */
type Annotations {

    // Human-readable title for the tool
    title: String; 

    // If true, the tool does not modify its environment
    readOnlyHint: Boolean;

    // If true, the tool may perform destructive updates
    destructiveHint: Boolean;

    // If true, repeated calls with same args have no additional effect
    idempotentHint: Boolean;

    // If true, tool interacts with external entities
    openWorldHint: Boolean;
}

/**
 * Tools are a powerful primitive in the Model Context Protocol (MCP) that enable servers to expose executable functionality to clients.
 * https://modelcontextprotocol.io/docs/concepts/tools
 */
entity Tools : managed {

    // Human-readable title for the tool
    key name: String not null; 

    // Human-readable description
    description: String not null;

    // JSON Schema for the tool's parameters
    inputSchema: JSONSchema not null;

    // JSON Schema for the tool's parameters
    outputSchema: JSONSchema not null;

    // Optional hints about tool behavior
    annotations: Annotations;
}